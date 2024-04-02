import io
from typing import Generator, Literal, Optional, Any, Callable
from contextlib import contextmanager
from unittest.mock import create_autospec
from httpx import Response, Request
from openai import RateLimitError, APIError, APITimeoutError

from src.retry_on.logging import logging
from src.retry_on.retry import logger as retry_logger
from src.retry_on.retry import RetryConfig, TaskManager

#################################
# Utility functions and classes
#################################


class ExceptionSimulator:
    @staticmethod
    def create_rate_limit_error() -> RateLimitError:
        mock_response: Response = create_autospec(Response, instance=True)
        mock_response.status_code: Literal[429] = 429  # type: ignore
        mock_response.json.return_value: dict[str, str] = {  # type: ignore
            "error": "Rate limit exceeded"
        }
        body: dict[str, str] | None = None

        return RateLimitError(
            message="Rate limit exceeded", body=body, response=mock_response
        )

    @staticmethod
    def create_api_error() -> APIError:
        mock_request: Request = create_autospec(Request, instance=True)
        mock_request.url: str = "http://foo.com"  # type: ignore
        mock_request.method: str = "POST"  # type: ignore
        mock_request.headers: dict[str, str] = {  # type: ignore
            "Content-Type": "application/json",
            "Authorization": "Bearer <BEARER_TOKEN>",
        }
        mock_request.content: bytes = (  # type: ignore
            b'{"model": "tmodel", "prompt": "Any prompt", "max_tokens": 5}'
        )
        body: dict[str, str] | None = None

        return APIError(message="API error occurred", request=mock_request, body=body)

    @staticmethod
    def create_api_timeout_error() -> APITimeoutError:
        mock_request: Request = create_autospec(Response, instance=True)
        mock_request.status_code: Literal[401] = 401  # type: ignore
        mock_request.method: str = "GET"  # type: ignore
        mock_request.url: str = "http://foo.com"  # type: ignore

        return APITimeoutError(request=mock_request)


class NetworkSimulator:
    def __init__(self):
        self.attempt_count: int = 0

    async def flaky_network_call(self) -> Optional[Literal["Success"]]:
        self.attempt_count += 1
        if self.attempt_count <= 2:
            raise ConnectionError("Simulated network failure")
        return "Success"


@contextmanager
def capture_logging(logger: logging.Logger = retry_logger) -> Generator[Any, Any, Any]:
    default_retry_config: RetryConfig = RetryConfig()
    log_stream: io.StringIO = io.StringIO()
    handler: logging.StreamHandler = logging.StreamHandler(log_stream)
    formatter: logging.Formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    original_handlers: list[logging.Handler] = logger.handlers[:]
    logger.handlers: list[logging.Handler] = []  # type: ignore
    logger.addHandler(handler)
    original_level: int = logger.level
    logger.setLevel(default_retry_config.log_level)
    try:
        yield log_stream
    finally:
        # type: ignore
        logger.handlers: list[logging.Handler] = original_handlers  # type: ignore
        logger.setLevel(original_level)
        handler.flush()
        log_stream.flush()


def transient_fail_func(max_failures: int = 1) -> Callable:
    call_count: dict[str, int] = {"count": 0}

    def func(*args, **kwargs) -> Literal["Success"]:
        if call_count["count"] < max_failures:
            call_count["count"] += 1
            # Raise the custom exception instead of APIStatusError or similar
            raise ExceptionSimulator.create_rate_limit_error()
        return "Success"

    # Wrap the synchronous function to make it compatible with async contexts
    async def async_wrapper(*args, **kwargs) -> str:
        return func(*args, **kwargs)

    # Return the synchronous function directly for synchronous contexts
    # and the async wrapper for asynchronous contexts
    return async_wrapper if TaskManager.is_async(func) else func
