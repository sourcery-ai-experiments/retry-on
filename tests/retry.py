import asyncio
import io
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    NoReturn,
    Optional,
    Type,
    Union,
)
from unittest.mock import Mock, AsyncMock, MagicMock, create_autospec, patch

import pytest
from httpx import Request, Response
# from memory_profiler import memory_usage
from openai import APIError, APITimeoutError, RateLimitError
from typing_extensions import Literal

from src.retry_on.utilities.logging import logging
from src.retry_on.retry import RetryConfig, RetryContext, SharedSemaphore, TaskManager
from src.retry_on.retry import logger as retry_logger
from src.retry_on.retry import retry

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
    if TaskManager.is_async(func):
        return async_wrapper
    else:
        return func


########
# Tests
########

################
# Retry Config

def test_retry_config_validation():
    with pytest.raises(ValueError):
        RetryConfig(max_retries=-1)
    with pytest.raises(ValueError):
        RetryConfig(initial_delay=-0.1)
    with pytest.raises(ValueError):
        RetryConfig(max_delay=-0.1)
    with pytest.raises(ValueError):
        RetryConfig(jitter=-2)
    with pytest.raises(ValueError):
        RetryConfig(concurrency_limit=-1)
    with pytest.raises(ValueError):
        RetryConfig(custom_sequence="not_a_list", retry_pattern="custom_sequence")


# Test Data for Happy Path, Edge Cases, and Error Cases
initialization_data = [
    ({}, {'max_retries': 3, 'retry_pattern': 'controlled_flow'}),
    ({'max_retries': 5, 'retry_pattern': 'fixed'}, {'max_retries': 5, 'retry_pattern': 'fixed'}),
    ({'retry_pattern': 'linear', 'linear_delay': 1.0}, {'retry_pattern': 'linear', 'linear_delay': 1.0})
]

setter_test_data = [
    ('max_retries', 5, None),
    ('initial_delay', 0, None),
    ('max_delay', 70, None),
    ('jitter', 0.5, None),
    ('burst_capacity', 5, None),
    ('rate_limit', 1.0, None),
    ('retry_pattern', 'exponential', None),
    ('linear_delay', 1.0, None),
    ('fixed_delay', 5.0, None),
    ('custom_sequence', [1.0, 2.0, 3.0], None),
    ('concurrency_limit', 10, None),
    ('function_timeout', 30.0, None),
    ('callback_timeout', 15.0, None),
    ('log_level', logging.INFO, None),
    # Error cases
    ('max_retries', -1, ValueError),
    ('initial_delay', -1, ValueError),
    ('max_delay', -1, ValueError),
    ('jitter', -0.1, ValueError),
    ('burst_capacity', -1, ValueError),
    ('rate_limit', -1, ValueError),
    ('retry_pattern', 'unsupported_pattern', ValueError),
    ('linear_delay', -1, ValueError),
    ('fixed_delay', -1, ValueError),
    ('custom_sequence', ['invalid'], ValueError),
    ('concurrency_limit', -1, ValueError),
    ('function_timeout', -1, ValueError),
    ('callback_timeout', -1, ValueError),
    ('log_level', 51, ValueError)
]

# Initialization Tests
@pytest.mark.parametrize("kwargs,expected", initialization_data)
def test_initialization(kwargs, expected):
    retry_config = RetryConfig(**kwargs)
    for key, value in expected.items():
        assert getattr(retry_config, key) == value, f"{key} does not match expected value."

# Setter and Validation Tests
@pytest.mark.parametrize("attribute,value,exception", setter_test_data)
def test_setters_and_validations(attribute, value, exception):
    retry_config = RetryConfig()
    if exception:
        with pytest.raises(exception):
            setattr(retry_config, attribute, value)
    else:
        setattr(retry_config, attribute, value)
        assert getattr(retry_config, attribute) == value, f"{attribute} was not set correctly."

# Custom Sequence Handling Tests
def test_valid_custom_sequence():
    retry_config = RetryConfig(custom_sequence=[1, 2, 3])
    assert retry_config.custom_sequence == [1.0, 2.0, 3.0], "Custom sequence not handled correctly."

def test_invalid_custom_sequence_elements():
    retry_config = RetryConfig(custom_sequence=[1, 'invalid', 3])
    assert retry_config.custom_sequence == [1.0, 3.0], "Custom sequence not correctly filtered."


# Callback Validation Tests
def test_valid_callback():
    retry_config = RetryConfig(on_retry_callback=lambda x: x)
    assert callable(retry_config.on_retry_callback), "Callback is not callable."

def test_invalid_callback():
    with pytest.raises(TypeError):
        RetryConfig(on_retry_callback="not_callable")

def test_retry_config_unsupported_retry_pattern() -> None:
    config: RetryConfig = RetryConfig()
    with pytest.raises(ValueError):
        config.retry_pattern = "unsupported_pattern"


def test_retry_config_with_dynamic_changes() -> None:
    config: RetryConfig = RetryConfig(
        max_retries=2, initial_delay=1.0, max_delay=10.0, jitter=0.1, retry_pattern="fixed"
    )
    config.max_retries = 5
    config.initial_delay = 2.0
    config.max_delay = 20.0
    config.jitter = 0.2
    config.retry_pattern = "custom_sequence"
    config.custom_sequence = [1, 2, 3]
    
    assert config.max_retries == 5
    assert config.initial_delay == 2.0
    assert config.max_delay == 20.0
    assert config.jitter == 0.2
    assert config.retry_pattern == "custom_sequence"
    assert config.custom_sequence == [1, 2, 3]


###################
# Retry Mechanism

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "should_raise, expected_attempts, test_scenario",
    [
        (False, 1, "immediate_success_async"),
        (False, 2, "success_before_max_retries_async"),
        (True, 3, "failure_at_max_retries_async"),
    ],
)
async def test_async_retry_behavior(
    should_raise, expected_attempts, test_scenario
) -> None:
    simulator: ExceptionSimulator = ExceptionSimulator()
    side_effects: list[Exception] = [
        simulator.create_rate_limit_error() for _ in range(expected_attempts - 1)
    ]
    if should_raise:
        side_effects.append(simulator.create_rate_limit_error())
    else:
        side_effects.append(True)  # type: ignore

    func: AsyncMock = AsyncMock(side_effect=side_effects)

    @retry(
        exceptions=(RateLimitError, APIError, APITimeoutError),
        max_retries=expected_attempts - 1,
    )
    async def async_test_func() -> NoReturn:
        await func()

    if should_raise:
        with pytest.raises((RateLimitError, APIError, APITimeoutError)):
            await async_test_func()
    else:
        await async_test_func()

    assert func.call_count == expected_attempts


@pytest.mark.parametrize(
    "should_raise, expected_attempts, test_scenario",
    [
        (False, 1, "immediate_success"),
        (False, 2, "success_before_max_retries"),
        (True, 3, "failure_at_max_retries"),
    ],
)
def test_sync_retry_behavior(should_raise, expected_attempts, test_scenario) -> None:
    simulator: ExceptionSimulator = ExceptionSimulator()
    side_effects: list[Exception] = [
        simulator.create_rate_limit_error() for _ in range(expected_attempts - 1)
    ]
    if should_raise:
        side_effects.append(simulator.create_rate_limit_error())
    else:
        side_effects.append(True)  # type: ignore

    func: MagicMock = MagicMock(side_effect=side_effects)

    @retry(exceptions=RateLimitError, max_retries=expected_attempts - 1)
    def sync_test_func() -> None:
        func()

    if should_raise:
        with pytest.raises(RateLimitError):
            sync_test_func()
    else:
        sync_test_func()

    assert func.call_count == expected_attempts, f"{test_scenario}: Expected {expected_attempts} calls, got {func.call_count}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "should_raise, max_attempts, test_scenario",
    [
        (False, 1, ExceptionSimulator.create_api_error),
        (False, 2, ExceptionSimulator.create_api_timeout_error),
        (True, 3, ExceptionSimulator.create_rate_limit_error),
    ],
)
async def test_retry_on_api_error_async(
    should_raise, max_attempts, test_scenario
) -> None:
    side_effects: list[Exception | bool] = [
        test_scenario() for _ in range(max_attempts)
    ]
    if should_raise:
        side_effects.append(test_scenario())
    else:
        side_effects.append(True)

    func: AsyncMock = AsyncMock(side_effect=side_effects)

    @retry(exceptions=(RateLimitError, APIError, APITimeoutError))
    async def async_test_func() -> NoReturn:
        await func()

    if should_raise:
        with pytest.raises((RateLimitError, APIError, APITimeoutError)):
            await async_test_func()
    else:
        await async_test_func()

    assert (
        func.call_count == max_attempts + 1
    ), f"Expected {max_attempts} attempts, got {func.call_count}"


@pytest.mark.parametrize(
    "should_raise, max_attempts, test_scenario",
    [
        (False, 1, ExceptionSimulator.create_api_error),
        (False, 2, ExceptionSimulator.create_api_timeout_error), 
        (True, 3, ExceptionSimulator.create_rate_limit_error)
    ]
)
def test_retry_on_api_error_sync(
    should_raise, max_attempts, test_scenario
):

    side_effects = [test_scenario() for _ in range(max_attempts)]
    if should_raise:
        side_effects.append(test_scenario())
    else:
        side_effects.append(True)

    func = Mock(side_effect=side_effects)

    @retry(exceptions=(RateLimitError, APIError, APITimeoutError)) 
    def sync_test_func():
        func()

    if should_raise:
        with pytest.raises((RateLimitError, APIError, APITimeoutError)):
            sync_test_func()
    else:
        sync_test_func()

    assert func.call_count == max_attempts + 1


@pytest.mark.asyncio
async def test_no_retry_on_success() -> None:
    func: AsyncMock = AsyncMock(return_value=Literal["Success"])

    @retry(exceptions=RateLimitError, max_retries=2, initial_delay=0.1, jitter=0)
    async def async_test_func() -> Literal["Success"]:
        return await func()

    result: Literal["Success"] = await async_test_func()
    assert result == Literal["Success"]
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_max_retry_exceeded() -> None:
    func: AsyncMock = AsyncMock(
        side_effect=ExceptionSimulator.create_rate_limit_error()
    )
    max_retries: int = 3
    attempts: int = 1 + max_retries  # 1 initial attempt + 3 retries

    @retry(
        exceptions=RateLimitError, max_retries=max_retries, initial_delay=0.1, jitter=0
    )
    async def async_test_func() -> Optional[Exception]:
        await func()

    with pytest.raises(RateLimitError):
        await async_test_func()

    assert func.call_count == attempts


@pytest.mark.asyncio
async def test_integration_retry_mechanism() -> None:
    async_call_count: int = 0
    sync_call_count: int = 0

    async def async_operation() -> Literal["Async success"]:
        nonlocal async_call_count
        async_call_count += 1
        if async_call_count < 3:
            raise ValueError("Async failure")
        return "Async success"

    def sync_operation() -> Literal["Sync success"]:
        nonlocal sync_call_count
        sync_call_count += 1
        if sync_call_count < 2:
            raise ValueError("Sync failure")
        return "Sync success"

    @retry(ValueError, max_retries=4, initial_delay=0.1, jitter=0)
    async def async_wrapper() -> Optional[Literal["Async success"]]:
        return await async_operation()

    @retry(ValueError, max_retries=3, initial_delay=0.1, jitter=0)
    def sync_wrapper() -> Optional[Literal["Sync success"]]:
        return sync_operation()

    async_result: Optional[Literal["Async success"]] = await async_wrapper()
    sync_result: Optional[Literal["Sync success"]] = sync_wrapper()

    assert async_result == "Async success"
    assert async_call_count == 3
    assert sync_result == "Sync success"
    assert sync_call_count == 2


@pytest.mark.asyncio
async def test_retry_on_network_failure() -> None:
    network_simulator: NetworkSimulator = NetworkSimulator()

    @retry(ConnectionError, max_retries=3, initial_delay=0.1, jitter=0)
    async def make_network_call() -> Optional[Literal["Success"]]:
        return await network_simulator.flaky_network_call()

    result: Optional[Literal["Success"]] = await make_network_call()
    assert result == "Success", "The network call should succeed after retries"
    assert (
        network_simulator.attempt_count == 3
    ), "The function should retry exactly 2 times before succeeding"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config,expected_attempts,exception_generator",
    [
        (
            {
                "max_retries": 2,
                "initial_delay": 0.1,
                "retry_pattern": "fixed",
                "fixed_delay": 0.1,
            },
            2,
            ExceptionSimulator.create_rate_limit_error,
        ),
        (
            {"max_retries": 3, "initial_delay": 0.1, "retry_pattern": "exponential"},
            3,
            ExceptionSimulator.create_api_timeout_error,
        ),
        (
            {
                "max_retries": 4,
                "custom_sequence": [0.1, 0.2, 0.4],
                "retry_pattern": "custom_sequence",
            },
            4,
            ExceptionSimulator.create_api_error,
        ),
    ],
)
async def test_retry_with_custom_config(
    config, expected_attempts, exception_generator
) -> None:
    side_effects: list[Exception] = [
        exception_generator() for _ in range(expected_attempts - 1)
    ]
    side_effects.append(True)  # type: ignore # Assume success on last attempt

    func: AsyncMock = AsyncMock(side_effect=side_effects)

    @retry(exceptions=(RateLimitError, APIError, APITimeoutError), **config)
    async def async_test_func_with_custom_config() -> None:
        await func()

    await async_test_func_with_custom_config()

    assert func.call_count == expected_attempts


######################
# Callback Functions

@pytest.mark.parametrize("async_func", [False, True])
def test_callback_after_retry_failure_with_sync_callback(async_func):
    call_count = {"main": 0, "callback": 0}

    def callback_function(exception: Type[BaseException]):
        call_count["callback"] += 1

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=callback_function)
    async def async_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=callback_function)
    def sync_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    main_function = async_main_function if async_func else sync_main_function

    # Execute the decorated function and verify behavior
    if async_func:
        with pytest.raises(ValueError):
            asyncio.run(main_function())
    else:
        with pytest.raises(ValueError):
            main_function()

    assert call_count["main"] == 3  # Initial call + 2 retries
    assert call_count["callback"] == 2  # Callback after each retry


@pytest.mark.parametrize("async_func", [False, True])
@pytest.mark.asyncio
async def test_callback_after_retry_failure_with_async_callback(async_func):
    call_count = {"main": 0, "callback": 0}

    async def async_callback_function(exception: Type[BaseException]) -> None:
        call_count["callback"] += 1

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=async_callback_function)
    async def async_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=async_callback_function)
    def sync_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    main_function = async_main_function if async_func else sync_main_function

    # Execute the decorated function and verify behavior
    if async_func:
        with pytest.raises(ValueError):
            await main_function()
    else:
        # Special handling to execute an async callback from a sync function context
        with pytest.raises(ValueError):
            def sync_wrapper():
                asyncio.run(main_function())
            sync_wrapper()

    assert call_count["main"] == 3  # Initial call + 2 retries
    assert call_count["callback"] == 2  # Callback after each retry


##################
# Retry Strategy

@pytest.mark.parametrize(
    "max_retries, jitter, initial_delay, max_delay, expected, test_id",
    [
        (0, 0.0, 1.0, 10.0, [], "no_retries"),
        (1, 0.0, 1.0, 10.0, [1.0], "single_retry_no_jitter"),
        (2, 0.0, 1.0, 10.0, [1.0, 2.0], "two_retries_no_jitter"),
        (3, 0.5, 1.0, 20.0, [1.0, 3.125, 15.625], "three_retries_with_jitter"),
        (3, 0.0, 0.0, 10.0, [0.0, 0.0, 0.0], "three_retries_no_initial_delay"),
        (3, 0.5, 1.0, 3.0, [1.0, 3.0, 3.0], "three_retries_with_max_delay"),
    ],
)
def test_retry_pattern_default_backoff(
    max_retries, jitter, initial_delay, max_delay, expected, test_id
):
    config = RetryConfig(
        max_retries=max_retries,
        jitter=jitter,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retry_pattern="exponential",
    )
    retry_context = RetryContext(Exception, config)

    def mock_random_uniform(a, b):
        return (a + b) / 2

    with patch("random.uniform", mock_random_uniform):
        result = retry_context.retry_strategy.exponential()
        assert result == expected, f"[{test_id}] Expected {expected} but got {result}"


@pytest.mark.parametrize(
    "max_retries, jitter, initial_delay, max_delay, expected, test_id",
    [
        (3, 0.0, 1.0, 10.0, [1.0, 2.0, 8.0], "happy-path-no-jitter"),
        (1, 0.0, 2.0, 10.0, [2.0], "edge-case-no-jitter"),
        (0, 0.0, 1.0, 10.0, [], "edge-case-no-retries-no-jitter"),
        (
            5,
            0.0,
            0.1,
            0.5,
            [0.1, 0.2, 0.5, 0.5, 0.5],
            "happy-path-max-delay-reached-no-jitter",
        ),
        (
            2,
            0.0,
            5.0,
            5.0,
            [5.0, 5.0],
            "edge-case-initial-delay-equals-max-delay-no-jitter",
        ),
        # Assuming negative values are error cases
        (-1, 0.5, 1.0, 10.0, ValueError, "error-case-negative-max-retries"),
        (3, -0.5, 1.0, 10.0, ValueError, "error-case-negative-jitter"),
        (3, 0.5, -1.0, 10.0, ValueError, "error-case-negative-initial-delay"),
        (3, 0.5, 1.0, -10.0, ValueError, "error-case-negative-max-delay"),
    ],
)
def test_retry_pattern_default_backoff_without_jitter(
    max_retries: int,
    jitter: float,
    initial_delay: float,
    max_delay: float,
    expected: Union[list, type],
    test_id: str,
) -> None:
    if isinstance(expected, type) and expected is ValueError:
        with pytest.raises(expected):
            _ = RetryConfig(
                max_retries=max_retries,
                jitter=jitter,
                initial_delay=initial_delay,
                max_delay=max_delay,
                retry_pattern="exponential",
            )
    elif isinstance(expected, list):
        config = RetryConfig(
            max_retries=max_retries,
            jitter=jitter,
            initial_delay=initial_delay,
            max_delay=max_delay,
            retry_pattern="exponential",
        )
        retry_context = RetryContext(Exception, config)
        result = retry_context.retry_strategy.exponential()
        assert result == expected, f"[{test_id}] Expected {expected} but got {result}"


@pytest.mark.parametrize(
    "rate_limit, burst_capacity, max_retries, jitter, expected_intervals, test_id",
    [
        # Happy path tests
        (0.5, 3, 5, 0.25, [0, 0, 0, 2.0, 4.0], "happy-path-low-retries"),
        (1.0, 2, 4, 0.0, [0, 0, 1.0, 2.0], "happy-path-no-jitter"),
        (2.0, 1, 3, 0.5, [0, 0.5, 1.0], "happy-path-high-rate-limit"),
        # Edge cases
        (0.5, 0, 3, 0.25, [2.0, 4.0, 6.0], "edge-case-zero-burst-capacity"),
        (0.1, 5, 5, 0.25, [0, 0, 0, 0, 0], "edge-case-burst-equals-retries"),
        (0.5, 5, 3, 0.25, [0, 0, 0], "edge-case-burst-exceeds-retries"),
    ],
    ids=str,
)
def test_retry_pattern_controlled_flow(
    rate_limit: float,
    burst_capacity: int,
    max_retries: int,
    jitter: float,
    expected_intervals: Union[list, type],
    test_id: str,
):
    config = RetryConfig(
        max_retries=max_retries,
        jitter=jitter,
        rate_limit=rate_limit,
        burst_capacity=burst_capacity,
        retry_pattern="controlled_flow",
    )
    retry_context = RetryContext(Exception, config)

    def mock_random() -> float:
        return 0.5

    with patch("random.random", mock_random):
        if isinstance(expected_intervals, type) and issubclass(
            expected_intervals, Exception
        ):
            with pytest.raises(expected_intervals):
                retry_context.retry_strategy.controlled_flow()
        else:
            intervals = retry_context.retry_strategy.controlled_flow()

    if not isinstance(expected_intervals, type):
        assert (
            intervals == expected_intervals
        ), f"[{test_id}] Expected {expected_intervals} but got {intervals}"


@pytest.mark.parametrize(
    "custom_sequence, expected, test_id",
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], "custom_sequence"),
    ],
)
def test_retry_pattern_custom_sequence(
    custom_sequence: list[float], expected: list[float], test_id: str
) -> None:
    config = RetryConfig(
        custom_sequence=custom_sequence,
        retry_pattern="custom_sequence",
    )
    retry_context = RetryContext(Exception, config)
    result = retry_context.retry_strategy.custom_sequence()
    assert result == expected, f"[{test_id}] Expected {expected} but got {result}"


@pytest.mark.asyncio
async def test_retry_pattern_application() -> None:
    call_counts: dict[str, int] = {"fixed": 0, "exponential": 0, "custom_sequence": 0}
    max_retries: int = 3
    expected_call_counts: int = max_retries + 1
    
    async def failing_task() -> Optional[None]:
        raise ValueError("Simulated task failure")

    @retry(ValueError, max_retries=max_retries, initial_delay=0.1, retry_pattern="fixed", fixed_delay=0.1)
    async def fixed_retry_task() -> Optional[None]:
        call_counts["fixed"] += 1
        await failing_task()

    @retry(ValueError, max_retries=max_retries, initial_delay=0.1, retry_pattern="exponential")
    async def exponential_retry_task() -> Optional[None]:
        call_counts["exponential"] += 1
        await failing_task()

    @retry(
        ValueError,
        max_retries=max_retries,
        custom_sequence=[0.1, 0.2, 0.4],
        retry_pattern="custom_sequence",
    )
    async def custom_sequence_retry_task() -> Optional[None]:
        call_counts["custom_sequence"] += 1
        await failing_task()

    with pytest.raises(ValueError):
        await fixed_retry_task()
    assert (
        call_counts["fixed"] == 4
    ), f"Fixed retry pattern did not execute the expected number of times (expected: {expected_call_counts}, actual: {call_counts['fixed']})"

    with pytest.raises(ValueError):
        await exponential_retry_task()
    assert (
        call_counts["exponential"] == 4
    ), f"Exponential retry pattern did not execute the expected number of times (expected: {expected_call_counts}, actual: {call_counts['exponential']})"

    with pytest.raises(ValueError):
        await custom_sequence_retry_task()
    assert (
        call_counts["custom_sequence"] == 4
    ), f"Custom sequence retry pattern did not execute the expected number of times (expected: {expected_call_counts}, actual: {call_counts['custom_sequence']})"


###############
# Concurrency

@pytest.mark.asyncio
async def test_async_func_concurrency_limit_max_retries_exceeded() -> None:
    concurrent_executions: int = 0
    max_concurrent_executions: int = 0
    concurrency_limit: int = 2
    max_retries: int = 5
    initial_delay: float = 0.1

    @retry(
        exceptions=(ExceptionSimulator.create_rate_limit_error().__class__,),
        max_retries=max_retries,
        initial_delay=initial_delay,
        concurrency_limit=concurrency_limit,
    )
    async def limited_func() -> NoReturn:
        nonlocal concurrent_executions, max_concurrent_executions
        concurrent_executions += 1
        max_concurrent_executions = max(
            max_concurrent_executions, concurrent_executions
        )
        await asyncio.sleep(0.01)  # Simulate operation that may fail
        concurrent_executions -= 1
        # Force a retry by raising the exception
        raise ExceptionSimulator.create_rate_limit_error()

    tasks: list = [limited_func() for _ in range(4)]
    await asyncio.gather(*tasks, return_exceptions=True)

    assert (
        concurrent_executions == 0
    ), "There should be no concurrent executions at the end."
    assert (
        max_concurrent_executions <= concurrency_limit
    ), (
        f"Concurrency limit of {concurrency_limit} exceeded. "
        f"Max concurrent executions: {max_concurrent_executions}"
    )


def test_sync_func_concurrency_limit_max_retries_exceeded() -> None:
    concurrent_executions: int = 0
    max_concurrent_executions: int = 0
    concurrency_limit = 2
    max_retries = 5
    initial_delay = 0.1

    @retry(
        exceptions=(Exception,),
        max_retries=max_retries,
        initial_delay=initial_delay,
        concurrency_limit=concurrency_limit
    )
    def limited_func():
        nonlocal concurrent_executions, max_concurrent_executions
        concurrent_executions += 1
        max_concurrent_executions = max(
            max_concurrent_executions,
            concurrent_executions
        )
        time.sleep(0.01)  # simulate operation
        concurrent_executions -= 1
        raise Exception("Forced retry")

    tasks = [limited_func for _ in range(4)]

    with pytest.raises(Exception):
        for task in tasks:
            task()

    assert concurrent_executions == 0, "Concurrent executions should be 0 at the end."
    assert max_concurrent_executions <= concurrency_limit, \
        f"Concurrent executions exceeded the limit of {concurrency_limit}."


@pytest.mark.asyncio
async def test_async_func_concurrency_limit_results() -> None:
    max_retries = 2
    worker_count = 4
    transient_failures = 1
    results: list = []

    sync_func = transient_fail_func(max_failures=transient_failures)

    @retry(Exception, max_retries=max_retries)
    async def decorated_func() -> str:
        return await asyncio.to_thread(sync_func)

    tasks = [decorated_func() for _ in range(worker_count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # await run_decorated()

    assert all(result == "Success" for result in results), "Not all functions succeeded as expected"


def test_sync_func_concurrency_limit_results() -> None:
    max_retries: int = 2
    worker_count: int = 4
    transient_failures: int = 1

    sync_func: Callable = transient_fail_func(max_failures=transient_failures)

    @retry(
        ExceptionSimulator.create_rate_limit_error().__class__, max_retries=max_retries
    )
    def decorated_func() -> str:
        return sync_func()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: list = [executor.submit(decorated_func) for _ in range(worker_count)]
        results: list = [future.result() for future in as_completed(futures)]

    assert all(
        result == "Success" for result in results
    ), "Not all functions succeeded as expected"


###############
# TaskManager

def test_is_async() -> None:
    async def async_func() -> None:
        pass

    def sync_func() -> None:
        pass

    assert (
        TaskManager.is_async(async_func) is True
    ), "Async function not correctly identified"
    assert (
        TaskManager.is_async(sync_func) is False
    ), "Sync function not correctly identified"


def test_is_async_variants() -> None:
    async def async_func() -> None:
        pass

    async def async_gen() -> AsyncGenerator:
        yield

    assert (
        TaskManager.is_async(async_func) is True
    ), "Coroutine functions should be recognized as async"
    assert (
        TaskManager.is_async(async_gen) is True
    ), "Async generator functions should be recognized as async"


###################
# SharedSemaphore

@pytest.mark.asyncio
async def test_shared_semaphore_async_lock() -> None:
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=2)
    try:
        # Attempt to acquire the semaphore twice to ensure it's properly released.
        async with semaphore.async_lock():
            pass  # First acquisition attempt
        async with semaphore.async_lock():
            pass  # Second acquisition attempt, should work if the semaphore was released
    except Exception as e:
        pytest.fail(f"Semaphore failed to lock/unlock properly: {str(e)}")


def test_shared_semaphore_executor_context() -> None:
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=2)
    with semaphore.executor_context() as executor:
        if executor is None:
            raise ValueError("Executor context should not be None")
        assert not executor._shutdown  # The executor should be active and not shut down


@pytest.mark.asyncio
async def test_concurrency_and_threading() -> None:
    shared_semaphore: SharedSemaphore = SharedSemaphore(max_workers=2)
    attempt_counter: int = 0
    max_retries: int = 2
    max_workers: int = 2

    @retry(Exception, max_retries=max_retries, initial_delay=0.1, concurrency_limit=2)
    async def failing_task() -> Optional[None]:
        nonlocal attempt_counter
        async with shared_semaphore.async_lock():
            attempt_counter += 1
            raise Exception("Simulated task failure")

    # Run the failing task in parallel to test concurrency limit
    tasks: list[Coroutine[Any, Any, None]] = [failing_task() for _ in range(4)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future] = [executor.submit(asyncio.run, task) for task in tasks]
        for future in futures:
            with pytest.raises(Exception):
                future.result()

    # number of tasks times (the retries plus the initial attempt)
    expected_attempts: int = len(tasks) * (max_retries + 1)

    # Verify that the semaphore correctly managed concurrency
    assert (
        attempt_counter == expected_attempts
    ), f"Expected {expected_attempts} attempts, got {attempt_counter}"


def test_shared_semaphore_concurrency() -> None:
    max_workers: int = 3
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=max_workers)
    barrier: threading.Barrier = threading.Barrier(max_workers + 1)
    results: list = []

    def task() -> None:
        with semaphore.executor_context() as executor:
            results.append(executor)
            barrier.wait()

    threads: list[threading.Thread] = [
        threading.Thread(target=task) for _ in range(max_workers)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()  # Wait for all threads to reach this point
    assert len(set(results)) == 1, "Executor context should be shared across threads"
    for thread in threads:
        thread.join()


@pytest.mark.asyncio
async def test_shared_semaphore_release_on_exception() -> None:
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=1)

    async def operation_with_exception() -> Optional[None]:
        async with semaphore.async_lock():
            raise ValueError("Operation failure")

    with pytest.raises(ValueError):
        await operation_with_exception()

    # If the semaphore was properly released, this operation should not block
    async with semaphore.async_lock():
        pass  # Operation successful

