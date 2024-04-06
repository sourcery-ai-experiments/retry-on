from functools import wraps
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    Type,
    Union
)
from src.retry_on.types import E, R

from src.retry_on.logging import get_logger, logging
from src.retry_on.config import RetryConfig
from src.retry_on.tasks import ConcurrencyManager, TaskManager
from src.retry_on.utils import validate_exceptions
from src.retry_on.context import RetryContext

logger: logging.Logger = get_logger(__name__)


def retry(
    exceptions: Union[Type[E], Tuple[Type[E], ...]], **kwargs: Any
) -> Callable:
    validate_exceptions(exceptions)
    retry_config: RetryConfig = RetryConfig(**kwargs)
    semaphore: Optional[ConcurrencyManager] = (
        ConcurrencyManager(max_workers=retry_config.concurrency_limit)
        if retry_config.concurrency_limit
        else None
    )

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            shared_semaphore: Optional[ConcurrencyManager] = (
                ConcurrencyManager(max_workers=retry_config.concurrency_limit)
                if retry_config.concurrency_limit
                else None
            )
            async with RetryContext(
                exceptions,
                retry_config,
                semaphore=shared_semaphore
            ) as context:
                return await context.retry_async_call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> R:
            with RetryContext(
                exceptions,
                retry_config,
                semaphore=semaphore
            ) as context:
                return context.retry_sync_call(func, *args, **kwargs)

        return async_wrapper if TaskManager.is_async(func) else sync_wrapper

    return decorator
