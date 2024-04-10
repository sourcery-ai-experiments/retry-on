import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from typing import (AsyncGenerator, Callable, Generator)

from src.retry_on.logging import get_logger, logging

logger: logging.Logger = get_logger(__name__)


class ConcurrencyManager:
    def __init__(self, max_workers: int = 10) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._is_shutdown: bool = False
        self._async_max_workers: int = max_workers
        self._async_semaphore: asyncio.Semaphore =\
            asyncio.Semaphore(max_workers)
        self._executor: ThreadPoolExecutor =\
            ThreadPoolExecutor(max_workers=max_workers)
        self._shutdown_callbacks = []
        self._sync_value: int = max_workers

    def __repr__(self) -> str:
        return (
            "<ConcurrencyManager async_value={async_value}, "
            "async_max_workers={async_max_workers}, "
            "sync_max_workers={sync_max_workers}, "
            "sync_shutdown={sync_shutdown}>"
        ).format(
            async_value=self.async_value,
            async_max_workers=self.async_max_workers,
            sync_max_workers=self.sync_max_workers,
            sync_shutdown=self.is_sync_shutdown,
        )

    def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Adds a callback to be called when the semaphore is shut down."""
        self._shutdown_callbacks.append(callback)

    @property
    def async_value(self) -> int:
        return self._async_semaphore._value

    @property
    def sync_value(self) -> int:
        # This value isn't available through Python's public API.
        # ThreadPoolExecutor uses the C-optimized version of
        # SimpleQueue, which doesn't expose its internals.
        # A workaround is implemented in the `executor_context`.
        return self._sync_value

    @property
    def async_max_workers(self) -> int:
        return self._async_max_workers

    @property
    def sync_max_workers(self) -> int:
        return self._executor._max_workers

    @property
    def is_async_shutdown(self) -> bool:
        return self._executor._shutdown

    @property
    def is_sync_shutdown(self) -> bool:
        return self._executor._shutdown

    @is_sync_shutdown.setter
    def is_sync_shutdown(self, value: bool) -> None:
        if isinstance(value, bool):
            self._is_shutdown = value
        else:
            raise TypeError("is_shutdown must be a boolean")

    @contextmanager
    def executor_context(self) -> Generator:
        try:
            if self.is_sync_shutdown:
                raise RuntimeError("Semaphore is shut down")
            self._sync_value -= 1
            yield self._executor
        except Exception as exception:
            raise exception
        finally:
            self._sync_value += 1

    @asynccontextmanager
    async def async_lock(self) -> AsyncGenerator:
        if self.is_sync_shutdown:
            raise RuntimeError("Semaphore is shut down")
        await self._async_semaphore.acquire()
        try:
            yield self._async_semaphore
        except Exception as exception:
            raise exception
        finally:
            self._async_semaphore.release()

    def shutdown(self) -> None:
        if self.is_sync_shutdown:
            return
        self._executor.shutdown(wait=True)
        self.is_sync_shutdown = True
