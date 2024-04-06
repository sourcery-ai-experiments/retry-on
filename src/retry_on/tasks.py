import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
import inspect
from contextlib import suppress
from typing import Any, Callable, AsyncGenerator, Generator, List, Optional, Union

from src.retry_on.logging import get_logger, logging
from src.retry_on.types import E, R

logger: logging.Logger = get_logger(__name__)


class ConcurrencyManager:
    def __init__(self, max_workers: int = 10) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._is_shutdown: bool = False
        self._async_max_workers: int = max_workers
        self._async_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_workers)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self._shutdown_callbacks = []
        self._sync_value: int = max_workers

    def __repr__(self) -> str:
        return (
            f"<ConcurrencyManager async_value={self.async_value}, "
            f"async_max_workers={self.async_max_workers}, "
            f"sync_max_workers={self.sync_max_workers}, "
            f"sync_shutdown={self.is_sync_shutdown}>"
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
    def executor_context(self) -> Generator[Any, Any, Any]:
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
    async def async_lock(self) -> AsyncGenerator[None, Any]:
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


class TaskManager:
    def __init__(self, semaphore: Optional[ConcurrencyManager] = None) -> None:
        self.semaphore: Optional[ConcurrencyManager] = semaphore
        self.tasks: List[Union[asyncio.Task, Future]] = []
        self.shutdown_required: bool = False
        if self.semaphore:
            self.semaphore.add_shutdown_callback(self.cancel_tasks)

    def signal_shutdown(self) -> None:
        """Signals that a shutdown is required."""
        self.shutdown_required = True

    async def submit_async_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Optional[R]:
        if self.semaphore and self.semaphore.is_sync_shutdown:
            logger.log(
                logging.WARNING,
                "Semaphore is shut down. Cannot submit new tasks."
            )
            return None

        async def async_task_wrapper(
            func: Callable,
            *args: Any,
            **kwargs: Any
        ) -> R:
            try:
                return await func(*args, **kwargs)
            except Exception as exception:
                raise exception
        task: asyncio.Task = None

        if self.semaphore:
            async with self.semaphore.async_lock():
                task: asyncio.Task = asyncio.create_task(
                    async_task_wrapper(func, *args, **kwargs)
                )
                self.tasks.append(task)
                return task
        else:
            task: asyncio.Task = asyncio.create_task(
                async_task_wrapper(func, *args, **kwargs)
            )
            self.tasks.append(task)
            return task        

    async def get_async_task_result(
        self,
        task: asyncio.Task,
        timeout: Optional[float] = None
    ) -> Optional[R]:
        try:
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError as task_exception:
            self._log_execution_exceeded_timeout(task_exception)
            task.cancel()
            return None
        except asyncio.CancelledError:
            logger.log(
                logging.DEBUG,
                f"Task {task.get_name()} was canceled."
            )
            return None

    def submit_sync_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> Any:
        if self.semaphore and self.semaphore.is_sync_shutdown:
            logger.log(
                logging.WARNING,
                "Semaphore is shut down. Cannot submit new tasks."
            )
            return None

        future: Future = Future()

        def sync_task_wrapper(func: Callable, *args, **kwargs) -> None:
            try:
                future.set_result(func(*args, **kwargs))
            except Exception as exception:
                future.set_exception(exception)

        if self.semaphore:
            with self.semaphore.executor_context() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except Exception as exception:
                    raise exception
        else:
            with ThreadPoolExecutor() as executor:
                executor.submit(sync_task_wrapper, func, *args, **kwargs)

        try:
            return future.result(timeout=timeout)
        except Exception as exception:
            raise exception

    def _log_execution_exceeded_timeout(self, exception: Exception) -> None:
        logger.log(
            logging.WARNING, (
                "Function execution exceeded timeout. "
                f"Error: {exception}."
            ),
        )

    def cancel_tasks(self) -> None:
        for task in self.tasks:
            if isinstance(task, (asyncio.Task, Future)):
                asyncio.create_task(self.cancel_task(task))
            else:
                logger.log(
                    logging.WARNING, (
                        "Unexpected task type encountered during tasks "
                        f"cancellation. Task type: {type(task)}"
                    ),
                )
        self.tasks.clear()
        # TODO: Track and cancel callbacks as well

    async def cancel_task(self, task: asyncio.Task) -> None:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def shutdown(self) -> None:
        """Shuts down the semaphore and cancels all tasks."""
        if self.shutdown_required:
            self.cancel_tasks()
            if self.semaphore and not self.semaphore.is_sync_shutdown:
                self.semaphore.shutdown()

    @staticmethod
    def is_async(func: Callable) -> bool:
        return (
            asyncio.iscoroutinefunction(func)
            or inspect.isasyncgenfunction(func)
        )
