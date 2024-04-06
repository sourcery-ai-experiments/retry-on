import asyncio
import inspect
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import (Any, Callable, List, Optional, Union)

from src.retry_on.logging import get_logger, logging
from src.retry_on.types import R
from src.retry_on.concurrency import ConcurrencyManager


logger: logging.Logger = get_logger(__name__)


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
    ) -> asyncio.Task:
        if (self.semaphore is not None and self.semaphore.is_sync_shutdown):
            self._log_semaphore_shutdown(func)
            raise Exception(
                "Semaphore is shut down. "
                "Cannot submit new functions for executions."
            )

        async def async_task_wrapper(
            func: Callable,
            *args: Any,
            **kwargs: Any
        ) -> R:
            try:
                return await func(*args, **kwargs)
            except Exception as exception:
                raise exception

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
    ) -> R:
        try:
            return await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            self._log_execution_exceeded_timeout(task)
            task.cancel()
            raise
        except asyncio.CancelledError:
            logger.debug(
                (
                    "Task {task_name} was canceled."
                ).format(
                    task_name=task.get_name()
                )
            )
            raise

    def submit_sync_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> Optional[R]:
        if (self.semaphore is not None and self.semaphore.is_sync_shutdown):
            logger.warning(
                "Semaphore is shut down. Cannot submit new functions "
                f"for executions. Function: {func.__name__}."
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

    def _log_execution_exceeded_timeout(self, task: asyncio.Task) -> None:
        logger.debug(
                "Task execution exceeded timeout. "
                f"Task: {task.get_name()}."
        )

    def _log_semaphore_shutdown(self, func: Callable) -> None:
        logger.debug(
                "Semaphore is shut down. Cannot submit new tasks. "
                f"Task: {func.__name__}."
        )

    def cancel_tasks(self) -> None:
        for task in self.tasks:
            if isinstance(task, (asyncio.Task)):
                asyncio.create_task(self.cancel_task(task))
            elif isinstance(task, (Future)):
                task.cancel()
            else:
                logger.warning(
                        "Unexpected task type encountered during tasks "
                        f"cancellation. Task type: {type(task)}."
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
            if (
                self.semaphore is not None
                and
                not self.semaphore.is_sync_shutdown
            ):
                self.semaphore.shutdown()

    @staticmethod
    def is_async(func: Callable) -> bool:
        return (
            asyncio.iscoroutinefunction(func)
            or inspect.isasyncgenfunction(func)
        )
