import asyncio
import inspect
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from src.utilities.logging import get_logger, logging

logger: logging.Logger = get_logger(__name__)

R = TypeVar("R")  # Return type
E = TypeVar("E", bound=Exception)  # Exception type


class RetryConfig:
    SUPPORTED_PATTERNS: list[str] = ["controlled_flow", "fixed", "exponential", "custom_sequence"]

    def __init__(self, **kwargs: Any) -> None:
        self._max_retries: int = kwargs.get("max_retries", 3)
        self._initial_delay: Optional[float] = (
            float(kwargs.get("initial_delay", 2.0))
            if isinstance(kwargs.get("initial_delay", None), Union[int, float])
            else None
        )
        self._max_delay: float = float(kwargs.get("max_delay", 60.0))
        self._jitter: Optional[float] = (
            float(kwargs.get("jitter", 0.25))
            if isinstance(kwargs.get("jitter", 0.25), Union[int, float])
            else None
        )
        self._burst_capacity: Optional[int] = kwargs.get("burst_capacity", 3)
        self._rate_limit: Optional[float] = (
            float(kwargs.get("rate_limit", 0.5))
            if isinstance(kwargs.get("rate_limit", 0.5), Union[int, float])
            else None
        )
        self._retry_pattern: str = kwargs.get("retry_pattern", "controlled_flow")
        self._fixed_delay: float = float(kwargs.get("fixed_delay", 5.0))
        self._custom_sequence: Optional[List[float]] = (
            [
                float(element)
                for element in kwargs.get("custom_sequence", None)
                if isinstance(element, Union[int, float])
            ]
            if isinstance(kwargs.get("custom_sequence", None), List)
            else None
        )
        self._log_level: int = kwargs.get("log_level", logging.WARNING)
        self._concurrency_limit: Optional[int] = (
            int(kwargs.get("concurrency_limit", None))
            if isinstance(kwargs.get("concurrency_limit", None), Union[int, float])
            else None
        )
        self._on_retry_callback: Optional[Callable[[Any, Exception], Any]] = (
            kwargs.get("on_retry_callback", None)
            if inspect.isfunction(kwargs.get("on_retry_callback", None))
            else None
        )
        self._function_timeout: Optional[float] = (
            float(kwargs.get("function_timeout", None))
            if isinstance(kwargs.get("function_timeout", None), Union[int, float])
            else None
        )
        self._callback_timeout: Optional[float] = (
            float(kwargs.get("callback_timeout", None))
            if isinstance(kwargs.get("callback_timeout", None), Union[int, float])
            else None
        )

        self.max_retries = self._max_retries
        self.initial_delay = self._initial_delay
        self.max_delay = self._max_delay
        self.jitter = self._jitter
        self.retry_pattern = self._retry_pattern
        self.fixed_delay = self._fixed_delay
        if self._custom_sequence:
            self.custom_sequence = self._custom_sequence
        if self._rate_limit:
            self.rate_limit = self._rate_limit
        if self._burst_capacity:
            self.burst_capacity = self._burst_capacity
        self.log_level = self._log_level
        self.concurrency_limit = self._concurrency_limit
        self.on_retry_callback = self._on_retry_callback
        self.function_timeout = self._function_timeout
        self.callback_timeout = self._callback_timeout
        self._validate_attributes()

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        if value < 0:
            raise ValueError("max_retries must be positive")
        self._max_retries = value

    @property
    def initial_delay(self) -> Optional[float]:
        return self._initial_delay

    @initial_delay.setter
    def initial_delay(self, value: Optional[float]) -> None:
        if value is not None and value < 0.0:
            raise ValueError("initial_delay must be a positive number")
        self._initial_delay = value

    @property
    def max_delay(self) -> float:
        return self._max_delay

    @max_delay.setter
    def max_delay(self, value: float) -> None:
        if value < 0:
            raise ValueError("max_delay must be positive")
        self._max_delay = value

    @property
    def jitter(self) -> Optional[float]:
        return self._jitter

    @jitter.setter
    def jitter(self, value: Optional[float]) -> None:
        if not value:
            self._jitter = None
        if value and not 0 <= value <= 1:
            raise ValueError("jitter must be between 0 and 1 inclusive")
        self._jitter = value

    @property
    def burst_capacity(self) -> Optional[int]:
        return self._burst_capacity

    @burst_capacity.setter
    def burst_capacity(self, value: int) -> None:
        if value < 0:
            raise ValueError("burst_capacity must be equal or greater than 0")
        self._burst_capacity = value

    @property
    def rate_limit(self) -> Optional[float]:
        return self._rate_limit

    @rate_limit.setter
    def rate_limit(self, value: float) -> None:
        if value <= 0 or not isinstance(value, float):
            raise ValueError("rate_limit must be greater than 0")
        self._rate_limit = value

    @property
    def retry_pattern(self) -> str:
        return self._retry_pattern

    @retry_pattern.setter
    def retry_pattern(self, value: str) -> None:
        if value not in self.SUPPORTED_PATTERNS:
            raise ValueError(f"Unsupported retry_pattern: {value}")
        self._retry_pattern = value

    @property
    def fixed_delay(self) -> float:
        return self._fixed_delay

    @fixed_delay.setter
    def fixed_delay(self, value: float) -> None:
        if value <= 0:
            raise ValueError("fixed_delay must be positive")

        self._fixed_delay = value

    @property
    def custom_sequence(self) -> Optional[List[float]]:
        return self._custom_sequence

    @custom_sequence.setter
    def custom_sequence(self, value: Optional[List[float]]) -> None:
        if value is not None and (
            not isinstance(value, list) or not all(isinstance(n, (float, int)) for n in value)
        ):
            raise ValueError("custom_sequence must be a list or tuple if specified")
        self._custom_sequence = value

    @property
    def log_level(self) -> int:
        return self._log_level

    @log_level.setter
    def log_level(self, value: int) -> None:
        self._log_level = value

    @property
    def concurrency_limit(self) -> Optional[int]:
        return self._concurrency_limit

    @concurrency_limit.setter
    def concurrency_limit(self, value: Optional[int]) -> None:
        if value is not None and value < 0:
            raise ValueError("concurrency_limit must be positive if provided")
        self._concurrency_limit = value

    @property
    def on_retry_callback(self) -> Optional[Callable]:  # type: ignore
        return self._on_retry_callback

    @on_retry_callback.setter
    def on_retry_callback(self, value: Optional[Callable]) -> None:
        if value is not None and not callable(value):
            raise ValueError("on_retry_callback must be a callable")
        self._on_retry_callback = value

    @property
    def function_timeout(self) -> Optional[float]:
        return self._function_timeout

    @function_timeout.setter
    def function_timeout(self, value: Optional[float]) -> None:
        if value is not None and value <= 0:
            raise ValueError("function_timeout must be positive if provided")
        self._function_timeout = value

    @property
    def callback_timeout(self) -> Optional[float]:
        return self._callback_timeout

    @callback_timeout.setter
    def callback_timeout(self, value: Optional[float]) -> None:
        if value is not None and value <= 0:
            raise ValueError("callback_timeout must be positive if provided")
        self._callback_timeout = value

    def _validate_attributes(self) -> None:
        if self._retry_pattern != "custom_sequence" and self._custom_sequence:
            raise ValueError(
                "custom_sequence must be empty if retry_pattern is not custom_sequence"
            )
        if self._retry_pattern == "custom_sequence" and self._custom_sequence is None:
            raise ValueError("custom_sequence must be provided for custom_sequence retry pattern")
        if self.retry_pattern == "controlled_flow" and self.burst_capacity is None:
            raise ValueError("burst_capacity must be provided for controlled_flow retry pattern")
        if self.retry_pattern == "controlled_flow" and self.rate_limit is None:
            raise ValueError("rate_limit must be provided for controlled_flow retry pattern")
        if self.retry_pattern == "fixed" and self.fixed_delay is None:
            raise ValueError("fixed_delay must be provided for fixed retry pattern")


class SharedSemaphore:
    def __init__(self, max_workers: int = 10):
        self._async_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_workers)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)

    @contextmanager
    def executor_context(self) -> Generator[Any, Any, Any]:
        if self._executor:
            try:
                yield self._executor
            finally:
                if self._executor:
                    self.shutdown()
        else:
            yield None

    @asynccontextmanager
    async def async_lock(self) -> AsyncGenerator[None, Any]:
        if self._async_semaphore:
            await self._async_semaphore.acquire()
        try:
            yield
        finally:
            if self._async_semaphore:
                self._async_semaphore.release()

    def shutdown(self):
        if self._executor:
            self._executor.shutdown(wait=True)


class TaskManager:
    def __init__(self, shared_semaphore: Optional[SharedSemaphore] = None):
        self.shared_semaphore: Optional[SharedSemaphore] = shared_semaphore
        self.tasks: List[Union[asyncio.Task, Future]] = []

    async def submit_async_task(
        self, func: Callable[..., Any], timeout: Optional[float] = None, *args: Any, **kwargs: Any
    ) -> asyncio.Task:
        if self.shared_semaphore:
            async with self.shared_semaphore.async_lock():
                task: asyncio.Task = asyncio.create_task(func(*args, **kwargs))
                self.tasks.append(task)
                try:
                    if timeout is not None:
                        await asyncio.wait_for(task, timeout=timeout)
                    else:
                        await task
                except Exception as task_exception:
                    logger.log(
                        logging.WARNING,
                        f"Exception encountered during async task execution. Error: {task_exception.__str__()}.",
                    )
                    task.cancel()
                    # raise task_exception
                return task
        else:
            task: asyncio.Task = asyncio.create_task(func(*args, **kwargs))
            self.tasks.append(task)
            try:
                if timeout is not None:
                    await asyncio.wait_for(task, timeout=timeout)
                else:
                    await task
            except Exception as task_exception:
                logger.log(
                    logging.WARNING,
                    f"Exception encountered during async task execution. Error: {task_exception.__str__()}.",
                )
                task.cancel()
                # raise task_exception
            return task

    def submit_sync_task(
        self, func: Callable[..., Any], timeout: Optional[float] = None, *args: Any, **kwargs: Any
    ) -> Future:
        with (
            self.shared_semaphore.executor_context()
            if self.shared_semaphore
            else ThreadPoolExecutor()
        ) as executor:
            future: Future = executor.submit(func, *args, **kwargs)
            self.tasks.append(future)
            try:
                if timeout is not None:
                    future.result(timeout=timeout)
                return future
            except Exception as task_exception:
                logger.log(
                    logging.WARNING,
                    f"Exception encountered during sync task execution. Error: {task_exception.__str__()}.",
                )
                future.cancel()
                # raise task_exception
            return future

    def cancel_tasks(self) -> None:
        for task in self.tasks:
            if isinstance(task, (asyncio.Task, Future)):
                task.cancel()
            else:
                logger.log(logging.WARNING, "Unknown task type encountered during cancellation.")
        self.tasks.clear()


class RetryContext:
    def __init__(
        self,
        exceptions: Union[Type[E], Tuple[Type[E], ...]],
        config: RetryConfig,
        shared_semaphore: Optional[SharedSemaphore] = None,
    ):
        self.exceptions: Union[Type[E], Tuple[Type[E], ...]] = exceptions
        self.config: RetryConfig = config
        self.shared_semaphore: Optional[SharedSemaphore] = shared_semaphore
        self.attempt: int = 0
        self.task_manager: TaskManager = TaskManager(shared_semaphore)
        self.retry_strategy: RetryStrategy = RetryStrategy(exceptions, self)
        self.delays: List[float] = []
        self.retry_strategy._calculate_delays()

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[types.TracebackType]) -> None:
        self.task_manager.cancel_tasks()
        if self.shared_semaphore:
            self.shared_semaphore.shutdown()

    def __enter__(self) -> "RetryContext":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[types.TracebackType]) -> None:
        if exc_type is None:
            logger.log(
                logging.INFO, f"Function completed successfully after {self.attempt} attempts."
            )
        else:
            logger.log(
                logging.ERROR, f"Function failed after {self.attempt} attempts due to {exc_val}."
            )
        self.task_manager.cancel_tasks()
        if self.shared_semaphore:
            self.shared_semaphore.shutdown()

    @staticmethod
    def is_async(func: Callable) -> bool:
        return asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)

    async def retry_async_call(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        while True:
            try:
                self.attempt += 1
                task = await self.task_manager.submit_async_task(
                    func, *args, **kwargs, timeout=self.config.function_timeout
                )
                return await task
            except self.exceptions as e:
                await self.handle_async_retry(e)

    def retry_sync_call(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        while True:
            try:
                self.attempt += 1
                future = self.task_manager.submit_sync_task(
                    func, *args, **kwargs, timeout=self.config.function_timeout
                )
                return future.result()
            except self.exceptions as e:
                self.handle_sync_retry(e)

    async def handle_async_retry(self, exception: Exception):
        delay: float = 0.0
        if isinstance(exception, asyncio.CancelledError):
            callback_message: str = ""
            if self.config.on_retry_callback:
                callback_message = " Callback function will be skipped for the current attempt."
            if not self._limit_reached():
                retry_message: str = ", but retrying"
            logger.log(logging.INFO, f"Task was cancelled{retry_message}.{callback_message}")
        elif isinstance(exception, TimeoutError):
            logger.log(logging.WARNING, f"Function execution exceeded timeout. Error: {exception}")
        elif isinstance(exception, Exception):
            logger.log(logging.WARNING, f"Function failed with exception. Error: {exception}")

        retry_message: str = "."
        if not self._limit_reached():
            delay = self.retry_strategy.get_delay()
            retry_message = f"; retrying in {delay}s due to exception: {exception}"

        logger.log(
            self.config.log_level,
            f"Attempt {self.attempt}/{self.config.max_retries + 1} failed{retry_message}",
        )
        if self.config.on_retry_callback is not None:
            if RetryContext.is_async(self.config.on_retry_callback):
                # Execute async callback directly
                await self.execute_callback(self.config.on_retry_callback, exception)
            else:
                # Execute sync callback within the async context
                await asyncio.get_running_loop().run_in_executor(
                    None, self.execute_callback, self.config.on_retry_callback, exception
                )

        if self._limit_reached():
            logger.log(
                logging.INFO,
                f"Retries exhausted. Function failed after {self.attempt} attempts due to {exception}.",
            )
            raise exception

        if delay >= 0:
            await asyncio.sleep(delay)

    def handle_sync_retry(self, exception: Exception):
        delay: float = 0.0
        if isinstance(exception, asyncio.CancelledError):
            callback_message: str = ""
            if self.config.on_retry_callback:
                callback_message = " Callback function will be skipped for the current attempt."
            if not self._limit_reached():
                retry_message: str = ", but retrying"
            logger.log(logging.INFO, f"Task was cancelled{retry_message}.{callback_message}")
        elif isinstance(exception, TimeoutError):
            logger.log(logging.WARNING, "Function execution exceeded timeout.")
        elif isinstance(exception, Exception):
            logger.log(logging.WARNING, f"Function failed with exception. Error: {exception}")

        retry_message: str = "."
        if not self._limit_reached():
            delay = self.retry_strategy.get_delay()
            retry_message = f"; retrying in {delay}s due to exception: {exception}"

        logger.log(
            self.config.log_level,
            f"Attempt {self.attempt}/{self.config.max_retries + 1} failed{retry_message}",
        )

        if self.config.on_retry_callback is not None:
            if RetryContext.is_async(self.config.on_retry_callback):
                # Execute async callback in an event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Run in a separate thread if loop is already running
                    self.task_manager.submit_sync_task(
                        asyncio.run,
                        self.config.callback_timeout,
                        self.config.on_retry_callback(exception),
                    )
                else:
                    # Directly run the coroutine if no event loop is running
                    loop.run_until_complete(
                        self.task_manager.submit_async_task(
                            self.config.on_retry_callback,
                            timeout=self.config.callback_timeout,
                            exception=exception,
                        )
                    )
            else:
                # Execute sync callback directly
                _ = self.execute_callback(self.config.on_retry_callback, exception)

        if self._limit_reached():
            logger.log(
                logging.INFO,
                f"Retries exhausted. Function failed after {self.attempt} attempts due to {exception}.",
            )
            raise exception

        if delay >= 0:
            time.sleep(delay)

    def _limit_reached(self) -> bool:
        return self.attempt > self.config.max_retries

    async def execute_callback(self, callback: Callable, exception: Exception) -> None:  # type: ignore
        if callback is None:
            return
        try:
            logger.log(
                logging.INFO,
                f"Executing callback function on attempt {self.attempt} due to exception: {exception}",
            )

            if RetryContext.is_async(callback):
                await self.task_manager.submit_async_task(
                    callback(exception=exception),
                    timeout=self.config.callback_timeout,
                    exception=exception,
                )
            else:
                self.task_manager.submit_sync_task(
                    callback(exception=exception),
                    timeout=self.config.callback_timeout,
                    exception=exception,
                )
        except asyncio.TimeoutError:
            logger.log(logging.WARNING, "Callback function exceeded timeout.")
        except Exception as callback_error:
            logger.log(
                logging.ERROR,
                f"Error executing callback function on attempt {self.attempt}: {callback_error}",
            )


class RetryStrategy:
    def __init__(
        self, exceptions: Union[Type[E], Tuple[Type[E], ...]], context: RetryContext
    ) -> None:
        self.exceptions: Union[Type[E], Tuple[Type[E], ...]] = exceptions
        self.retry_context: RetryContext = context

    def default_backoff(self) -> list[float]:
        retries: dict = {
            "total": self.retry_context.config.max_retries,
        }

        jitter: float = self.retry_context.config.jitter or 0.0
        initial_delay: float = self.retry_context.config.initial_delay or 0.0

        def default_backoff_values(initial_backoff: float, retry: int = 0) -> list[float]:
            nonlocal retries
            nonlocal jitter
            if retry >= retries["total"]:
                return []
            delay: float = initial_backoff * (2**retry)
            jitter_val: float = delay * random.uniform(0, jitter)
            current_backoff: float = min(delay + jitter_val, self.retry_context.config.max_delay)
            return (
                [current_backoff] + default_backoff_values(current_backoff, retry + 1)
                if retry > 0
                else [initial_delay] + default_backoff_values(current_backoff, retry + 1)
            )

        return default_backoff_values(initial_delay)

    def controlled_flow(self) -> list[float]:
        rate_limit: float = self.retry_context.config.rate_limit or 0.0  # 0.5 is a good value
        burst_capacity: int = self.retry_context.config.burst_capacity or 0  # 3 is a good value
        jitter_factor: float = self.retry_context.config.jitter or 0.0  # 0.25 is a good value

        if rate_limit is not None and burst_capacity is not None:
            refill_rate = 1.0 / rate_limit  # Time to refill one token.

            # Utilize burst capacity for initial retries
            burst_intervals: List[float] = [0] * min(self.retry_context.config.max_retries, burst_capacity)

            # Calculate adaptive delays for retries beyond the burst capacity
            adaptive_delays: List[float] = [
                refill_rate * (i + 1)
                for i in range(self.retry_context.config.max_retries - burst_capacity)
            ]

            # Combine burst intervals with adaptive delays
            combined_intervals: List[float] = burst_intervals + adaptive_delays

            # Apply jitter to all intervals except the initial burst
            for i, interval in enumerate(combined_intervals):
                if i >= burst_capacity:  # Apply jitter only beyond burst capacity
                    jitter: float = interval * jitter_factor * (random.random() * 2 - 1)
                    combined_intervals[i] = float(
                        max(interval + jitter, 0)
                    )  # Ensure non-negative delay

            return combined_intervals

        raise ValueError("Rate limit and burst capacity must be specified for controlled flow.")

    def custom_sequence(self) -> list[float]:
        if self.retry_context.config.custom_sequence is None:
            raise ValueError("Custom sequence must be specified for custom sequence retry pattern.")
        return self.retry_context.config.custom_sequence

    def _calculate_delays(self):
        def custom_sequence() -> List[float]:
            return self.retry_context.config.custom_sequence

        def fixed() -> List[float]:
            return [
                self.retry_context.config.fixed_delay
                for _ in range(self.retry_context.config.max_retries)
            ]

        def exponential() -> List[float]:
            return self.default_backoff()

        def controlled_flow() -> List[float]:
            return self.controlled_flow()

        # Mapping retry patterns to their corresponding functions
        pattern_actions: dict[str, Callable[[], List[float]]] = {
            "custom_sequence": custom_sequence,
            "fixed": fixed,
            "exponential": exponential,
            "controlled_flow": controlled_flow,
        }

        action: Optional[Callable[[], List[float]]] = pattern_actions.get(self.retry_context.config.retry_pattern)
        if action:
            self.retry_context.delays = action()
        else:
            raise ValueError("Unsupported retry pattern.")

    def get_delay(self) -> float:
        return (
            self.retry_context.delays[self.retry_context.attempt]
            if self.retry_context.attempt < len(self.retry_context.delays)
            else 0.0
        )


def retry_on_exception(exceptions: Union[Type[E], Tuple[Type[E], ...]], **kwargs: Any) -> Callable:
    def _validate_exceptions(exc: Type[E]) -> None:
        if not issubclass(exc, Exception):
            raise ValueError("exception must subclass Exception")
        if exceptions is None:
            raise ValueError("exceptions cannot be None")

    if isinstance(exceptions, Type):
        _validate_exceptions(exceptions)
    if isinstance(exceptions, tuple):
        for exc in exceptions:
            _validate_exceptions(exc)

    retry_config: RetryConfig = RetryConfig(**kwargs)
    shared_semaphore: Optional[SharedSemaphore] = (
        SharedSemaphore(max_workers=retry_config.concurrency_limit)
        if retry_config.concurrency_limit
        else None
    )

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            async with RetryContext(
                exceptions, retry_config, shared_semaphore=shared_semaphore
            ) as context:
                return await context.retry_async_call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> R:
            with RetryContext(
                exceptions, retry_config, shared_semaphore=shared_semaphore
            ) as context:
                return context.retry_sync_call(func, *args, **kwargs)

        return async_wrapper if RetryContext.is_async(func) else sync_wrapper

    return decorator