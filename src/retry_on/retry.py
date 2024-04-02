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
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union
)
from types import TracebackType

from src.retry_on.logging import get_logger, logging

logger: logging.Logger = get_logger(__name__)

R = TypeVar("R")  # Return type
E = TypeVar("E", bound=Exception)  # Exception type


class RetryConfig:
    SUPPORTED_RETRY_PATTERNS: List[str] = [
       "controlled_flow",
       "fixed",
       "exponential",
       "custom_sequence",
       "linear"
    ]

    FALLBACK_RETRY_PATTERN_DEFAULT_PROPERTIES: Dict[str, dict] = {
        "controlled_flow": {
            "rate_limit": 0.5,
            "burst_capacity": 3,
            "jitter": 0.25,
            "initial_delay": 2.0,
            "max_delay": 60.0,
        },
        "fixed": {
            "fixed_delay": 5.0,
            "initial_delay": 2.0,
            "max_delay": 60.0
        },
        "exponential": {
            "initial_delay": 2.0,
            "max_delay": 60.0,
            "jitter": 0.25,
        },
        "linear": {
            "linear_delay": 0.0,
            "initial_delay": 2.0,
            "max_delay": 60.0
        },
        "custom_sequence": {
            "custom_sequence": [1, 2, 3],
            "initial_delay": 2.0,
            "max_delay": 60.0
        },
    }

    def __init__(self, **kwargs):
        # Setting default values to improve developer experience
        self.max_retries = kwargs.get("max_retries", 3)
        self.initial_delay = self._get_float_or_value(kwargs, "initial_delay", 2.0)
        self.max_delay = self._get_float_or_value(kwargs, "max_delay", 60.0)
        self.jitter = self._get_float_or_value(kwargs, "jitter", 0.25)
        self.burst_capacity = self._get_int_or_value(kwargs, "burst_capacity", 3)
        self.rate_limit = self._get_float_or_value(kwargs, "rate_limit", 0.5)
        self.retry_pattern = kwargs.get("retry_pattern", "controlled_flow")
        self.linear_delay = self._get_float_or_value(kwargs, "linear_delay", 0.0)
        self.fixed_delay = self._get_float_or_value(kwargs, "fixed_delay", 5.0)
        self.custom_sequence = self._get_custom_sequence(kwargs, "custom_sequence")
        self.log_level = kwargs.get("log_level", logging.WARNING)
        self.concurrency_limit = self._get_int_or_value(kwargs, "concurrency_limit")
        self.on_retry_callback = kwargs.get("on_retry_callback")
        self.function_timeout = self._get_float_or_value(kwargs, "function_timeout")
        self.callback_timeout = self._get_float_or_value(kwargs, "callback_timeout")
        self._ensure_retry_pattern()
        self._validate_attributes()

    def _set_fallback_retry_pattern_properies(self, strategy) -> None:
        if strategy not in self.SUPPORTED_RETRY_PATTERNS:
            raise ValueError(
                "Unsupported retry pattern while setting "
                f"fallback retry stratergy: {strategy}"
            )
        for prop in self.FALLBACK_RETRY_PATTERN_DEFAULT_PROPERTIES[strategy]:
            if getattr(self, prop) is None:
                setattr(self, prop, self.FALLBACK_RETRY_PATTERN_DEFAULT_PROPERTIES[strategy][prop])

    def _set_fallback_retry_pattern(self, strategy) -> None:
        if strategy not in self.SUPPORTED_RETRY_PATTERNS:
            raise ValueError(
                "Unsupported retry pattern while setting "
                f"fallback retry stratergy: {strategy}"
            )
        self._set_fallback_retry_pattern_properies(strategy)

    @staticmethod
    def _get_float_or_value(kwargs, key, default=None) -> Optional[float]:
        value: Any = kwargs.get(key, default)
        return value if value is None or isinstance(value, (float, int)) else default

    @staticmethod
    def _get_int_or_value(kwargs, key, default=None) -> Optional[int]:
        value: Any = kwargs.get(key, default)
        if value is None or isinstance(value, int):
            return value
        elif isinstance(value, (float, int)):
            return int(value)
        else:
            return default

    @staticmethod
    def _get_custom_sequence(kwargs, key) -> Optional[List[float]]:
        value: Any = kwargs.get(key)
        if isinstance(value, list):
            return (
                [float(element) for element in value if isinstance(element, (int, float))]
            )
        return None

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("max_retries must be an integer")
        if value < 0:
            raise ValueError("max_retries must be a positive integer")
        self._max_retries = value

    @property
    def initial_delay(self) -> Optional[float]:
        return self._initial_delay

    @initial_delay.setter
    def initial_delay(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("initial_delay must be a float or integer")
        if value is not None and value < 0:
            raise ValueError("initial_delay must be a positive number if provided")
        self._initial_delay = value

    @property
    def max_delay(self) -> Optional[float]:
        return self._max_delay

    @max_delay.setter
    def max_delay(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("max_delay must be a float or integer")
        if value is not None and value < 0:
            raise ValueError("max_delay must be a positive float or integer if provided")
        self._max_delay = value

    @property
    def jitter(self) -> Optional[float]:
        return self._jitter

    @jitter.setter
    def jitter(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("jitter must be a float or integer")
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("jitter must be a number between 0 and 1 if provided")
        self._jitter = value

    @property
    def burst_capacity(self) -> Optional[int]:
        return self._burst_capacity

    @burst_capacity.setter
    def burst_capacity(self, value: Optional[int]) -> None:
        if value is not None and not isinstance(value, int):
            raise TypeError("burst_capacity must be an integer")
        if value is not None and value < 0:
            raise ValueError("burst_capacity must be a positive integer if provided")
        self._burst_capacity = value

    @property
    def rate_limit(self) -> Optional[float]:
        return self._rate_limit

    @rate_limit.setter
    def rate_limit(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("rate_limit must be a float or integer")
        if value is not None and value <= 0:
            raise ValueError(
                "rate_limit must be a positive float or integer "
                "greater than zero if provided"
            )
        self._rate_limit = value

    @property
    def retry_pattern(self) -> Optional[str]:
        return self._retry_pattern

    @retry_pattern.setter
    def retry_pattern(self, value: Optional[str]) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError("retry_pattern must be a string")
        if value not in self.SUPPORTED_RETRY_PATTERNS:
            raise ValueError(f"Unsupported retry_pattern: {value}")
        if hasattr(self, '_retry_pattern') and getattr(self, '_retry_pattern', None) is not None:
            self._update_retry_pattern(value)
        self._retry_pattern = value

    @property
    def linear_delay(self) -> Optional[float]:
        return self._linear_delay

    @linear_delay.setter
    def linear_delay(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("linear_delay must be a float or integer")
        if value is not None and value < 0:
            raise ValueError("linear_delay must be a positive float or integer if provided")
        self._linear_delay = value

    @property
    def fixed_delay(self) -> Optional[float]:
        return self._fixed_delay

    @fixed_delay.setter
    def fixed_delay(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("fixed_delay must be a float or integer")
        if value is not None and value < 0:
            raise ValueError("fixed_delay must be a positive float or integer")
        self._fixed_delay = value

    @property
    def custom_sequence(self) -> Optional[List[float]]:
        return self._custom_sequence

    @custom_sequence.setter
    def custom_sequence(self, value: Optional[List[float]]) -> None:
        if value is not None and not isinstance(value, (list, tuple)):
            raise TypeError("custom_sequence must be a list or a tuple")
        if value is not None and not all(isinstance(n, (int, float)) for n in value):
            raise ValueError("custom_sequence elements must be floats or integers if provided")
        self._custom_sequence = value

    @property
    def concurrency_limit(self) -> Optional[int]:
        return self._concurrency_limit

    @concurrency_limit.setter
    def concurrency_limit(self, value: Optional[int]) -> None:
        if value is not None and not isinstance(value, int):
            raise TypeError("concurrency_limit must be an integer")
        if value is not None and value < 0:
            raise ValueError("concurrency_limit must be a positive integer if provided")
        self._concurrency_limit = value

    @property
    def on_retry_callback(self) -> Optional[Callable]:
        return self._on_retry_callback

    @on_retry_callback.setter
    def on_retry_callback(self, value) -> None:
        if value is not None and not callable(value):
            raise TypeError("on_retry_callback must be callable")
        self._on_retry_callback: Optional[Callable] = value

    @property
    def function_timeout(self) -> Optional[float]:
        return self._function_timeout

    @function_timeout.setter
    def function_timeout(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("function_timeout must be a float or integer")
        if value is not None and value <= 0:
            raise ValueError(
                "function_timeout must be a positive float or integer "
                "greater than zero if provided"
            )
        self._function_timeout = value

    @property
    def callback_timeout(self) -> Optional[float]:
        return self._callback_timeout

    @callback_timeout.setter
    def callback_timeout(self, value: Optional[float]) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("callback_timeout must be a float or integer")
        if value is not None and value < 0:
            raise ValueError("callback_timeout must be a positive float or integer if provided")
        self._callback_timeout = value

    @property
    def log_level(self) -> int:
        return self._log_level

    @log_level.setter
    def log_level(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("log_level must be an integer")
        if value < 0 or value > 50:
            raise ValueError("log_level must be between 0 and 50 inclusive")
        self._log_level = value
        if value > 30:
            logger.warning(
                "log_level is set to a value greater than 30. "
                "This may cause excessive logging and may impact performance."
            )
        if value < logging.CRITICAL:
            logger.warning(
                "log_level is set to a value less than logging.CRITICAL. "
                "This may cause unexpected behavior."
            )

    def _ensure_retry_pattern(self) -> None:
        if self.retry_pattern is None:
            self._set_fallback_retry_pattern("controlled_flow")

    def _validate_attributes(self) -> None:
        if (
            self.retry_pattern == "custom_sequence"
            and self.custom_sequence is None
        ):
            raise ValueError(
                "custom_sequence must be provided for custom_sequence retry pattern"
            )
        if self.retry_pattern == "controlled_flow" and (
            self.burst_capacity is None
            or self.rate_limit is None
        ):
            raise ValueError(
                "burst_capacity and rate_limit must be provided for controlled_flow retry pattern"
            )
        if self.retry_pattern == "fixed" and self.fixed_delay is None:
            raise ValueError(
                "fixed_delay must be provided for fixed retry pattern"
            )
        if (
            self.max_delay is not None
            and self.initial_delay is not None
            and self.max_delay < self.initial_delay
        ):
            raise ValueError("max_delay must be greater than or equal to initial_delay")

    def _update_retry_pattern(self, new_retry_pattern: str) -> None:
        default_properties = self.FALLBACK_RETRY_PATTERN_DEFAULT_PROPERTIES.get(new_retry_pattern)
        if default_properties is None:
            raise ValueError("Invalid retry pattern provided")
        
        current_properties = {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
            "burst_capacity": self.burst_capacity,
            "rate_limit": self.rate_limit,
            "linear_delay": self.linear_delay,
            "fixed_delay": self.fixed_delay,
            "custom_sequence": self.custom_sequence
        }
        
        updated_properties = {**default_properties, **current_properties}
        
        self.max_retries = updated_properties["max_retries"]
        self.initial_delay = updated_properties["initial_delay"]
        self.max_delay = updated_properties["max_delay"]
        self.jitter = updated_properties["jitter"]
        self.burst_capacity = updated_properties["burst_capacity"]
        self.rate_limit = updated_properties["rate_limit"]
        self.linear_delay = updated_properties["linear_delay"]
        self.fixed_delay = updated_properties["fixed_delay"]
        self.custom_sequence = updated_properties["custom_sequence"]
        
        self._validate_attributes()
        

class SharedSemaphore:
    def __init__(self, max_workers: int = 10):
        self._async_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_workers)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)

    @contextmanager
    def executor_context(self) -> Generator[Any, Any, Any]:
        try:
            yield self._executor
        except Exception as exception:
            raise exception
        finally:
            self._shutdown()

    @asynccontextmanager
    async def async_lock(self) -> AsyncGenerator[None, Any]:
        await self._async_semaphore.acquire()
        try:
            yield
        finally:
            self._async_semaphore.release()

    def _shutdown(self):
        self._executor.shutdown(wait=True)


class TaskManager:
    def __init__(self, shared_semaphore: Optional[SharedSemaphore] = None):
        self.shared_semaphore: Optional[SharedSemaphore] = shared_semaphore
        self.tasks: List[Union[asyncio.Task, Future]] = []

    async def submit_async_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> R:
        async def async_task_wrapper():
            try:
                return await func(*args, **kwargs)
            except Exception as exception:
                raise exception

        if self.shared_semaphore:
            async with self.shared_semaphore.async_lock():
                task: asyncio.Task = asyncio.create_task(async_task_wrapper())
                self.tasks.append(task)
                return (
                    await asyncio.wait_for(task, timeout=timeout)
                    if timeout
                    else await task
                )
        else:
            task: asyncio.Task = asyncio.create_task(async_task_wrapper())
            self.tasks.append(task)
            try:
                return (
                    await asyncio.wait_for(task, timeout=timeout)
                    if timeout
                    else await task
                )
            except asyncio.TimeoutError as task_exception:
                self._log_execution_exceeded_timeout(task_exception)
                task.cancel()
                return None

    def submit_sync_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> R:
        future: Future = Future()
        
        def sync_task_wrapper():
            try:
                future.set_result(func(*args, **kwargs))
            except Exception as exception:
                future.set_exception(exception)

        if self.shared_semaphore:
            with self.shared_semaphore.executor_context() as executor:
                future: Future = executor.submit(func, *args, **kwargs)
                self.tasks.append(future)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError as task_exception:
                    self._log_execution_exceeded_timeout(task_exception)
                    future.cancel()
                    return None
                except Exception as exception:
                    raise exception
        else:
            with ThreadPoolExecutor() as executor:
                executor.submit(sync_task_wrapper)
                self.tasks.append(future)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError as task_exception:
                    self._log_execution_exceeded_timeout(task_exception)
                    future.cancel()
                    return None

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
                task.cancel()
            else:
                logger.log(
                    logging.WARNING, (
                        "Unexpected task type encountered during tasks cancellation. "
                        f"Task type: {type(task)}"
                    ),
                )
        self.tasks.clear()

    @staticmethod
    def is_async(func: Callable) -> bool:
        return (
            asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)
        )


class RetryContext:
    def __init__(
        self,
        exceptions: Union[Type[E], Tuple[Type[E], ...]],
        config: RetryConfig,
        shared_semaphore: Optional[SharedSemaphore] = None,
    ):
        self.allowed_exceptions: Union[Type[E], Tuple[Type[E], ...]] = exceptions
        self.config: RetryConfig = config
        self.shared_semaphore: Optional[SharedSemaphore] = shared_semaphore
        self.attempt: int = 0
        self.task_manager: TaskManager = TaskManager(shared_semaphore)
        self.retry_strategy: RetryStrategy = RetryStrategy(self)
        self.delays: List[float] = []
        self.retry_strategy._calculate_delays()

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._exit_message(exc_type, exc_val)
        self._graceful_exit()

    def __enter__(self) -> "RetryContext":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._exit_message(exc_type, exc_val)
        self._graceful_exit()

    async def retry_async_call(
        self, func: Callable[..., R], *args: Any, **kwargs: Any
    ) -> R:
        while True:
            try:
                self.attempt += 1
                return await self.task_manager.submit_async_task(
                    func, *args, timeout=self.config.function_timeout, **kwargs
                )
            except self.allowed_exceptions as exception:
                await self.handle_async_retry(exception)
            except Exception as exception:
                self._handle_unallowed_exception(exception)
                raise exception

    def retry_sync_call(
        self, func: Callable[..., R], *args: Any, **kwargs: Any
    ) -> R:
        while True:
            try:
                self.attempt += 1
                return self.task_manager.submit_sync_task(
                    func, *args, timeout=self.config.function_timeout, **kwargs
                )
            except self.allowed_exceptions as exception:
                self.handle_sync_retry(exception)
            except Exception as exception:
                self._handle_unallowed_exception(exception)
                raise exception

    def _handle_unallowed_exception(self, exception: Exception) -> None:
        if not isinstance(exception, self.allowed_exceptions):
            self._log_unallowed_exception(exception)
            self._graceful_exit()
            raise exception
    
    def _log_unallowed_exception(self, exception: Exception) -> None:
        logger.log(
            logging.ERROR, (
                "Function raised an unallowed exception "
                f"during attempt {self.attempt}/{self.config.max_retries + 1}. "
                "Function will not be retried. "
                f"Error: {exception}."
            ),
        )

    async def handle_async_retry(self, exception: Exception):
        if self._limit_reached():
            raise exception
        self._log_failed_attempt(exception)
        if delay := self.retry_strategy.get_delay():
            await asyncio.sleep(delay)
        if self.config.on_retry_callback:
            await self.async_execute_callback(exception)

    def handle_sync_retry(self, exception: Exception):
        if self._limit_reached():
            raise exception
        self._log_failed_attempt(exception)
        if delay := self.retry_strategy.get_delay():
            time.sleep(delay)
        if self.config.on_retry_callback:
            self.sync_execute_callback(exception)

    def _limit_reached(self) -> bool:
        return self.attempt > self.config.max_retries

    async def async_execute_callback(self, exception: Exception) -> None:
        if not self.config.on_retry_callback:
            return
        self._log_callback_attempt(exception)
        try:
            if TaskManager.is_async(self.config.on_retry_callback):
                await self.task_manager.submit_async_task(
                    self.config.on_retry_callback,
                    timeout=self.config.callback_timeout,
                    exception=exception
                )
            else:
                self.task_manager.submit_sync_task(
                    self.config.on_retry_callback,
                    timeout=self.config.callback_timeout,
                    exception=exception
                )
        except Exception as callback_exception:
            self._log_failed_callback_attempt(callback_exception)

    def sync_execute_callback(self, exception: Exception) -> None:
        if not self.config.on_retry_callback:
            return
        self._log_callback_attempt(exception)
        try:
            if TaskManager.is_async(self.config.on_retry_callback):
                loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
                if loop.is_running():
                    def run_async_coro(coro: Callable, loop: asyncio.AbstractEventLoop):
                        try:
                            return loop.run_until_complete(coro)
                        except RuntimeError:
                            loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            return loop.run_until_complete(coro)

                    with ThreadPoolExecutor() as executor:
                        future: Future = executor.submit(
                            run_async_coro,
                            self.task_manager.submit_async_task(
                                self.config.on_retry_callback,
                                timeout=self.config.callback_timeout,
                                exception=exception
                            ),
                            loop
                        )
                        future.result()
                else:
                    loop.run_until_complete(
                        self.task_manager.submit_async_task(
                            self.config.on_retry_callback,
                            timeout=self.config.callback_timeout,
                            exception=exception
                        )
                    )
            else:
                self.task_manager.submit_sync_task(
                    self.config.on_retry_callback,
                    timeout=self.config.callback_timeout,
                    exception=exception,
                )
        except Exception as callback_exception:
            self._log_failed_callback_attempt(callback_exception)

    def _exit_message(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException]
    ) -> None:
        if exc_type is None:
            logger.log(logging.INFO, (
                f"Function completed successfully after "
                f"{self.attempt} attempts."
                )
            )
        else:
            logger.log(logging.ERROR, (
                f"Function failed after {self.attempt} "
                f"attempts due to exception {exc_type.__name__}. "
                f"Error message: {exc_val}."
                )
            )

    def _log_failed_attempt(self, exception: Exception) -> None:
        logger.log(self.config.log_level, (
            f"Attempt {self.attempt}/{self.config.max_retries + 1} "
            f"failed: {exception}"
            )
        )

    def _log_callback_attempt(self, exception: Exception) -> None:
        logger.log(logging.INFO, (
                f"Executing callback function on attempt "
                f"{self.attempt} due to exception: {exception}"
            )
        )

    def _log_failed_callback_attempt(self, exception: Exception) -> None:
        logger.log(logging.ERROR, (
                f"Error executing callback function on attempt "
                f"{self.attempt}: {exception}"
            )
        )

    def _graceful_exit(self) -> None:
        self.task_manager.cancel_tasks()
        if self.shared_semaphore:
            self.shared_semaphore.shutdown()


class RetryStrategy:
    def __init__(
        self,
        context: RetryContext,
    ) -> None:
        self.retry_context: RetryContext = context

    def exponential(self) -> List[float]:
        retries: dict = {"total": self.retry_context.config.max_retries}
        jitter: Optional[float] = self.retry_context.config.jitter
        initial_delay: Optional[float] = self.retry_context.config.initial_delay
        max_delay: Optional[float] = self.retry_context.config.max_delay
        strategy_name: str = inspect.currentframe().f_code.co_name

        self._validate_strategy_properties(
            strategy_name,
            max_retries=retries["total"],
            jitter=jitter,
            max_delay=max_delay,
            initial_delay=initial_delay
        )

        def _intervals(
            initial_backoff: float,
            _retry: int = 0
        ) -> List[float]:
            if _retry >= retries["total"]:
                return []
            delay: float = initial_backoff * (2 ** _retry)
            jitter_val: float = delay * random.uniform(0, jitter)
            current_backoff: float = min(
                delay + jitter_val, max_delay
            )
            return (
                [current_backoff]
                + _intervals(current_backoff, _retry + 1)
                if _retry > 0
                else [initial_delay]
                + _intervals(current_backoff, _retry + 1)
            )

        return _intervals(initial_delay)

    def controlled_flow(self) -> List[float]:
        rate_limit: Optional[float] = self.retry_context.config.rate_limit
        burst_capacity: Optional[int] = self.retry_context.config.burst_capacity
        max_retries: int = self.retry_context.config.max_retries
        jitter_factor: float = self.retry_context.config.jitter
        strategy_name: str = inspect.currentframe().f_code.co_name

        self._validate_strategy_properties(
            strategy_name,
            rate_limit=rate_limit,
            burst_capacity=burst_capacity,
            max_retries=max_retries,
            jitter=jitter_factor,
        )

        def _intervals(rate_limit: float, burst_capacity: int) -> List[float]:
            refill_rate: float = 1.0 / rate_limit

            burst_intervals: List[float] = [0] * min(
                max_retries, burst_capacity
            )

            adaptive_delays: List[float] = [
                refill_rate * (i + 1)
                for i in range(
                    max_retries - burst_capacity
                )
            ]

            combined_intervals: List[float] = burst_intervals + adaptive_delays

            for i, interval in enumerate(combined_intervals):
                if i >= burst_capacity:
                    jitter: float = interval * jitter_factor * (random.random() * 2 - 1)
                    combined_intervals[i] = float(max(interval + jitter, 0))

            return combined_intervals

        return _intervals(rate_limit, burst_capacity)

    def custom_sequence(self) -> List[float]:
        custom_sequence: List[float] = self.retry_context.config.custom_sequence
        strategy_name: str = inspect.currentframe().f_code.co_name

        self._validate_strategy_properties(
            strategy_name,
            custom_sequence=custom_sequence
        )
        
        return custom_sequence

    def fixed(self) -> List[float]:
        fixed_delay: Optional[float] = self.retry_context.config.fixed_delay
        strategy_name: str = inspect.currentframe().f_code.co_name
        max_retries: int = self.retry_context.config.max_retries

        self._validate_strategy_properties(
            strategy_name,
            fixed_delay=fixed_delay,
            max_retries=max_retries
        )

        return [
                fixed_delay
                for _ in range(max_retries)
            ]

    def linear(self) -> List[float]:
        retries: dict = {"total": self.retry_context.config.max_retries}
        initial_delay: Optional[float] = self.retry_context.config.initial_delay
        linear_delay: Optional[float] = self.retry_context.config.linear_delay
        strategy_name: str = inspect.currentframe().f_code.co_name

        self._validate_strategy_properties(
            strategy_name,
            initial_delay=initial_delay,
            linear_delay=linear_delay,
            max_retries=retries["total"]
        )

        def _interval_value(attempt: int) -> float:
            return initial_delay + attempt * linear_delay

        return [_interval_value(attempt) for attempt in range(retries["total"]+1)]

    def _validate_strategy_properties(self, strategy: str, **properties) -> None:
        if errors := [
            f"{prop_name} must be specified for the {strategy} retry pattern."
            for prop_name, prop_value in properties.items()
            if prop_value is None
        ]:
            error_message: str = "\n".join(errors)
            raise ValueError(error_message)

    def _calculate_delays(self) -> None:
        pattern_actions: dict[str, Callable[[], List[float]]] = {
            "custom_sequence": self.custom_sequence,
            "fixed": self.fixed,
            "exponential": self.exponential,
            "controlled_flow": self.controlled_flow,
            "linear": self.linear,
        }

        if action := pattern_actions.get(self.retry_context.config.retry_pattern):
            self.retry_context.delays = action()
        else:
            raise ValueError("Unsupported retry pattern.")

    def get_delay(self) -> float:
        return (
            self.retry_context.delays[self.retry_context.attempt - 1]
            if self.retry_context.attempt <= len(self.retry_context.delays)
            else 0.0
        )


def validate_exceptions(exc: Union[Type[E], Tuple[Type[E], ...]]) -> None:
    excs = (exc,) if isinstance(exc, type) else exc
    if errors := [
        f"{exception} must subclass Exception"
        for exception in excs
        if not issubclass(exception, Exception)
    ]:
        error_message = "\n".join(errors)
        raise ValueError(error_message)


def retry(
    exceptions: Union[Type[E], Tuple[Type[E], ...]], **kwargs: Any
) -> Callable:
    validate_exceptions(exceptions)
    retry_config: RetryConfig = RetryConfig(**kwargs)
    shared_semaphore: Optional[SharedSemaphore] = (
        retry_config.concurrency_limit
        and SharedSemaphore(max_workers=retry_config.concurrency_limit)
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

        return async_wrapper if TaskManager.is_async(func) else sync_wrapper

    return decorator
