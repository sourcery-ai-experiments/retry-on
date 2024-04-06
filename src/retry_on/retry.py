import asyncio
import inspect
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from functools import wraps
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union
)
from types import TracebackType
from src.retry_on.types import E, R

from src.retry_on.logging import get_logger, logging
from src.retry_on.config import RetryConfig
from src.retry_on.tasks import ConcurrencyManager, TaskManager

logger: logging.Logger = get_logger(__name__)


class RetryContext:
    def __init__(
        self,
        exceptions: Union[Type[E], Tuple[Type[E], ...]],
        config: RetryConfig,
        shared_semaphore: Optional[ConcurrencyManager] = None,
    ):
        self.allowed_exceptions: Set[Type[E]] = set(
            exceptions if isinstance(exceptions, tuple) else [exceptions]
        )
        self.config: RetryConfig = config
        self.shared_semaphore: Optional[ConcurrencyManager] = shared_semaphore
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
        if exc_type is not None and exc_type not in self.allowed_exceptions:
            self._graceful_exit()
        loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.task_manager.shutdown)

    def __enter__(self) -> "RetryContext":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._exit_message(exc_type, exc_val)
        if exc_type is not None and exc_type not in self.allowed_exceptions:
            self._graceful_exit()
        self.task_manager.shutdown()

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

    def _graceful_exit(self) -> None:
        self.task_manager.cancel_tasks()
        if not self.shared_semaphore.is_sync_shutdown:
            self.task_manager.signal_shutdown()

    async def retry_async_call(
        self, func: Callable[..., R],
        *args: Any, **kwargs: Any
    ) -> R:
        while True:
            try:
                self.attempt += 1
                logger.debug((
                    f"Attempting to execute function: {func.__name__} "
                    f"(attempt {self.attempt})"
                ))
                task: Future[R] = await self.task_manager.submit_async_task(
                        func, *args, **kwargs
                    )
                return await self.task_manager.get_async_task_result(
                    task,
                    self.config.function_timeout
                )
            except Exception as exception:
                if type(exception) in self.allowed_exceptions:
                    self.handle_sync_retry(exception)
                else:
                    self._handle_unallowed_exception(exception)
                    raise

    async def handle_async_retry(self, exception: Exception):
        self._log_failed_attempt(exception)
        if self._limit_reached():
            raise exception
        if delay := self.retry_strategy.get_delay():
            await asyncio.sleep(delay)
        if self.config.on_retry_callback:
            await self.async_execute_callback(exception)
            
    def retry_sync_call(
        self, func: Callable[..., R], *args: Any, **kwargs: Any
    ) -> R:
        while True:
            try:
                self.attempt += 1
                return self.task_manager.submit_sync_task(
                    func, *args, timeout=self.config.function_timeout, **kwargs
                )
            except Exception as exception:
                if type(exception) in self.allowed_exceptions:
                    self.handle_sync_retry(exception)
                else:
                    self._handle_unallowed_exception(exception)
                    raise

    def handle_sync_retry(self, exception: Exception):
        self._log_failed_attempt(exception)
        if self._limit_reached():
            raise exception
        if delay := self.retry_strategy.get_delay():
            time.sleep(delay)
        if self.config.on_retry_callback:
            self.sync_execute_callback(exception)

    def _handle_unallowed_exception(self, exception: Exception) -> None:
        self._log_unallowed_exception(exception)
        self._graceful_exit()
        raise exception
    
    def _log_unallowed_exception(self, exception: Exception) -> None:
        logger.log(
            logging.ERROR, (
                "Function raised an unallowed exception "
                f"during attempt {self.attempt}/{self.config.max_retries + 1}."
                " Function will not be retried. "
                f"Error: {exception}."
            ),
        )

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
                    def run_async_coro(
                        coro: Callable,
                        loop: asyncio.AbstractEventLoop
                    ):
                        try:
                            return loop.run_until_complete(coro)
                        except RuntimeError:
                            loop: asyncio.AbstractEventLoop =\
                                asyncio.new_event_loop()
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

    def _log_failed_attempt(self, exception: Exception) -> None:
        logger.log(logging.WARNING, (
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
        burst_capacity: Optional[int] = \
            self.retry_context.config.burst_capacity
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

    def custom_sequence(self) -> Tuple[float]:
        custom_sequence: Tuple[float] = self.retry_context.config.custom_sequence
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
        initial_delay: Optional[float] = \
            self.retry_context.config.initial_delay
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

        return [
            _interval_value(attempt) for attempt in range(retries["total"]+1)
            ]

    def _validate_strategy_properties(
        self,
        strategy: str,
        **properties
    ) -> None:
        if errors := [
            f"{prop_name} must be specified for the {strategy} retry pattern."
            for prop_name, prop_value in properties.items()
            if prop_value is None
        ]:
            error_message: str = "\n".join(errors)
            raise ValueError(error_message)

    def _calculate_delays(self) -> None:
        pattern_actions: dict[str, Callable] = {
            "custom_sequence": self.custom_sequence,
            "fixed": self.fixed,
            "exponential": self.exponential,
            "controlled_flow": self.controlled_flow,
            "linear": self.linear,
        }

        if action := pattern_actions.get(
            self.retry_context.config.retry_pattern
        ):
            self.retry_context.delays = action()
            logger.debug(f"Calculated delays: {self.retry_context.delays}")
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

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            shared_semaphore: Optional[ConcurrencyManager] = (
                ConcurrencyManager(max_workers=retry_config.concurrency_limit)
                if retry_config.concurrency_limit
                else None
            )
            async with RetryContext(
                exceptions, retry_config, shared_semaphore=shared_semaphore
            ) as context:
                return await context.retry_async_call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> R:
            shared_semaphore: Optional[ConcurrencyManager] = (
                ConcurrencyManager(max_workers=retry_config.concurrency_limit)
                if retry_config.concurrency_limit
                else None
            )            
            with RetryContext(
                exceptions, retry_config, shared_semaphore=shared_semaphore
            ) as context:
                return context.retry_sync_call(func, *args, **kwargs)

        return async_wrapper if TaskManager.is_async(func) else sync_wrapper

    return decorator
