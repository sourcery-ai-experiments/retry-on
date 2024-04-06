import inspect
import random
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
)

from src.retry_on.logging import get_logger, logging
from src.retry_on.context import RetryContext


logger: logging.Logger = get_logger(__name__)


class RetryStrategy:
    def __init__(
        self,
        context: RetryContext,
    ) -> None:
        self.retry_context: RetryContext = context

    def exponential(self) -> List[float]:
        retries: dict = {"total": self.retry_context.config.max_retries}
        jitter: Optional[float] = self.retry_context.config.jitter
        initial_delay: Optional[float] =\
            self.retry_context.config.initial_delay
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
                    jitter: float =\
                        interval * jitter_factor * (random.random() * 2 - 1)
                    combined_intervals[i] = float(max(interval + jitter, 0))

            return combined_intervals

        return _intervals(rate_limit, burst_capacity)

    def custom_sequence(self) -> Tuple[float]:
        custom_sequence: Tuple[float] =\
            self.retry_context.config.custom_sequence
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
