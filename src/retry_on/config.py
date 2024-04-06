from typing import (
    Any,
    Callable,
    FrozenSet,
    List,
    Dict,
    Optional,
    Tuple
)

from src.retry_on.logging import get_logger, logging

logger: logging.Logger = get_logger(__name__)


class RetryConfig:
    SUPPORTED_RETRY_PATTERNS: FrozenSet = frozenset([
       "controlled_flow",
       "fixed",
       "exponential",
       "custom_sequence",
       "linear"
    ])

    RETRY_PATTERN_DEFAULT_VALUES: Dict[str, dict] = {
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
            "custom_sequence": (1, 2, 3),
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
        self.log_level = kwargs.get("log_level", logging.ERROR)
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
        for prop in self.RETRY_PATTERN_DEFAULT_VALUES[strategy]:
            if getattr(self, prop) is None:
                setattr(self, prop, self.RETRY_PATTERN_DEFAULT_VALUES[strategy][prop])

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
    def _get_custom_sequence(kwargs, key) -> Optional[Tuple[float]]:
        value: Any = kwargs.get(key)
        if isinstance(value, list):
            return (
                tuple(float(element) for element in value if isinstance(element, (int, float)))
            )
        if isinstance(value, tuple):
            return [float(element) for element in value if isinstance(element, (int, float))]
        return (float(value), ) if isinstance(value, (int, float)) else None

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
    def custom_sequence(self) -> Optional[Tuple[float]]:
        return self._custom_sequence

    @custom_sequence.setter
    def custom_sequence(self, value: Optional[Tuple[float]]) -> None:
        if value is not None and not isinstance(value, (list, tuple)):
            raise TypeError("custom_sequence must be a list or a tuple")
        if value is not None and not all(isinstance(n, (int, float)) for n in value):
            raise ValueError("custom_sequence elements must be floats or integers if provided")
        self._custom_sequence = value if isinstance(value, tuple) else tuple(value) if value else None

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
        # if value > 30:
        #     logger.warning(
        #         "log_level is set to a value greater than 30. "
        #         "This may cause excessive logging and may impact performance."
        #     )
        # if value >= logging.CRITICAL:
        #     logger.warning(
        #         "log_level is set to a value greater than or equal to logging.CRITICAL. "
        #         "This may cause unexpected behavior."
        #     )

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
        default_properties = self.RETRY_PATTERN_DEFAULT_VALUES.get(new_retry_pattern)
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
