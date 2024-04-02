from typing import Any
import pytest
from src.retry_on.logging import logging
from src.retry_on.retry import RetryConfig

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
    (
        {},
        {'max_retries': 3, 'retry_pattern': 'controlled_flow'}
    ),
    (
        {'max_retries': 5, 'retry_pattern': 'fixed'},
        {'max_retries': 5, 'retry_pattern': 'fixed'}
    ),
    (
        {'retry_pattern': 'linear', 'linear_delay': 1.0},
        {'retry_pattern': 'linear', 'linear_delay': 1.0}
    )
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
def test_retry_config_initialization(kwargs, expected):
    retry_config = RetryConfig(**kwargs)
    for key, value in expected.items():
        attr: Any = getattr(retry_config, key)
        assert attr == value, f"{key} does not match expected value."


# Setter and Validation Tests
@pytest.mark.parametrize("attribute,value,exception", setter_test_data)
def test_retry_config_setters_and_validations(attribute, value, exception):
    retry_config = RetryConfig()
    if exception:
        with pytest.raises(exception):
            setattr(retry_config, attribute, value)
    else:
        setattr(retry_config, attribute, value)
        attr: Any = getattr(retry_config, attribute)
        assert attr == value, f"{attribute} was not set correctly."


# Custom Sequence Handling Tests
def test_valid_custom_sequence():
    retry_config = RetryConfig(custom_sequence=[1, 2, 3])
    cust_seq: list = retry_config.custom_sequence
    assert cust_seq == [1.0, 2.0, 3.0], "Custom sequence not handled correctly."


def test_invalid_custom_sequence_elements():
    retry_config = RetryConfig(custom_sequence=[1, 'invalid', 3])
    cust_seq: list = retry_config.custom_sequence
    assert cust_seq == [1.0, 3.0], "Custom sequence not correctly filtered."


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
        max_retries=2,
        initial_delay=1.0,
        max_delay=10.0,
        jitter=0.1,
        retry_pattern="fixed"
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
