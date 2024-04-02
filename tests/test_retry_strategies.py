from typing import Optional, Union

from unittest.mock import patch
import pytest

from src.retry_on.retry import RetryConfig, RetryContext
from src.retry_on.retry import retry

##################
# Retry Strategy

@pytest.mark.parametrize(
    "max_retries, jitter, initial_delay, max_delay, expected, test_id",
    [
        (0, 0.0, 1.0, 10.0, [], "no_retries"),
        (1, 0.0, 1.0, 10.0, [1.0], "single_retry_no_jitter"),
        (2, 0.0, 1.0, 10.0, [1.0, 2.0], "two_retries_no_jitter"),
        (3, 0.5, 1.0, 20.0, [1.0, 3.125, 15.625], "three_retries_with_jitter"),
        (3, 0.0, 0.0, 10.0, [0.0, 0.0, 0.0], "three_retries_no_initial_delay"),
        (3, 0.5, 1.0, 3.0, [1.0, 3.0, 3.0], "three_retries_with_max_delay"),
    ],
)
def test_retry_pattern_default_backoff(
    max_retries, jitter, initial_delay, max_delay, expected, test_id
):
    config = RetryConfig(
        max_retries=max_retries,
        jitter=jitter,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retry_pattern="exponential",
    )
    retry_context = RetryContext(Exception, config)

    def mock_random_uniform(a, b):
        return (a + b) / 2

    with patch("random.uniform", mock_random_uniform):
        result = retry_context.retry_strategy.exponential()
        assert result == expected, f"[{test_id}] Expected {expected} but got {result}"


@pytest.mark.parametrize(
    "max_retries, jitter, initial_delay, max_delay, expected, test_id",
    [
        (3, 0.0, 1.0, 10.0, [1.0, 2.0, 8.0], "happy-path-no-jitter"),
        (1, 0.0, 2.0, 10.0, [2.0], "edge-case-no-jitter"),
        (0, 0.0, 1.0, 10.0, [], "edge-case-no-retries-no-jitter"),
        (
            5,
            0.0,
            0.1,
            0.5,
            [0.1, 0.2, 0.5, 0.5, 0.5],
            "happy-path-max-delay-reached-no-jitter",
        ),
        (
            2,
            0.0,
            5.0,
            5.0,
            [5.0, 5.0],
            "edge-case-initial-delay-equals-max-delay-no-jitter",
        ),
        # Assuming negative values are error cases
        (-1, 0.5, 1.0, 10.0, ValueError, "error-case-negative-max-retries"),
        (3, -0.5, 1.0, 10.0, ValueError, "error-case-negative-jitter"),
        (3, 0.5, -1.0, 10.0, ValueError, "error-case-negative-initial-delay"),
        (3, 0.5, 1.0, -10.0, ValueError, "error-case-negative-max-delay"),
    ],
)
def test_retry_pattern_default_backoff_without_jitter(
    max_retries: int,
    jitter: float,
    initial_delay: float,
    max_delay: float,
    expected: Union[list, type],
    test_id: str,
) -> None:
    if isinstance(expected, type) and expected is ValueError:
        with pytest.raises(expected):
            _ = RetryConfig(
                max_retries=max_retries,
                jitter=jitter,
                initial_delay=initial_delay,
                max_delay=max_delay,
                retry_pattern="exponential",
            )
    elif isinstance(expected, list):
        config = RetryConfig(
            max_retries=max_retries,
            jitter=jitter,
            initial_delay=initial_delay,
            max_delay=max_delay,
            retry_pattern="exponential",
        )
        retry_context = RetryContext(Exception, config)
        result = retry_context.retry_strategy.exponential()
        assert result == expected, f"[{test_id}] Expected {expected} but got {result}"


@pytest.mark.parametrize(
    "rate_limit, burst_capacity, max_retries, jitter, expected_intervals, test_id",
    [
        # Happy path tests
        (0.5, 3, 5, 0.25, [0, 0, 0, 2.0, 4.0], "happy-path-low-retries"),
        (1.0, 2, 4, 0.0, [0, 0, 1.0, 2.0], "happy-path-no-jitter"),
        (2.0, 1, 3, 0.5, [0, 0.5, 1.0], "happy-path-high-rate-limit"),
        # Edge cases
        (0.5, 0, 3, 0.25, [2.0, 4.0, 6.0], "edge-case-zero-burst-capacity"),
        (0.1, 5, 5, 0.25, [0, 0, 0, 0, 0], "edge-case-burst-equals-retries"),
        (0.5, 5, 3, 0.25, [0, 0, 0], "edge-case-burst-exceeds-retries"),
    ],
    ids=str,
)
def test_retry_pattern_controlled_flow(
    rate_limit: float,
    burst_capacity: int,
    max_retries: int,
    jitter: float,
    expected_intervals: Union[list, type],
    test_id: str,
):
    config = RetryConfig(
        max_retries=max_retries,
        jitter=jitter,
        rate_limit=rate_limit,
        burst_capacity=burst_capacity,
        retry_pattern="controlled_flow",
    )
    retry_context = RetryContext(Exception, config)

    def mock_random() -> float:
        return 0.5

    with patch("random.random", mock_random):
        if isinstance(expected_intervals, type) and issubclass(
            expected_intervals, Exception
        ):
            with pytest.raises(expected_intervals):
                retry_context.retry_strategy.controlled_flow()
        else:
            intervals = retry_context.retry_strategy.controlled_flow()

    if not isinstance(expected_intervals, type):
        assert (
            intervals == expected_intervals
        ), f"[{test_id}] Expected {expected_intervals} but got {intervals}"


@pytest.mark.parametrize(
    "custom_sequence, expected, test_id",
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], "custom_sequence"),
    ],
)
def test_retry_pattern_custom_sequence(
    custom_sequence: list[float], expected: list[float], test_id: str
) -> None:
    config = RetryConfig(
        custom_sequence=custom_sequence,
        retry_pattern="custom_sequence",
    )
    retry_context = RetryContext(Exception, config)
    result = retry_context.retry_strategy.custom_sequence()
    assert result == expected, f"[{test_id}] Expected {expected} but got {result}"


@pytest.mark.asyncio
async def test_retry_pattern_application() -> None:
    call_counts: dict[str, int] = {"fixed": 0, "exponential": 0, "custom_sequence": 0}
    max_retries: int = 3
    expected_call_counts: int = max_retries + 1
    
    async def failing_task() -> Optional[None]:
        raise ValueError("Simulated task failure")

    @retry(ValueError, max_retries=max_retries, initial_delay=0.1, retry_pattern="fixed", fixed_delay=0.1)
    async def fixed_retry_task() -> Optional[None]:
        call_counts["fixed"] += 1
        await failing_task()

    @retry(ValueError, max_retries=max_retries, initial_delay=0.1, retry_pattern="exponential")
    async def exponential_retry_task() -> Optional[None]:
        call_counts["exponential"] += 1
        await failing_task()

    @retry(
        ValueError,
        max_retries=max_retries,
        custom_sequence=[0.1, 0.2, 0.4],
        retry_pattern="custom_sequence",
    )
    async def custom_sequence_retry_task() -> Optional[None]:
        call_counts["custom_sequence"] += 1
        await failing_task()

    with pytest.raises(ValueError):
        await fixed_retry_task()
    assert (
        call_counts["fixed"] == 4
    ), f"Fixed retry pattern did not execute the expected number of times (expected: {expected_call_counts}, actual: {call_counts['fixed']})"

    with pytest.raises(ValueError):
        await exponential_retry_task()
    assert (
        call_counts["exponential"] == 4
    ), f"Exponential retry pattern did not execute the expected number of times (expected: {expected_call_counts}, actual: {call_counts['exponential']})"

    with pytest.raises(ValueError):
        await custom_sequence_retry_task()
    assert (
        call_counts["custom_sequence"] == 4
    ), f"Custom sequence retry pattern did not execute the expected number of times (expected: {expected_call_counts}, actual: {call_counts['custom_sequence']})"
