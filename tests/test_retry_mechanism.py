from typing import (
    NoReturn,
    Optional
)
from unittest.mock import Mock, AsyncMock, MagicMock

import pytest
from openai import APIError, APITimeoutError, RateLimitError
from typing_extensions import Literal

from tests.utils import ExceptionSimulator, NetworkSimulator
from src.retry_on.retry import retry


###################
# Retry Mechanism

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "should_raise, expected_attempts, test_scenario",
    [
        (False, 1, "immediate_success_async"),
        (False, 2, "success_before_max_retries_async"),
        (True, 3, "failure_at_max_retries_async"),
    ],
)
async def test_async_retry_behavior(
    should_raise, expected_attempts, test_scenario
) -> None:
    simulator: ExceptionSimulator = ExceptionSimulator()
    side_effects: list[Exception] = [
        simulator.create_rate_limit_error() for _ in range(expected_attempts - 1)
    ]
    if should_raise:
        side_effects.append(simulator.create_rate_limit_error())
    else:
        side_effects.append(True)  # type: ignore

    func: AsyncMock = AsyncMock(side_effect=side_effects)

    @retry(
        exceptions=(RateLimitError, APIError, APITimeoutError),
        max_retries=expected_attempts - 1,
    )
    async def async_test_func() -> NoReturn:
        await func()

    if should_raise:
        with pytest.raises((RateLimitError, APIError, APITimeoutError)):
            await async_test_func()
    else:
        await async_test_func()

    assert func.call_count == expected_attempts


@pytest.mark.parametrize(
    "should_raise, expected_attempts, test_scenario",
    [
        (False, 1, "immediate_success"),
        (False, 2, "success_before_max_retries"),
        (True, 3, "failure_at_max_retries"),
    ],
)
def test_sync_retry_behavior(should_raise, expected_attempts, test_scenario) -> None:
    simulator: ExceptionSimulator = ExceptionSimulator()
    side_effects: list[Exception] = [
        simulator.create_rate_limit_error() for _ in range(expected_attempts - 1)
    ]
    if should_raise:
        side_effects.append(simulator.create_rate_limit_error())
    else:
        side_effects.append(True)  # type: ignore

    func: MagicMock = MagicMock(side_effect=side_effects)

    @retry(exceptions=RateLimitError, max_retries=expected_attempts - 1)
    def sync_test_func() -> None:
        func()

    if should_raise:
        with pytest.raises(RateLimitError):
            sync_test_func()
    else:
        sync_test_func()

    assert func.call_count == expected_attempts, (
        f"{test_scenario}: Expected {expected_attempts} "
        f"calls, got {func.call_count}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "should_raise, max_attempts, test_scenario",
    [
        (False, 1, ExceptionSimulator.create_api_error),
        (False, 2, ExceptionSimulator.create_api_timeout_error),
        (True, 3, ExceptionSimulator.create_rate_limit_error),
    ],
)
async def test_retry_on_api_error_async(
    should_raise, max_attempts, test_scenario
) -> None:
    side_effects: list[Exception | bool] = [
        test_scenario() for _ in range(max_attempts)
    ]
    if should_raise:
        side_effects.append(test_scenario())
    else:
        side_effects.append(True)

    func: AsyncMock = AsyncMock(side_effect=side_effects)

    @retry(exceptions=(RateLimitError, APIError, APITimeoutError))
    async def async_test_func() -> NoReturn:
        await func()

    if should_raise:
        with pytest.raises((RateLimitError, APIError, APITimeoutError)):
            await async_test_func()
    else:
        await async_test_func()

    assert (
        func.call_count == max_attempts + 1
    ), f"Expected {max_attempts} attempts, got {func.call_count}"


@pytest.mark.parametrize(
    "should_raise, max_attempts, test_scenario",
    [
        (False, 1, ExceptionSimulator.create_api_error),
        (False, 2, ExceptionSimulator.create_api_timeout_error), 
        (True, 3, ExceptionSimulator.create_rate_limit_error)
    ]
)
def test_retry_on_api_error_sync(
    should_raise, max_attempts, test_scenario
):

    side_effects = [test_scenario() for _ in range(max_attempts)]
    if should_raise:
        side_effects.append(test_scenario())
    else:
        side_effects.append(True)

    func = Mock(side_effect=side_effects)

    @retry(exceptions=(RateLimitError, APIError, APITimeoutError)) 
    def sync_test_func():
        func()

    if should_raise:
        with pytest.raises((RateLimitError, APIError, APITimeoutError)):
            sync_test_func()
    else:
        sync_test_func()

    assert func.call_count == max_attempts + 1


@pytest.mark.asyncio
async def test_no_retry_on_success() -> None:
    func: AsyncMock = AsyncMock(return_value=Literal["Success"])

    @retry(exceptions=RateLimitError, max_retries=2, initial_delay=0.1, jitter=0)
    async def async_test_func() -> Literal["Success"]:
        return await func()

    result: Literal["Success"] = await async_test_func()
    assert result == Literal["Success"]
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_max_retry_exceeded() -> None:
    func: AsyncMock = AsyncMock(
        side_effect=ExceptionSimulator.create_rate_limit_error()
    )
    max_retries: int = 3
    attempts: int = 1 + max_retries  # 1 initial attempt + 3 retries

    @retry(
        exceptions=RateLimitError, max_retries=max_retries, initial_delay=0.1, jitter=0
    )
    async def async_test_func() -> Optional[Exception]:
        await func()

    with pytest.raises(RateLimitError):
        await async_test_func()

    assert func.call_count == attempts


@pytest.mark.asyncio
async def test_integration_retry_mechanism() -> None:
    async_call_count: int = 0
    sync_call_count: int = 0

    async def async_operation() -> Literal["Async success"]:
        nonlocal async_call_count
        async_call_count += 1
        if async_call_count < 3:
            raise ValueError("Async failure")
        return "Async success"

    def sync_operation() -> Literal["Sync success"]:
        nonlocal sync_call_count
        sync_call_count += 1
        if sync_call_count < 2:
            raise ValueError("Sync failure")
        return "Sync success"

    @retry(ValueError, max_retries=4, initial_delay=0.1, jitter=0)
    async def async_wrapper() -> Optional[Literal["Async success"]]:
        return await async_operation()

    @retry(ValueError, max_retries=3, initial_delay=0.1, jitter=0)
    def sync_wrapper() -> Optional[Literal["Sync success"]]:
        return sync_operation()

    async_result: Optional[Literal["Async success"]] = await async_wrapper()
    sync_result: Optional[Literal["Sync success"]] = sync_wrapper()

    assert async_result == "Async success"
    assert async_call_count == 3
    assert sync_result == "Sync success"
    assert sync_call_count == 2


@pytest.mark.asyncio
async def test_retry_on_network_failure() -> None:
    network_simulator: NetworkSimulator = NetworkSimulator()

    @retry(ConnectionError, max_retries=3, initial_delay=0.1, jitter=0)
    async def make_network_call() -> Optional[Literal["Success"]]:
        return await network_simulator.flaky_network_call()

    result: Optional[Literal["Success"]] = await make_network_call()
    assert result == "Success", "The network call should succeed after retries"
    assert (
        network_simulator.attempt_count == 3
    ), "The function should retry exactly 2 times before succeeding"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config,expected_attempts,exception_generator",
    [
        (
            {
                "max_retries": 2,
                "initial_delay": 0.1,
                "retry_pattern": "fixed",
                "fixed_delay": 0.1,
            },
            2,
            ExceptionSimulator.create_rate_limit_error,
        ),
        (
            {"max_retries": 3, "initial_delay": 0.1, "retry_pattern": "exponential"},
            3,
            ExceptionSimulator.create_api_timeout_error,
        ),
        (
            {
                "max_retries": 4,
                "custom_sequence": [0.1, 0.2, 0.4],
                "retry_pattern": "custom_sequence",
            },
            4,
            ExceptionSimulator.create_api_error,
        ),
    ],
)
async def test_retry_with_custom_config(
    config, expected_attempts, exception_generator
) -> None:
    side_effects: list[Exception] = [
        exception_generator() for _ in range(expected_attempts - 1)
    ]
    side_effects.append(True)  # type: ignore # Assume success on last attempt

    func: AsyncMock = AsyncMock(side_effect=side_effects)

    @retry(exceptions=(RateLimitError, APIError, APITimeoutError), **config)
    async def async_test_func_with_custom_config() -> None:
        await func()

    await async_test_func_with_custom_config()

    assert func.call_count == expected_attempts