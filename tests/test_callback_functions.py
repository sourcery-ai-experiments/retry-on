import asyncio
from typing import Type

import pytest

from src.retry_on.retry import retry

######################
# Callback Functions


@pytest.mark.parametrize("async_func", [False, True])
def test_callback_after_retry_failure_with_sync_callback(async_func):
    call_count = {"main": 0, "callback": 0}

    def callback_function(exception: Type[BaseException]):
        call_count["callback"] += 1

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=callback_function)
    async def async_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=callback_function)
    def sync_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    main_function = async_main_function if async_func else sync_main_function

    # Execute the decorated function and verify behavior
    if async_func:
        with pytest.raises(ValueError):
            asyncio.run(main_function())
    else:
        with pytest.raises(ValueError):
            main_function()

    assert call_count["main"] == 3  # Initial call + 2 retries
    assert call_count["callback"] == 2  # Callback after each retry


@pytest.mark.parametrize("async_func", [False, True])
@pytest.mark.asyncio
async def test_callback_after_retry_failure_with_async_callback(async_func):
    call_count = {"main": 0, "callback": 0}

    async def async_callback_function(exception: Type[BaseException]) -> None:
        call_count["callback"] += 1

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=async_callback_function)
    async def async_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    @retry(exceptions=ValueError, max_retries=2, on_retry_callback=async_callback_function)
    def sync_main_function():
        call_count["main"] += 1
        raise ValueError("Triggering retry")

    main_function = async_main_function if async_func else sync_main_function

    # Execute the decorated function and verify behavior
    if async_func:
        with pytest.raises(ValueError):
            await main_function()
    else:
        # Special handling to execute an async callback from a sync function context
        with pytest.raises(ValueError):
            def sync_wrapper():
                asyncio.run(main_function())
            sync_wrapper()

    assert call_count["main"] == 3  # Initial call + 2 retries
    assert call_count["callback"] == 2  # Callback after each retry