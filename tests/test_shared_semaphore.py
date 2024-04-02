import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Coroutine, Any

import pytest

from src.retry_on.retry import SharedSemaphore
from src.retry_on.retry import retry


###################
# SharedSemaphore


@pytest.mark.asyncio
async def test_shared_semaphore_async_lock() -> None:
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=2)
    try:
        # Attempt to acquire the semaphore twice to ensure it's properly released.
        async with semaphore.async_lock():
            pass  # First acquisition attempt
        async with semaphore.async_lock():
            pass  # Second acquisition attempt, should work if the semaphore was released
    except Exception as e:
        pytest.fail(f"Semaphore failed to lock/unlock properly: {str(e)}")


def test_shared_semaphore_executor_context() -> None:
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=2)
    with semaphore.executor_context() as executor:
        if executor is None:
            raise ValueError("Executor context should not be None")
        assert not executor._shutdown  # The executor should be active and not shut down


@pytest.mark.asyncio
async def test_concurrency_and_threading() -> None:
    shared_semaphore: SharedSemaphore = SharedSemaphore(max_workers=2)
    attempt_counter: int = 0
    max_retries: int = 2
    max_workers: int = 2

    @retry(Exception, max_retries=max_retries, initial_delay=0.1, concurrency_limit=2)
    async def failing_task() -> Optional[None]:
        nonlocal attempt_counter
        async with shared_semaphore.async_lock():
            attempt_counter += 1
            raise Exception("Simulated task failure")

    # Run the failing task in parallel to test concurrency limit
    tasks: list[Coroutine[Any, Any, None]] = [failing_task() for _ in range(4)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future] = [executor.submit(asyncio.run, task) for task in tasks]
        for future in futures:
            with pytest.raises(Exception):
                future.result()

    # number of tasks times (the retries plus the initial attempt)
    expected_attempts: int = len(tasks) * (max_retries + 1)

    # Verify that the semaphore correctly managed concurrency
    assert (
        attempt_counter == expected_attempts
    ), f"Expected {expected_attempts} attempts, got {attempt_counter}"


def test_shared_semaphore_concurrency() -> None:
    max_workers: int = 3
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=max_workers)
    barrier: threading.Barrier = threading.Barrier(max_workers + 1)
    results: list = []

    def task() -> None:
        with semaphore.executor_context() as executor:
            results.append(executor)
            barrier.wait()

    threads: list[threading.Thread] = [
        threading.Thread(target=task) for _ in range(max_workers)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()  # Wait for all threads to reach this point
    assert len(set(results)) == 1, "Executor context should be shared across threads"
    for thread in threads:
        thread.join()


@pytest.mark.asyncio
async def test_shared_semaphore_release_on_exception() -> None:
    semaphore: SharedSemaphore = SharedSemaphore(max_workers=1)

    async def operation_with_exception() -> Optional[None]:
        async with semaphore.async_lock():
            raise ValueError("Operation failure")

    with pytest.raises(ValueError):
        await operation_with_exception()

    # If the semaphore was properly released, this operation should not block
    async with semaphore.async_lock():
        pass  # Operation successful
