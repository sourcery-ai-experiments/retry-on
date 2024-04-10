import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, NoReturn
from concurrent.futures import Future

import pytest

from tests.utils import ExceptionSimulator, transient_fail_func
from src.retry_on.retry import retry
from typing import List

###############
# Concurrency


@pytest.mark.asyncio
async def test_async_func_concurrency_limit_max_retries_exceeded() -> None:
    concurrent_executions: int = 0
    max_concurrent_executions: int = 0
    concurrency_limit: int = 2
    max_retries: int = 5
    initial_delay: float = 0.1

    @retry(
        exceptions=(type(ExceptionSimulator.create_rate_limit_error()),),
        max_retries=max_retries,
        initial_delay=initial_delay,
        concurrency_limit=concurrency_limit,
    )
    async def failing_func() -> NoReturn:
        nonlocal concurrent_executions, max_concurrent_executions
        concurrent_executions += 1

        max_concurrent_executions = max(
            max_concurrent_executions, concurrent_executions
        )
        await asyncio.sleep(0.01)

        concurrent_executions -= 1
        raise ExceptionSimulator.create_rate_limit_error()

    tasks: List[asyncio.Task] = [
        asyncio.create_task(failing_func())
        for _ in range(4)
    ]
    await asyncio.gather(
        *tasks,
        return_exceptions=True
    )

    assert (
        concurrent_executions == 0
    ), "There should be no concurrent executions at the end."
    assert (
        max_concurrent_executions <= concurrency_limit
    ), (
        f"Concurrency limit of {concurrency_limit} exceeded. "
        f"Max concurrent executions: {max_concurrent_executions}"
    )


def test_sync_func_concurrency_limit_max_retries_exceeded() -> None:
    concurrent_executions: int = 0
    max_concurrent_executions: int = 0
    concurrency_limit = 2
    max_retries = 5
    initial_delay = 0.1

    @retry(
        exceptions=(Exception,),
        max_retries=max_retries,
        initial_delay=initial_delay,
        concurrency_limit=concurrency_limit
    )
    def limited_func():
        nonlocal concurrent_executions, max_concurrent_executions
        concurrent_executions += 1
        max_concurrent_executions = max(
            max_concurrent_executions,
            concurrent_executions
        )
        time.sleep(0.01)  # simulate operation
        concurrent_executions -= 1
        raise Exception("Forced retry")

    tasks: List[Callable] = [limited_func for _ in range(4)]

    with pytest.raises(Exception):
        for task in tasks:
            task()

    assert concurrent_executions == 0,\
        "Concurrent executions should be 0 at the end."
    assert max_concurrent_executions <= concurrency_limit, \
        f"Concurrent executions exceeded the limit of {concurrency_limit}."


@pytest.mark.asyncio
async def test_async_func_concurrency_limit_results() -> None:
    max_retries: int = 2
    worker_count: int = 4
    transient_failures: int = 0

    sync_func: Callable = transient_fail_func(max_failures=transient_failures)

    @retry(
        Exception,
        max_retries=max_retries
    )
    async def decorated_func() -> str:
        return await asyncio.to_thread(sync_func)

    tasks: List[asyncio.Task] = [decorated_func() for _ in range(worker_count)]
    results: List[str] = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            print(result.__str__())
        else:
            print(result)

    assert all(result == "Success" for result in results),\
        "Not all functions succeeded as expected"


def test_sync_func_concurrency_limit_results() -> None:
    max_retries: int = 2
    worker_count: int = 4
    transient_failures: int = 1

    sync_func: Callable = transient_fail_func(max_failures=transient_failures)

    @retry(
        ExceptionSimulator.create_rate_limit_error().__class__,
        max_retries=max_retries
    )
    def decorated_func() -> str:
        return sync_func()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: List[Future] = [
            executor.submit(decorated_func) for _ in range(worker_count)
        ]
        results: List[str] = [
            future.result() for future in as_completed(futures)
        ]

    assert all(
        result == "Success" for result in results
    ), "Not all functions succeeded as expected"
