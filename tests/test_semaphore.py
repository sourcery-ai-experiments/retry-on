# pylint: disable=redefined-outer-name
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Coroutine, Any, Callable, List, Union, NoReturn

import pytest
import aiohttp
import httpx

from src.retry_on.retry import retry
from src.retry_on.tasks import ConcurrencyManager
from tests.utils import semaphore_factory  # pylint: disable=unused-import


###################
# SEMAPHORE INITIALIZATION


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_workers",
    [0, -5, 1, 1000000],
    ids=["0_workers", "-5_workers", "1_worker", "1_million_workers"]
)
def test_semaphore_init(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests the initialization of SharedSemaphore."""
    if max_workers <= 0:
        with pytest.raises(ValueError):
            semaphore_factory(max_workers)
    else:
        semaphore: ConcurrencyManager = semaphore_factory(max_workers)
        assert semaphore.sync_max_workers == max_workers, \
            "Max workers does not match the expected value."


###################
# SEMAPHORE ASYNC LOCK MANAGEMENT


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [2], ids=["2_workers"])
@pytest.mark.asyncio
async def test_semaphore_async_lock(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests async lock management in SharedSemaphore."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    try:
        async with semaphore.async_lock():
            pass
        async with semaphore.async_lock():
            pass
    except Exception as e:
        pytest.fail(f"Semaphore failed to lock/unlock properly: {str(e)}")

    # Test semaphore can still be acquired after handling exception
    with pytest.raises(RuntimeError):
        async with semaphore.async_lock():
            raise RuntimeError("Simulated exception")


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [2], ids=["2_workers"])
def test_semaphore_sync_lock(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests sync lock management in SharedSemaphore."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    def task() -> None:
        with semaphore.executor_context() as executor:
            executor.submit(lambda: None)
            executor.submit(lambda: None)

    try:
        task()
    except Exception as e:
        pytest.fail(f"Semaphore failed to lock/unlock properly: {str(e)}")

    # Test semaphore can still be acquired after handling exception
    with pytest.raises(RuntimeError):
        with semaphore.executor_context() as executor:
            executor.submit(lambda: None)
            raise RuntimeError("Simulated exception")


###################
# EXECUTOR CONTEXT MANAGEMENT


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [2], ids=["2_workers"])
def test_executor_context_manager_management(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests executor context management within SharedSemaphore."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    with semaphore.executor_context() as executor:
        if executor is None:
            raise ValueError("Executor context should not be None")

        assert semaphore.is_sync_shutdown is False, "Executor should be active"
        assert isinstance(executor, ThreadPoolExecutor), \
            "Executor should be a ThreadPoolExecutor"

    # Test executor context after shutdown
    semaphore.shutdown()
    with pytest.raises(RuntimeError):
        with semaphore.executor_context() as executor:
            pass


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_workers, tasks",
    [(1, 3)],
    ids=["1_worker_3_tasks"]
)
def test_executor_context_manager_execute_tasks(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests task execution within SharedSemaphore's executor context."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    def task() -> None:
        pass

    with semaphore.executor_context() as executor:
        futures: List[Future] = [executor.submit(task) for _ in range(tasks)]
        for future in futures:
            future.result()


###################
# TASK EXECUTION


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [3], ids=["3_workers"])
@pytest.mark.asyncio
async def test_semaphore_async_task_execution_of_failing_and_succeeding_tasks(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests direct interactions with semaphore and task execution."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    async def long_running_task() -> NoReturn:
        async with semaphore.async_lock():
            await asyncio.sleep(0.1)

    async def failing_task() -> NoReturn:
        async with semaphore.async_lock():
            raise ValueError("Simulated task failure")

    # Execute tasks with varying outcomes
    tasks: List[Coroutine] = [
        long_running_task(),
        failing_task(),
        long_running_task()
    ]
    with pytest.raises(ValueError):
        await asyncio.gather(*tasks)


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [3], ids=["3_workers"])
def test_semaphore_sync_task_execution_of_failing_and_succeeding_tasks(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests direct interactions with semaphore and task execution."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    def long_running_task() -> None:
        with semaphore.executor_context():
            time.sleep(0.1)

    def failing_task() -> None:
        with semaphore.executor_context():
            raise ValueError("Simulated task failure")

    # Execute tasks with varying outcomes
    task_list: List[Callable] = [
        long_running_task,
        failing_task,
        long_running_task
    ]
    with pytest.raises(ValueError):
        for task in task_list:
            task()


###################
# SEMAPHORE EXCEPTION HANDLING


@pytest.mark.parametrize(
    "max_workers, tasks",
    [
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3)
    ],
    ids=[
        "1_worker_1_task",
        "1_worker_2_tasks",
        "2_workers_2_tasks",
        "2_workers_3_tasks"
    ]
)
@pytest.mark.asyncio
async def test_semaphore_exception_handling_during_async_tasks(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests semaphore behavior when an exception is raised
    during task execution with multiple workers."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    async def task_that_raises_exception() -> None:
        raise ValueError("Simulated task failure")

    task_list: List[Coroutine] = [
        task_that_raises_exception() for _ in range(tasks)
    ]
    with pytest.raises(ValueError):
        await asyncio.gather(*task_list)

    async with semaphore.async_lock():
        assert True


@pytest.mark.parametrize(
    "max_workers, tasks",
    [
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3)
    ],
    ids=[
        "1_worker_1_task",
        "1_worker_2_tasks",
        "2_workers_2_tasks",
        "2_workers_3_tasks"
    ]
)
def test_semaphore_exception_handling_during_sync_tasks(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests semaphore behavior when an exception is raised
    during task execution with multiple workers."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    def task_that_raises_exception() -> None:
        raise ValueError("Simulated task failure")

    task_list: List[Callable] = [
        task_that_raises_exception for _ in range(tasks)
    ]
    with semaphore.executor_context() as executor:
        futures = [executor.submit(task) for task in task_list]
        for future in futures:
            with pytest.raises(ValueError):
                future.result()

    with semaphore.executor_context() as executor:
        assert True


###################
# EXECUTOR SHUTDOWN BEHAVIOR AND FLAG ACCURACY


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [1], ids=["1_worker"])
def test_executor_shutdown_behavior_and_flag(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests executor shutdown behavior
    and the accuracy of the is_shutdown flag."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    def task() -> None:
        pass

    with semaphore.executor_context() as executor:
        assert semaphore.is_sync_shutdown is False, "Executor should be active"
        executor.submit(task)

    semaphore.shutdown()
    assert semaphore.is_sync_shutdown is True, "Executor should be inactive"


###################
# SEMAPHORE REENTRANCY AND LOCKING MECHANICS


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    'max_workers',
    [1, 1000000],
    ids=["1_worker", "1_million_workers"]
)
async def test_async_semaphore_locking_mechanics(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests basic semaphore locking mechanics and edge cases."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    async def task() -> None:
        value_before: int = semaphore.async_value
        async with semaphore.async_lock():
            await asyncio.sleep(0.1)
            assert semaphore.async_value == value_before - 1,\
                "Lock was not acquired correctly"
        assert semaphore.async_value == value_before,\
            "Lock was not released correctly"

    await task()


@pytest.mark.unit
@pytest.mark.parametrize(
    'max_workers',
    [1, 1000000],
    ids=["1_worker", "1_million_workers"]
)
def test_sync_semaphore_locking_mechanics(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests basic semaphore locking mechanics and edge cases."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    def task() -> None:
        value_before: int = semaphore.sync_value
        with semaphore.executor_context() as executor:
            executor.submit(lambda: None)
            assert semaphore.sync_value == value_before - 1,\
                "Lock was not acquired correctly"
        assert semaphore.sync_value == value_before,\
            "Lock was not released correctly"
    task()


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [2], ids=["2_workers"])
@pytest.mark.asyncio
async def test_async_nested_semaphore_lock_and_reentrancy(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests nesting of semaphore async locks."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    async def reentrant_task() -> str:
        async with semaphore.async_lock():
            async with semaphore.async_lock():
                return "Nested lock acquired"

    result: str = await reentrant_task()
    assert result == "Nested lock acquired",\
        "Failed to re-acquire a lock in a nested context."


@pytest.mark.unit
@pytest.mark.parametrize("max_workers", [2], ids=["2_workers"])
def test_sync_nested_semaphore_lock_and_reentrancy(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests basic semaphore locking mechanics in a synchronous context."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    def reentrant_task() -> str:
        with semaphore.executor_context() as executor:
            executor.submit(lambda: None)
            with semaphore.executor_context() as nested_executor:
                nested_executor.submit(lambda: None)
            return "Locks acquired and released"

    result: str = reentrant_task()
    assert result == "Locks acquired and released",\
        "Failed to acquire and release locks in a synchronous context."

###################
# SEMAPHORE CONCURRENCY AND THREADING


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_workers, tasks",
    [(3, 3)],
    ids=["3_workers_3_tasks"]
)
@pytest.mark.asyncio
async def test_async_semaphore_concurrency_unique_executor(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests semaphore concurrency with multiple worker threads
    in an async context."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)
    event: asyncio.Event = asyncio.Event()
    results: List = []
    counter: int = 0

    async def task() -> None:
        nonlocal counter
        async with semaphore.async_lock():
            results.append(semaphore._executor)
            counter += 1
            if counter == tasks:
                event.set()
            await event.wait()

    tasks_list: List[Coroutine] = [
        task() for _ in range(tasks)
    ]
    await asyncio.gather(*tasks_list)
    unique_executors: int = len(set(results))

    assert unique_executors == 1, \
        (
            "Expected a single shared executor context, "
            f"found {unique_executors}"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_workers, tasks",
    [(3, 3)],
    ids=("3_workers_3_tasks",)
)
def test_sync_semaphore_concurrency_unique_executor(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests semaphore concurrency with multiple worker threads."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)
    barrier: threading.Barrier = threading.Barrier(max_workers + 1)
    results: List = []

    def task() -> None:
        with semaphore.executor_context() as executor:
            results.append(executor)
            barrier.wait()

    threads: List[threading.Thread] = [
        threading.Thread(target=task) for _ in range(tasks)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()  # Wait for all threads to reach this point
    unique_executors: int = len(set(results))

    assert unique_executors == 1, \
        (
            "Expected a single shared executor context, "
            f"found {unique_executors}"
        )
    for thread in threads:
        thread.join()


###################
# EFFECTIVENESS OF RETRY LOGIC


@pytest.mark.integration
@pytest.mark.parametrize("max_workers", [2], ids=("2_workers",))
@pytest.mark.asyncio
async def test_async_semaphore_retry_logic_effectiveness_with_nested_contexts(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests the effectiveness of retry logic in asynchronous tasks."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    @retry(
        Exception,
        max_retries=3,
        initial_delay=0.1,
        concurrency_limit=max_workers
    )
    async def fetch_data() -> str:
        async with semaphore.async_lock():
            async with aiohttp.ClientSession() as session:
                async with session.get("https://example.com") as response:
                    if response.status != 200:
                        raise Exception("Failed to fetch data")
                    return await response.text()

    try:
        data: str = await fetch_data()
        assert data  # Ensure data is fetched successfully
    except Exception as e:
        pytest.fail(f"Retry logic failed: {str(e)}")


@pytest.mark.integration
@pytest.mark.parametrize("max_workers", [2], ids=["2_workers"])
def test_sync_semaphore_retry_logic_effectiveness_with_nested_contexts(
    max_workers: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests the effectiveness of retry logic in synchronous tasks."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)

    @retry(
        Exception,
        max_retries=3,
        initial_delay=0.1,
        concurrency_limit=max_workers
    )
    def fetch_data() -> str:
        with semaphore.executor_context():
            with httpx.Client() as client:
                response = client.get("https://example.com")
                if response.status_code != 200:
                    raise Exception("Failed to fetch data")
                return response.text

    try:
        data: str = fetch_data()
        assert data  # Ensure data is fetched successfully
    except Exception as e:
        pytest.fail(f"Retry logic failed: {str(e)}")


###################
# CONCURRENCY AND STRESS TESTING

@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.parametrize(
    "max_workers, tasks",
    [(10, 1000)],
    ids=["10_workers_1000_tasks"]
)
@pytest.mark.asyncio
async def test_async_semaphore_performance_under_network_latency(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests semaphore under high concurrency and stress conditions."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)
    results: List[bool] = []

    async def simulated_network_call() -> None:
        async with semaphore.async_lock():
            # Simulate network call latency
            await asyncio.sleep(0.05)
            results.append(True)

    task_list: List[Coroutine[Any, Any, None]] = [
        asyncio.create_task(simulated_network_call()) for _ in range(tasks)
    ]
    await asyncio.gather(*task_list)
    assert len(results) == tasks, "Not all tasks completed successfully"


@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.parametrize(
    "max_workers, tasks",
    [(10, 1000)],
    ids=["10_workers_1000_tasks"]
)
def test_sync_semaphore_performance_under_network_latency(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests semaphore under high concurrency and stress conditions."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)
    results: List[bool] = []

    def simulated_network_call() -> None:
        with semaphore.executor_context():
            # Simulate network call latency
            time.sleep(0.05)
            results.append(True)

    # Submit tasks through the semaphore's executor to manage concurrency
    with semaphore.executor_context() as executor:
        futures: List[Future] = [
            executor.submit(simulated_network_call) for _ in range(tasks)
        ]
        for future in futures:
            future.result()  # Wait for the task to complete

    assert len(results) == tasks, "Not all tasks completed successfully"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.parametrize(
    "max_workers, tasks",
    [(10, 1000)],
    ids=["10_workers_1_thousand_tasks"]
)
@pytest.mark.asyncio
async def test_async_retry_logic_performance_under_stress(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests the performance of the async retry logic 
    under stress conditions."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)
    successful_calls: dict[str, int] = {"total": 0}
    max_workers: int = semaphore.async_max_workers

    @retry(Exception,
           max_retries=1,
           initial_delay=0.001,
           concurrency_limit=max_workers
    )
    async def unreliable_task() -> None:
        nonlocal successful_calls
        async with semaphore.async_lock():
            if successful_calls["total"] >= tasks/2:
                raise Exception("Simulated failure")
            successful_calls["total"] += 1

    task_list: List[Coroutine] = [
        unreliable_task() for _ in range(tasks)
    ]
    results: List[Union[Exception, None]] = await asyncio.gather(
        *task_list,
        return_exceptions=True
    )

    assert successful_calls["total"] == tasks/2,\
        "Expected 50 successful results after retries."
    for result in results:
        assert result is None or isinstance(result, Exception),\
            "Task did not complete as expected."


@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.integration
@pytest.mark.parametrize(
    "max_workers, tasks",
    [(10, 1000)],
    ids=["10_workers_1_thousand_tasks"]
)
def test_sync_retry_logic_performance_under_stress(
    max_workers: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests the performance of the sync retry logic
    under stress conditions."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)
    successful_calls: dict[str, int] = {"total": 0}
    max_workers: int = semaphore.sync_max_workers

    @retry(
        Exception,
        max_retries=1,
        initial_delay=0.001,
        concurrency_limit=max_workers
    )
    def unreliable_task():
        nonlocal successful_calls
        if successful_calls["total"] >= tasks/2:
            raise Exception("Simulated failure")

        successful_calls["total"] += 1

    # Submit tasks through the semaphore's executor to manage concurrency
    with semaphore.executor_context() as executor:
        futures: list[Future] = [
            executor.submit(unreliable_task) for _ in range(tasks)
        ]

    results: List[Union[Exception, None]] = []

    for future in futures:
        try:
            results.append(future.result())
        except Exception:
            pass

    # Directly assert the number of successful calls
    assert successful_calls["total"] == tasks/2, \
        f"Expected {tasks/2} successful results after retries."

    # Verify all tasks either completed successfully or raised an Exception
    for result in results:
        assert result is None or isinstance(result, Exception), \
            "Task did not complete as expected."


@pytest.mark.integration
@pytest.mark.parametrize(
    "max_workers, max_retries, tasks",
    [(2, 2, 4)],
    ids=["2_workers_2_retries_4_tasks"],
)
@pytest.mark.asyncio
async def test_async_retry_logic_retries_all_tasks(
    max_workers: int,
    max_retries: int,
    tasks: int,
) -> None:
    """Tests concurrency and threading with a semaphore and retries."""
    attempts: dict[str, int] = {"total": 0}

    @retry(
        Exception,
        max_retries=max_retries,
        concurrency_limit=max_workers
    )
    async def failing_task() -> None:
        nonlocal attempts
        attempts["total"] += 1
        raise Exception("Simulated task failure")

    task_list: List[Coroutine] = \
        [failing_task() for _ in range(tasks)]

    await asyncio.gather(
        *task_list,
        return_exceptions=True
    )

    expected_attempts: int = len(task_list) * (max_retries + 1)
    assert attempts["total"] == expected_attempts, \
        f"Expected {expected_attempts} attempts, got {attempts['total']}"


@pytest.mark.integration
@pytest.mark.parametrize(
    "max_workers, max_retries, tasks",
    [(2, 2, 4)],
    ids=["2_workers_2_retries_4_tasks"],
)
def test_sync_retry_logic_retries_all_tasks(
    max_workers: int,
    max_retries: int,
    tasks: int,
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    """Tests concurrency and threading with a semaphore and retries in a
    synchronous context."""
    semaphore: ConcurrencyManager = semaphore_factory(max_workers)
    attempts: dict[str, int] = {"total": 0}

    @retry(
        Exception,
        max_retries=max_retries,
        concurrency_limit=2
    )
    def failing_task() -> None:
        nonlocal attempts
        attempts["total"] += 1
        raise Exception("Simulated task failure")

    task_list: List[Callable[[], None]] = [failing_task for _ in range(tasks)]

    # Submit tasks through the semaphore's executor to manage concurrency
    with semaphore.executor_context() as executor:
        futures: list[Future] = [executor.submit(task) for task in task_list]
        for future in futures:
            with pytest.raises(Exception):
                future.result()

    # Directly assert the number of successful calls
    expected_attempts: int = len(task_list) * (max_retries + 1)
    assert attempts["total"] == expected_attempts, \
        f"Expected {expected_attempts} attempts, got {attempts['total']}"
