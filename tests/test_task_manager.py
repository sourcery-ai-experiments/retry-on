import asyncio
import pytest
from typing import AsyncGenerator, Callable
from src.retry_on.tasks import TaskManager, ConcurrencyManager
from tests.utils import semaphore_factory


def test_is_async() -> None:
    async def async_func() -> None:
        pass

    def sync_func() -> None:
        pass

    assert (
        TaskManager.is_async(async_func) is True
    ), "Async function not correctly identified"
    assert (
        TaskManager.is_async(sync_func) is False
    ), "Sync function not correctly identified"


def test_is_async_variants() -> None:
    async def async_func() -> None:
        pass

    async def async_gen() -> AsyncGenerator:
        yield

    assert (
        TaskManager.is_async(async_func) is True
    ), "Coroutine functions should be recognized as async"
    assert (
        TaskManager.is_async(async_gen) is True
    ), "Async generator functions should be recognized as async"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_manager_async_task_submission(
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    semaphore: ConcurrencyManager = semaphore_factory(2)
    task_manager: TaskManager = TaskManager(semaphore=semaphore)

    async def async_task() -> bool:
        return True

    task = await task_manager.submit_async_task(async_task)

    assert await task_manager.get_async_task_result(task) is True,\
        "Async task submission failed"


@pytest.mark.unit
def test_task_manager_sync_task_submission(
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    semaphore: ConcurrencyManager = semaphore_factory(2)
    task_manager: TaskManager = TaskManager(semaphore=semaphore)

    def sync_task() -> str:
        return "task completed"

    task = task_manager.submit_sync_task(sync_task)
    assert task is not None, "Sync task submission failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_manager_cancel_task(
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    semaphore: ConcurrencyManager = semaphore_factory(1)
    task_manager: TaskManager = TaskManager(semaphore=semaphore)

    async def cancellable_task():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise

    task = await task_manager.submit_async_task(cancellable_task)
    await task_manager.cancel_task(task)

    assert task.cancelled(), "Task was not correctly cancelled"

@pytest.mark.unit
@pytest.mark.asyncio
async def test_task_manager_signal_shutdown(
    semaphore_factory: Callable[[int], ConcurrencyManager]
) -> None:
    semaphore: ConcurrencyManager = semaphore_factory(2)
    task_manager: TaskManager = TaskManager(semaphore=semaphore)

    task_manager.signal_shutdown()
    assert task_manager.shutdown_required, \
        "Shutdown signal was not correctly set"
