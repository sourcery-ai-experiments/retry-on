from typing import AsyncGenerator
from src.retry_on.retry import TaskManager

###############
# TaskManager


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
