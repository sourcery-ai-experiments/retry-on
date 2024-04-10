from typing import (
    Union,
    Tuple,
    Type,
)
from src.retry_on.types import E


def validate_exceptions(exc: Union[Type[E], Tuple[Type[E], ...]]) -> None:
    excs = (exc,) if isinstance(exc, type) else exc
    if errors := [
        f"{exception} must subclass Exception"
        for exception in excs
        if not issubclass(exception, Exception)
    ]:
        error_message = "\n".join(errors)
        raise ValueError(error_message)
