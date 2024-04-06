from typing import TypeVar
import sys

R = TypeVar("R")
E = TypeVar("E", bound=Exception)

sys.modules[__name__].R = R
sys.modules[__name__].E = E
