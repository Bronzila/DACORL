from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class Teacher(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(self, state: Tensor) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
