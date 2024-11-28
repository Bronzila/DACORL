from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.agents.teacher import Teacher

if TYPE_CHECKING:
    from torch import Tensor


class Constant(Teacher):
    def __init__(
        self,
        initial_learning_rate: float,
    ) -> None:
        self.learning_rate = initial_learning_rate

    def act(self, _: Tensor) -> float:
        return math.log10(self.learning_rate)

    def reset(self) -> None:
        pass
