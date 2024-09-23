from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class ConstantAgent:
    def __init__(
        self,
        learning_rate: float,
    ) -> None:
        self.learning_rate = learning_rate

    def act(self, _: Tensor) -> float:
        return math.log10(self.learning_rate)

    def reset(self) -> None:
        pass
