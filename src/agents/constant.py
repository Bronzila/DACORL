from __future__ import annotations

import math


class ConstantAgent:
    def __init__(
        self,
        learning_rate: float,
    ) -> None:
        self.learning_rate = learning_rate

    def act(self, state):
        return math.log10(self.learning_rate)

    def reset(self):
        pass
