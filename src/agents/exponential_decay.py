from __future__ import annotations

import math


class ExponentialDecayAgent:
    def __init__(
        self,
        decay_rate: float = 0.96,
        decay_steps: int = 100,
    ) -> None:
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps

    def act(self, state):
        # Since log learning rate is given in state, transform first
        prev_learning_rate = 10 ** state[1]
        learning_rate = prev_learning_rate * self._decay_rate ** (1 / self._decay_steps)

        # return log of learning rate
        return math.log10(learning_rate)

    def reset(self):
        pass
