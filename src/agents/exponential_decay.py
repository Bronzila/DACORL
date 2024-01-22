from __future__ import annotations

import math


class ExponentialDecayAgent:
    def __init__(
        self,
        initial_learning_rate: float,
        decay_rate: float = 0.96,
        decay_steps: int = 100,
    ) -> None:
        self._initial_learning_rate = initial_learning_rate
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self._step = 0

    def act(self, state):
        self._step += 1

        learning_rate = self._initial_learning_rate * self._decay_rate ** (
            self._step / self._decay_steps
        )

        # return log of learning rate
        return math.log10(learning_rate)

    def reset(self):
        self._step = 0
