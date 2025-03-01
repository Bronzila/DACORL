from __future__ import annotations

import torch

from src.agents.teacher import Teacher


class StepDecay(Teacher):
    def __init__(self, step_size: int = 20, gamma: float = 0.2) -> None:
        self._step_size = step_size
        self._gamma = gamma
        self._step = 0

    def act(self, state: torch.Tensor) -> float:
        self._step += 1
        # Since log learning rate is given in state, transform first
        learning_rate = 10 ** state[1]

        if self._step % self._step_size == 0:
            learning_rate = learning_rate * self._gamma

        # return log of learning rate
        return torch.log10(learning_rate).item()

    def reset(self) -> None:
        self._step = 0
