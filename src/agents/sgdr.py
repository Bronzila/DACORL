from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class SGDRAgent:
    def __init__(
        self,
        initial_learning_rate: float,
        t_i: float = 2.0,
        t_mult: float = 2,
        batches_per_epoch: int = 10,
    ) -> None:
        self._t_i = t_i
        self._t_mult = t_mult
        self._batches_since_last_reset = 0

        # SGDR constants
        self._min_lr = 1e-10
        self._batches_per_epoch = batches_per_epoch
        self._initial_lr = initial_learning_rate

        # Required for reset
        self._initial_t_i = t_i

    def act(self, _: Tensor) -> float:
        # First normalize current batch to batch per epoch
        t_cur = self._batches_since_last_reset / self._batches_per_epoch
        # Note: our initial lr is the max lr in our case
        learning_rate = self._min_lr + 0.5 * (
            self._initial_lr - self._min_lr
        ) * (1 + math.cos(t_cur / self._t_i * math.pi))

        # Increment step counter
        self._batches_since_last_reset += 1

        # Restart if T_cur reaches T_i, here we can simply reset batches since
        # last reset and then start counting again
        if t_cur == self._t_i:
            self._batches_since_last_reset = 0
            self._t_i = self._t_i * self._t_mult

        # return log of learning rate
        return math.log10(learning_rate)

    def reset(self) -> None:
        self._batches_since_last_reset = 0
        self._t_i = self._initial_t_i
