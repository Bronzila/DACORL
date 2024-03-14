from __future__ import annotations

import math


class SGDRAgent:
    def __init__(
        self,
        initial_learning_rate: float,
        T_i: float = 2.0,
        T_mult: float = 2,
        batches_per_epoch: int = 10,
    ) -> None:
        self._T_i = T_i
        self._T_mult = T_mult
        self._batches_since_last_reset = 0

        # SGDR constants
        self._min_lr = 1e-10
        self._batches_per_epoch = batches_per_epoch
        self._initial_lr = initial_learning_rate

        # Required for reset
        self._initial_T_i = T_i

    def act(self, state):
        # First normalize current batch to batch per epoch
        T_cur = self._batches_since_last_reset / self._batches_per_epoch
        # Note: our initial lr is the max lr in our case
        learning_rate = self._min_lr + 0.5 *(self._initial_lr - self._min_lr) *\
                        (1 + math.cos(T_cur / self._T_i * math.pi))

        # Increment step counter
        self._batches_since_last_reset += 1

        # Restart if T_cur reaches T_i, here we can simply reset batches since
        # last reset and then start counting again
        if T_cur == self._T_i:
            self._batches_since_last_reset = 0
            self._T_i = self._T_i * self._T_mult

        # return log of learning rate
        return math.log10(learning_rate)

    def reset(self):
        self._batches_since_last_reset = 0
        self._T_i = self._initial_T_i
