import numpy as np

class StepDecayAgent():
    def __init__(self, step_size: int=20, gamma: float=0.2) -> None:
        self._step_size = step_size
        self._gamma = gamma
        self._step = 0

    def act(self, state):
        self._step += 1
        learning_rate = state[1]

        if self._step % self._step_size == 0:
            learning_rate = learning_rate * self._gamma

        # return log of learning rate
        return np.log10(learning_rate)

    def reset(self):
        self._step = 0
