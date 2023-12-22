import os
import pickle
from typing import Dict, List

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size: int) -> List[torch.tensor]:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self, state, action, next_state, reward, done):
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

        if self._size + 1 > self._buffer_size:
            print("Buffer full. Transitions will now start to be overwritten.")

    def save(self, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        except FileNotFoundError:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
                with open(filename, 'wb') as f:
                    pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)