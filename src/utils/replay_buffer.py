from __future__ import annotations

import pickle
from pathlib import Path
from typing import cast

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim),
            dtype=torch.float32,
            device=device,
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim),
            dtype=torch.float32,
            device=device,
        )
        self._rewards = torch.zeros(
            (buffer_size, 1),
            dtype=torch.float32,
            device=device,
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim),
            dtype=torch.float32,
            device=device,
        )
        self._dones = torch.zeros(
            (buffer_size, 1),
            dtype=torch.float32,
            device=device,
        )
        self._device = device
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size: int) -> list[torch.Tensor]:
        indices = self.rng.integers(
            0,
            self._size,
            size=batch_size,
        )
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: torch.FloatTensor,
        action: torch.FloatTensor,
        next_state: torch.FloatTensor,
        reward: torch.FloatTensor,
        done: torch.FloatTensor,
    ) -> None:
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

        if self._size + 1 > self._buffer_size:
            print("Buffer full. Transitions will now start to be overwritten.")

    def save(self, filename: Path) -> None:
        # Only save actual collected data, not zeros
        self._states = self._states[: self._size]
        self._actions = self._actions[: (self._size)]
        self._next_states = self._next_states[: (self._size)]
        self._rewards = self._rewards[: (self._size)]
        self._dones = self._dones[: (self._size)]
        self._buffer_size = self._size
        self._pointer = self._pointer % self._buffer_size

        # Check if any state is entirely zero
        states_zero_mask = (self._states == 0).all(dim=1)
        num_zero_states = states_zero_mask.sum().item()
        assert num_zero_states == 0

        try:
            with filename.open(mode="wb") as f:
                pickle.dump(self, f)
        except FileNotFoundError:
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)
                with filename.open(mode="wb") as f:
                    pickle.dump(self, f)

    def checkpoint(self, filename: Path) -> None:
        """Saves the ReplayBuffer without cutting its size."""
        try:
            with filename.open(mode="wb") as f:
                pickle.dump(self, f)
        except FileNotFoundError:
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)
                with filename.open(mode="wb") as f:
                    pickle.dump(self, f)

    def merge(self, other: ReplayBuffer) -> None:
        self._buffer_size += other._buffer_size
        self._size = self._size + other._size
        self._pointer = self._pointer + other._pointer

        self._states = torch.cat([self._states, other._states])
        self._actions = torch.cat([self._actions, other._actions])
        self._next_states = torch.cat([self._next_states, other._next_states])
        self._rewards = torch.cat([self._rewards, other._rewards])
        self._dones = torch.cat([self._dones, other._dones])

    @classmethod
    def load(cls, filename: Path) -> ReplayBuffer:
        with filename.open(mode="rb") as f:
            return cast(ReplayBuffer, pickle.load(f))
