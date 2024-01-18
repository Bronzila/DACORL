from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class Agent(ABC):
    def __init__(self, device: str = "cpu") -> None:
        self.total_it = 0
        self.device = device

    @abstractmethod
    def train(self, batch: list[torch.Tensor]) -> dict[str, Any]:
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
