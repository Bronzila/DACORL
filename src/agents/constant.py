from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class ConstantAgent:
    def act(self, state: Tensor) -> Tensor:
        return state[1].item()

    def reset(self) -> None:
        pass
