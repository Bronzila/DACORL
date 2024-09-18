from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from DACBench.dacbench.abstract_env import AbstractMADACEnv


class ConstantCMAES:
    def __init__(self) -> None:
        pass

    def act(self, env: AbstractMADACEnv) -> None:
        if env.es.parameters.used_budget == env.es.parameters.lambda_:
            self._sigma = env.es.parameters.sigma + 1e-10
        return self._sigma

    def reset(self) -> None:
        pass