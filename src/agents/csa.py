from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from DACBench.dacbench.envs.cma_es import CMAESEnv


class CSA:
    def __init__(self) -> None:
        pass

    def act(self, env: CMAESEnv) -> float:
        params = env.es.parameters  # type: ignore
        sigma: float = params.sigma
        sigma *= np.exp(
            (params.cs / params.damps)
            * ((np.linalg.norm(params.ps) / params.chiN) - 1),
        )
        return sigma + 1e-10

    def reset(self) -> None:
        pass
