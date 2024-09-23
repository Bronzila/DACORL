from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from DACBench.dacbench.benchmarks import CMAESBenchmark


class CSA:
    def __init__(self) -> None:
        pass

    def act(self, env: CMAESBenchmark) -> None:
        sigma = env.es.parameters.sigma
        sigma *= np.exp(
            (env.es.parameters.cs / env.es.parameters.damps)
            * (
                (np.linalg.norm(env.es.parameters.ps) / env.es.parameters.chiN)
                - 1
            ),
        )
        return sigma + 1e-10

    def reset(self) -> None:
        pass
