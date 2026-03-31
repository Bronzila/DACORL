# flake8: noqa: F401
import importlib
import warnings

from dacbench.envs.fast_downward import FastDownwardEnv

from dacbench.envs.geometric import GeometricEnv
from dacbench.envs.layerwise_nanoGPT import LayerwiseNanoGPTEnv
from dacbench.envs.layerwise_sgd import LayerwiseSGDEnv
from dacbench.envs.luby import LubyEnv, luby_gen
from dacbench.envs.sigmoid import (
    ContinuousSigmoidEnv,
    ContinuousStateSigmoidEnv,
    SigmoidEnv,
)
from dacbench.envs.theory import TheoryEnv
from dacbench.envs.toysgd import ToySGDEnv
from dacbench.envs.toysgd_2D import ToySGD2DEnv

__all__ = [
    "LubyEnv",
    "luby_gen",
    "SigmoidEnv",
    "ContinuousSigmoidEnv",
    "ContinuousStateSigmoidEnv",
    "FastDownwardEnv",
    "ToySGDEnv",
    "ToySGD2DEnv",
    "GeometricEnv",
    "TheoryEnv",
    "LayerwiseSGDEnv",
    "LayerwiseNanoGPTEnv",
]

modcma_spec = importlib.util.find_spec("modcma")
found = modcma_spec is not None
if found:
    from dacbench.envs.cma_es import CMAESEnv

    __all__.append("CMAESEnv")
else:
    warnings.warn(  # noqa: B028
        "CMA-ES Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )

sgd_spec = importlib.util.find_spec("backpack")
found = sgd_spec is not None
if found:
    from dacbench.envs.sgd import SGDEnv

    __all__.append("SGDEnv")
else:
    warnings.warn(  # noqa: B028
        "SGD Benchmark not installed. If you want to use this benchmark, "
        "please follow the installation guide."
    )
    
