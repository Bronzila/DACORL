from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import MISSING, DictConfig


@dataclass
class Config(DictConfig):
    # Required fields
    results_dir: Path = MISSING  # Mandatory, no default value

    # Optional fields with default values
    env: dict = MISSING
    num_train_iter: int = 30000
    seed: int = 0
    eval_seed: int = 0
    eval_protocol: str = "train"
    teacher: str = "step_decay"
    instance_mode: str | None = None  # None if not provided
    id: int = 0
    agent_type: Literal[
        "bc",
        "td3_bc",
        "cql",
        "awac",
        "edac",
        "sac_n",
        "lb_sac",
        "iql",
        "td3",
    ] = "td3_bc"
    tanh_scaling: bool = False
    combination: Literal["homogeneous", "heterogeneous", "single"] = "single"
    mode: Literal["data_generation", "training", "both"] = "both"
    data_exists: bool = False
