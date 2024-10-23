from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from dacbench import AbstractEnv
    from torch import Tensor


class ExperimentData(metaclass=ABCMeta):
    data: dict[str, Any]

    @abstractmethod
    def init_data(
        self,
        run_idx: int,
        state: list[Tensor],
        env: AbstractEnv,
    ) -> None:
        """Initialize the data dictionary, dependent on the experiment."""

    @abstractmethod
    def add(self, logs: dict) -> None:
        """Add a new data point based on experiment-specific fields."""

    def _add(self, logs: dict) -> None:
        """Add a new data point for common fields."""
        self.data["reward"].append(logs["reward"].item())
        self.data["state"].append(logs["state"])
        self.data["batch_idx"].append(logs["batch_idx"])
        self.data["run_idx"].append(logs["run_idx"])

    def concatenate_data(self) -> pd.DataFrame:
        """Return concatenated run data."""
        return pd.DataFrame(self.data)


class ToySGDExperimentData(ExperimentData):
    def init_data(
        self,
        run_idx: int,
        state: list[Tensor],
        env: AbstractEnv,
    ) -> None:
        if not hasattr(self, "data"):
            self.data = {
                "reward": [],
                "action": [],
                "state": [],
                "batch_idx": [],
                "run_idx": [],
                "f_cur": [],
                "x_cur": [],
            }

        initial_log = {
            "reward": torch.tensor(float("nan")),
            "state": state[0].numpy(),
            "batch_idx": 0,
            "run_idx": run_idx,
            "env": env,
        }
        self.add(initial_log)

    def add(self, logs: dict) -> None:
        super()._add(logs)
        self.data["action"].append(math.log10(logs["env"].learning_rate))
        self.data["x_cur"].append(logs["env"].x_cur.tolist())
        self.data["f_cur"].append(logs["env"].f_cur.tolist())


class SGDExperimentData(ExperimentData):
    def init_data(
        self,
        run_idx: int,
        state: list[Tensor],
        env: AbstractEnv,
    ) -> None:
        if not hasattr(self, "data"):
            self.data = {
                "reward": [],
                "action": [],
                "state": [],
                "batch_idx": [],
                "run_idx": [],
                "train_loss": [],
                "validation_loss": [],
                "train_accuracy": [],
                "validation_accuracy": [],
                "test_loss": [],
                "test_accuracy": [],
            }

        initial_log = {
            "reward": torch.tensor(float("nan")),
            "state": state[0].numpy(),
            "batch_idx": 0,
            "run_idx": run_idx,
            "env": env,
        }
        self.add(initial_log)

    def add(self, logs: dict) -> None:
        super()._add(logs)
        self.data["action"].append(math.log10(logs["env"].learning_rate))
        self.data["train_loss"].append(logs["env"].train_loss)
        self.data["validation_loss"].append(logs["env"].validation_loss)
        self.data["test_loss"].append(logs["env"].test_loss)
        self.data["train_accuracy"].append(logs["env"].train_accuracy)
        self.data["validation_accuracy"].append(logs["env"].validation_accuracy)
        self.data["test_accuracy"].append(logs["env"].test_accuracy)


class LayerwiseSGDExperimentData(ExperimentData):
    def init_data(
        self,
        run_idx: int,
        states: list[Tensor],
        env: AbstractEnv,
    ) -> None:
        if not hasattr(self, "data"):
            self.data = {
                "reward": [],
                "action": [],
                "state": [],
                "batch_idx": [],
                "run_idx": [],
                "layer_idx": [],
                "train_loss": [],
                "validation_loss": [],
                "train_accuracy": [],
                "validation_accuracy": [],
                "test_loss": [],
                "test_accuracy": [],
            }

        for i, state in enumerate(states):
            initial_log = {
                "reward": torch.tensor(float("nan")),
                "state": state.numpy(),
                "batch_idx": 0,
                "run_idx": run_idx,
                "layer_idx": i,
                "env": env,
            }
            self.add(initial_log)

    def add(self, logs: dict) -> None:
        super()._add(logs)
        layer_idx = logs["layer_idx"]
        self.data["layer_idx"] = layer_idx
        self.data["action"].append(
            math.log10(logs["env"].learning_rates[layer_idx]),
        )
        self.data["train_loss"].append(logs["env"].train_loss)
        self.data["validation_loss"].append(logs["env"].validation_loss)
        self.data["test_loss"].append(logs["env"].test_loss)
        self.data["train_accuracy"].append(logs["env"].train_accuracy)
        self.data["validation_accuracy"].append(logs["env"].validation_accuracy)
        self.data["test_accuracy"].append(logs["env"].test_accuracy)


class CMAESExperimentData(ExperimentData):
    def init_data(
        self,
        run_idx: int,
        state: list[Tensor],
        env: AbstractEnv,
    ) -> None:
        initial_log = {
            "reward": [np.nan],
            "action": [env.es.parameters.sigma],
            "state": [state[0].numpy()],
            "batch_idx": [0],
            "run_idx": [run_idx],
            "lambda": [env.es.parameters.lambda_],
            "f_cur": [env.es.parameters.fopt],
            "population": [env.es.parameters.population.f],
            "target_value": [env.target],
            "fid": [env.fid],
        }

        if not hasattr(self, "data"):
            self.data = initial_log
        else:
            self.add(initial_log)

    def add(self, logs: dict) -> None:
        super()._add(logs)
        self.data["action"].append(logs["env"].sigma)
        self.data["lambda"].append(logs["env"].es.parameters.lambda_)
        self.data["f_cur"].append(logs["env"].es.parameters.fopt)
        self.data["population"].append(logs["env"].es.parameters.population.f)
        self.data["target_value"].append(logs["env"].target)
        self.data["fid"].append(logs["env"].fid)
