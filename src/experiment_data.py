from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dacbench import AbstractEnv
    from torch import Tensor


class ExperimentData(metaclass=ABCMeta):
    data: dict[str, Any]
    aggregated_data: list[pd.DataFrame]
    df_aggregated_data: pd.DataFrame

    @abstractmethod
    def init_data(self, run_idx: int, state: Tensor, env: AbstractEnv) -> None:
        """Initialize the data per run, dependant of the experiment."""

    @abstractmethod
    def add(self, logs: dict) -> None:
        """Add a new data point based on experiment-specific fields."""

    def _add(self, logs: dict) -> None:
        """Add a new data point for common fields."""
        self.data["reward"].append(logs["reward"])
        self.data["state"].append(logs["state"])
        self.data["batch_idx"].append(logs["batch_idx"])
        self.data["run_idx"].append(logs["run_idx"])

    def _reset(self) -> None:
        """Reset the fields for the next run."""
        self.data.clear()

    def save(self) -> None:
        """Convert the collected data to a DataFrame and reset."""
        self.aggregated_data.append(pd.DataFrame(self.data))
        self._reset()

    def concatenate_data(self) -> pd.DataFrame:
        """Concatenate all run DataFrames together."""
        if self.df_aggregated_data is None:
            self.df_aggregated_data = pd.concat(
                self.aggregated_data,
                ignore_index=True,
            )
        return self.df_aggregated_data


class ToySGDExperimentData(ExperimentData):
    def init_data(self, run_idx: int, state: Tensor, env: AbstractEnv) -> None:
        self.data = {
            "reward": np.nan,
            "action": math.log10(env.learning_rate),
            "state": state.numpy(),
            "batch": 0,
            "run": run_idx,
            "f_cur": env.objective_function(env.x_cur).numpy(),
            "x_cur": env.x_cur.tolist(),
        }

    def add(self, logs: dict) -> None:
        super()._add(logs)
        self.data["x_cur"].append(logs["env"].x_cur)
        self.data["f_cur"].append(logs["env"].f_cur)


class SGDExperimentData(ExperimentData):
    def init_data(self, run_idx: int, state: Tensor, env: AbstractEnv) -> None:
        self.data = {
            "reward": np.nan,
            "action": math.log10(env.learning_rate),
            "state": state.numpy(),
            "batch": 0,
            "run": run_idx,
            "train_loss": env.train_loss,
            "valid_loss": env.validation_loss,
            "train_acc": env.train_accuracy,
            "valid_acc": env.validation_accuracy,
            "test_loss": env.test_loss,
            "test_acc": env.test_accuracy,
        }

    def add(self, logs: dict) -> None:
        super()._add(logs)
        self.data["train_loss"].append(logs["env"].train_loss)
        self.data["valid_loss"].append(logs["env"].valid_loss)
        self.data["test_loss"].append(logs["env"].test_loss)
        self.data["train_acc"].append(logs["env"].train_acc)
        self.data["valid_acc"].append(logs["env"].valid_acc)
        self.data["test_acc"].append(logs["env"].test_acc)


class CMAESExperimentData(ExperimentData):
    def init_data(self, run_idx: int, state: Tensor, env: AbstractEnv) -> None:
        self.data = {
            "reward": np.nan,
            "action": env.es.parameters.sigma,
            "state": state.numpy(),
            "batch": 0,
            "run": run_idx,
            "lambda": env.es.parameters.lambda_,
            "f_cur": env.es.parameters.fopt,
            "population": env.es.parameters.population.f,
            "target_value": env.target,
            "fid": env.fid,
        }

    def add(self, logs: dict) -> None:
        super()._add(logs)
        self.data["lambda"].append(logs["env"].es.parameters.lambda_)
        self.data["f_cur"].append(logs["env"].es.parameters.fopt)
        self.data["population"].append(logs["env"].es.parameters.population.f)
        self.data["target_value"].append(logs["env"].target)
        self.data["fid"].append(logs["env"].fid)
