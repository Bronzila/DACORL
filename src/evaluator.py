from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from DACBench.dacbench.envs import LayerwiseSGDEnv, SGDEnv, ToySGD2DEnv

from src.experiment_data import (
    ExperimentData,
    LayerwiseSGDExperimentData,
    SGDExperimentData,
    ToySGDExperimentData,
)
from src.utils import ActorType, get_environment, set_seeds

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class Evaluator:
    def __init__(
        self,
        data_dir: Path,
        eval_protocol: str,
        n_runs: int,
        seed: int,
    ) -> None:
        self._starting_points: np.ndarray | None

        with (data_dir / "run_info.json").open(mode="rb") as f:
            run_info = json.load(f)
        self._env = get_environment(run_info["environment"])

        if eval_protocol == "train":
            self._starting_points = run_info["starting_points"]
            eval_seed = run_info["seed"]
        elif eval_protocol == "interpolation":
            self._starting_points = None
            eval_seed = seed

        self._exp_data: ExperimentData
        if isinstance(self._env, ToySGD2DEnv):
            self._exp_data = ToySGDExperimentData()
            self._n_runs = (
                n_runs
                if n_runs is not None
                else len(run_info["starting_points"])
            )
            self._n_batches = run_info["environment"]["num_batches"]
        elif isinstance(self._env, SGDEnv):
            self._exp_data = SGDExperimentData()
            self._env.reset()
            self._n_runs = n_runs
            self._n_batches = run_info["environment"]["num_epochs"] * len(
                self._env.train_loader,
            )
        elif isinstance(self._env, LayerwiseSGDEnv):
            self._exp_data = LayerwiseSGDExperimentData()
            self._env.reset()
            self._n_runs = n_runs
            self._n_batches = run_info["environment"]["num_epochs"] * len(
                self._env.train_loader,
            )
        else:
            raise RuntimeError(
                f"Evaluation unsupported for environment: {type(self._env)}",
            )

        self._env.seed(eval_seed)
        set_seeds(eval_seed)

    def _run_batches(self, actor: ActorType, run_idx: int) -> None:
        """Evaluate specific run for a given number of batches.

        Args:
            actor (ActorType): Actor to be evaluated
            run_idx (int): Instance to be evaluated
        """
        print(f"Evaluating run {run_idx}")
        state = self._env.get_state()

        # [state] workaround due to layerwise/Liskov principle
        self._exp_data.init_data(run_idx, [state], self._env)

        for batch_idx in range(1, self._n_batches + 1):
            action = actor.act(state)
            next_state, reward, done, _, _ = self._env.step(action.item())
            state = next_state

            self._exp_data.add(
                {
                    "state": [state.cpu().numpy()],
                    "action": action,
                    "reward": reward.cpu().numpy(),
                    "batch_idx": batch_idx,
                    "run_idx": run_idx,
                    "env": self._env,
                },
            )

            if done:
                break

    def evaluate(
        self,
        actor: ActorType,
    ) -> pd.DataFrame:
        """Evaluates n starting points."""
        actor.eval()

        if self._starting_points is not None and len(self._starting_points) > 0:
            for run_id, starting_point in enumerate(
                self._starting_points[: self._n_runs],
            ):
                self._env.reset(
                    seed=None,
                    options={
                        "starting_point": torch.tensor(starting_point),
                    },
                )
                self._run_batches(actor, run_id)
        else:
            for run_id in range(self._n_runs):
                self._env.reset()
                self._run_batches(actor, run_id)

        actor.train()
        return self._exp_data.concatenate_data()


class LayerwiseEvaluator(Evaluator):
    def _run_batches(self, actor: ActorType, run_idx: int) -> None:
        """Evaluate specific run for a given number of batches.

        Args:
            actor (ActorType): Actor to be evaluated
            run_idx (int): Instance to be evaluated
        """
        print(f"Evaluating run {run_idx}")
        states = self._env.get_states()

        self._exp_data.init_data(run_idx, states, self._env)

        for batch_idx in range(1, self._n_batches + 1):
            actions = []
            for state in states:
                action = actor.act(state)
                actions.append(action.item())
            next_states, reward, done, _, _ = self._env.step(actions)

            for layer_idx, (state, action) in enumerate(zip(states, actions)):
                self._exp_data.add(
                    {
                        "state": state.cpu().numpy(),
                        "action": action,
                        "reward": reward.cpu().numpy(),
                        "layer_idx": layer_idx,
                        "batch_idx": batch_idx,
                        "run_idx": run_idx,
                        "env": self._env,
                    },
                )

            states = next_states

            if done:
                break
