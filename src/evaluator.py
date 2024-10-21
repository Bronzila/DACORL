from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from dacbench.envs import CMAESEnv, SGDEnv, ToySGD2DEnv

from src.experiment_data import (
    CMAESExperimentData,
    ExperimentData,
    SGDExperimentData,
    ToySGDExperimentData,
)
from src.utils.general import ActorType, EnvType, set_seeds

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class Evaluator:
    def __init__(
        self,
        env: EnvType,
        n_runs: int,
        n_batches: int,
        seed: int,
        starting_points: np.ndarray | None = None,
    ) -> None:
        self._env = env
        self._n_runs = n_runs
        self._n_batches = n_batches
        self._starting_points = starting_points

        self._env.seed(seed)
        set_seeds(seed)

        self._exp_data: ExperimentData
        if isinstance(env, ToySGD2DEnv):
            self._exp_data = ToySGDExperimentData()
        elif isinstance(env, SGDEnv):
            self._exp_data = SGDExperimentData()
        elif isinstance(env, CMAESEnv):
            self._exp_data = CMAESExperimentData()
        else:
            raise RuntimeError(f"Unknown enviroment instance {type(env)}")

    def _run_batches(self, actor: ActorType, run_idx: int) -> None:
        """Evaluate specific run for a given number of batches.

        Args:
            actor (ActorType): Actor to be evaluated
            run_idx (int): Instance to be evaluated
        """
        print(f"Evaluating run {run_idx}")
        state, _ = self._env.reset()

        self._exp_data.init_data(run_idx, state, self._env)

        for batch_idx in range(1, self._n_batches + 1):
            action = actor.act(state)
            next_state, reward, done, _, _ = self._env.step(action.item())
            state = next_state

            self._exp_data.add(
                {
                    "state": state.numpy(),
                    "action": action,
                    "reward": reward.numpy(),
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

        if self._starting_points is not None:
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
