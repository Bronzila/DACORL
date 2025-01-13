from __future__ import annotations

from typing import TYPE_CHECKING

from DACBench.dacbench.envs import (
    LayerwiseNanoGPTEnv,
    LayerwiseSGDEnv,
    SGDEnv,
    ToySGD2DEnv,
)

from src.experiment_data import (
    ExperimentData,
    LayerwiseSGDExperimentData,
    SGDExperimentData,
    ToySGDExperimentData,
)
from src.utils import ActorType, get_environment, set_seeds

if TYPE_CHECKING:
    import pandas as pd


class Evaluator:
    def __init__(
        self,
        env_config: dict,
    ) -> None:
        self._env = get_environment(env_config)

        self._exp_data: ExperimentData
        if isinstance(self._env, ToySGD2DEnv):
            self._exp_data = ToySGDExperimentData()
            self._num_runs = env_config["num_runs"]
            self._num_batches = env_config["num_batches"]
        elif isinstance(self._env, SGDEnv):
            self._exp_data = SGDExperimentData()
            self._env.reset()
            self._num_runs = env_config["num_runs"]
            self._num_batches = env_config["num_epochs"] * len(
                self._env.train_loader,
            )
        elif isinstance(self._env, LayerwiseSGDEnv | LayerwiseNanoGPTEnv):
            self._exp_data = LayerwiseSGDExperimentData()
            self._env.reset()
            self._num_runs = env_config["num_runs"]
            self._num_batches = env_config["num_epochs"] * len(
                self._env.train_loader,
            )
        else:
            raise RuntimeError(
                f"Evaluation unsupported for environment: {type(self._env)}",
            )

        self._env.seed(env_config["seed"])
        set_seeds(env_config["seed"])

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

        for batch_idx in range(1, self._num_batches + 1):
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

        for run_id in range(self._num_runs):
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

        for batch_idx in range(1, self._num_batches + 1):
            actions = []
            for state in states:
                action = actor.act(state)
                actions.append(action.item())
            next_states, reward, done, _, _ = self._env.step(actions)

            for layer_idx, (state, action) in enumerate(
                zip(states, actions, strict=True),
            ):
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
