from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from src.experiment_data import (
    CMAESExperimentData,
    ExperimentData,
    SGDExperimentData,
    ToySGDExperimentData,
)
from src.utils.general import (
    OutOfTimeError,
    get_environment,
    get_teacher,
    set_seeds,
)
from src.utils.generate_data import load_checkpoint
from src.utils.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    import numpy as np


class Generator:
    def __init__(
        self,
        teacher_config: dict,
        env_config: dict,
        result_dir: Path,
        check_if_exists: bool,
        num_runs: int,
        checkpoint: int,
        seed: int,
        timeout: int,
        verbose: bool,
    ) -> None:
        self.env_config = env_config

        self.environment_type = env_config["type"]
        self.agent_type = teacher_config["type"]

        self.seed = seed
        self.timeout = timeout
        self.verbose = verbose

        self._init_seeds()
        self._init_env()
        self._init_teacher(teacher_config)

        state = self.env.reset()[0]
        self.env.seed(
            self.seed,
        )  # Reseed environment here to allow for proper starting point generation
        state_dim = state.shape[0]

        if check_if_exists and (result_dir.exists() and checkpoint == 0):
            print(f"Data already exists: {result_dir}")
            return

        self.exp_data: ExperimentData
        if self.environment_type == "ToySGD":
            self.exp_data = ToySGDExperimentData()
        elif self.environment_type == "SGD":
            self.exp_data = SGDExperimentData()
        elif self.environment_type == "CMAES":
            self.exp_data = CMAESExperimentData()
        else:
            raise NotImplementedError(
                f"No experiment data class for experiment {self.environment_type}",
            )

        self.result_dir: Path = (
            result_dir
            / self.environment_type
            / self.agent_type
            / str(teacher_config["id"])
        )

        if self.environment_type == "ToySGD":
            self.result_dir = self.result_dir / env_config["function"]

            # Start with instance 0
            self.env.instance_index = -1

        buffer_size = num_runs * self._num_batches
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=1,
            buffer_size=buffer_size,
            seed=seed,
        )

        self.run_info = {
            "agent": teacher_config,
            "environment": env_config,
            "seed": seed,
            "num_runs": num_runs,
            "num_batches": self._num_batches,
        }

        self.start_run = 0
        self.starting_points: list[np.ndarray] = []

        if checkpoint != 0:
            self._handle_checkpoint(result_dir, checkpoint)

    def generate_data(self, checkpointing_freq: int = 0) -> None:
        """Generate data and checkpoints if required."""
        num_runs: int = self.run_info["num_runs"]  # type: ignore
        num_batches: int = self.run_info["num_batches"]  # type: ignore

        save_checkpoints = checkpointing_freq != 0

        try:
            for run in range(self.start_run, num_runs):
                state, meta_info = self.env.reset()
                if self.environment_type == ("ToySGD"):
                    self.starting_points.append(meta_info["start"])
                self.agent.reset()

                self.exp_data.init_data(run, state, self.env)

                start = time()
                for batch in range(1, num_batches + 1):
                    if self.verbose:
                        print(
                            f"Starting {self._phase} {batch}/{num_batches} of run {run}. \
                            Total {batch + run * num_batches}/{num_runs * num_batches}",
                        )

                    if self.agent_type in ("csa", "cmaes_constant"):
                        action = self.agent.act(self.env)
                    else:
                        action = self.agent.act(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                    self.replay_buffer.add_transition(
                        state,
                        action,
                        next_state,
                        reward,
                        done,
                    )
                    self.exp_data.add(
                        {
                            "state": state.numpy(),
                            "action": action,
                            "reward": reward.numpy(),
                            "batch_idx": batch,
                            "run_idx": run,
                            "env": self.env,
                        },
                    )

                    state = next_state
                    if done:
                        break

                end = time()
                print(f"Run {run} took {end - start} sec.")

                if save_checkpoints and (run + 1) % checkpointing_freq == 0:
                    checkpoint_dir = self.result_dir / "checkpoints" / str(run)
                    if not checkpoint_dir.exists():
                        checkpoint_dir.mkdir(parents=True)

                    if self.environment_type in ["ToySGD", "CMAES"]:
                        raise UserWarning(
                            f"Are you sure you want to checkpoint {self.environment_type}?",
                        )
                    if self.environment_type == "SGD":
                        self.run_info.update(
                            {
                                "checkpoint_info": {
                                    "run": run,
                                    "rng": self.env.rng.bit_generator.state,
                                    "instance_index": self.env.instance_index,
                                },
                            },
                        )
                    self.save_data(save_checkpoints)
        except OutOfTimeError:
            print("Run was about to end. Let's quickly save the progress.")

        print(f"Saved data and ReplayBuf to {self.result_dir}")

    def save_data(
        self,
        save_checkpoint: bool = False,
    ) -> None:
        """Save aggregated run data and current ReplayBuffer."""
        aggregated_run_data = self.exp_data.concatenate_data()

        if not self.result_dir.exists():
            self.result_dir.mkdir(parents=True)

        self.run_info["starting_points"] = self.starting_points
        save_path = self.result_dir / "rep_buffer"
        if save_checkpoint:
            self.replay_buffer.checkpoint(save_path)
        else:
            self.replay_buffer.save(save_path)

        with (self.result_dir / "run_info.json").open(mode="w") as f:
            json.dump(self.run_info, f, indent=4)

        aggregated_run_data.to_csv(
            self.result_dir / "aggregated_run_data.csv",
        )

        print(
            f"Saved {'checkpoint' if save_checkpoint else 'data'} in {self.result_dir}",
        )

    def _init_seeds(self) -> None:
        """Sets global seeds."""
        set_seeds(self.seed)
        self.env_config["seed"] = self.seed

    def _init_env(self) -> None:
        """Builds the environment based on the env_config field."""
        print(f"Environment: {self.env_config}")

        env = get_environment(self.env_config.copy())
        env.reset()

        self._num_batches: int
        self._phase = "batch"
        batches_per_epoch = 1
        if self.environment_type == "SGD":
            print(f"Generating data for {self.env_config['dataset_name']}")
            if env.epoch_mode is False:
                num_epochs = self.env_config["num_epochs"]
                # if SGD env, translates num_batches to num_epochs
                batches_per_epoch = len(env.train_loader)
                print(f"One epoch consists of {batches_per_epoch} batches.")
                self._num_batches = num_epochs * batches_per_epoch
                self.env_config["cutoff"] = self._num_batches
            else:
                self._phase = "epoch"
                print("Currently running in epoch mode.")
        else:
            self._num_batches = self.env_config["num_batches"]
        self.env = get_environment(self.env_config.copy())

    def _init_teacher(self, teacher_config: dict) -> None:
        """Builds the teacher agent based on the teacher_config field."""
        print(f"Teacher: {teacher_config}")
        self.agent = get_teacher(teacher_config)

    def _handle_checkpoint(self, result_dir: Path, checkpoint: int) -> None:
        """Loads the given checkpoint data and ."""
        checkpoint_dir = result_dir / "checkpoints" / str(checkpoint)
        (
            checkpoint_data,
            self.replay_buffer,
            checkpoint_run_info,
        ) = load_checkpoint(
            checkpoint_dir,
        )

        self.starting_points = checkpoint_run_info["starting_points"]
        self.start_run = checkpoint_run_info["checkpoint_info"]["run"] + 1
        if self.environment_type == "SGD":
            # assuming we use PCG64 as rng
            self.env.rng.bit_generator.state = checkpoint_run_info[
                "checkpoint_info"
            ]["rng"]
            self.env.instance_index = checkpoint_run_info["checkpoint_info"][
                "instance_index"
            ]

        del checkpoint_run_info["starting_points"]
        del checkpoint_run_info["checkpoint_info"]

        assert self.run_info == checkpoint_run_info
