from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.general import get_environment, get_teacher, set_seeds
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
    ) -> None:
        self.env_config = env_config

        self.environment_type = env_config["type"]
        self.agent_type = teacher_config["type"]

        self.seed = seed
        self.timeout = timeout

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

        result_dir = (
            result_dir
            / self.environment_type
            / self.agent_type
            / str(teacher_config["id"])
        )

        if self.environment_type == "ToySGD":
            result_dir = result_dir / env_config["function"]
            num_batches = env_config["num_batches"]

        if self.environment_type == "CMAES":
            num_batches = env_config["num_batches"]

            # Start with instance 0
            self.env.instance_index = -1

        buffer_size = num_runs * num_batches
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
            "num_batches": num_batches,
        }

        self.start_run = 0
        self.starting_points: list[np.ndarray] = []

        if checkpoint != 0:
            self._handle_checkpoint(result_dir, checkpoint)

    def generate_data(self) -> None:
        """Generate data based on ."""

    def _init_seeds(self) -> None:
        """Sets global seeds."""
        set_seeds(self.seed)
        self.env_config["seed"] = self.seed

    def _init_env(self) -> None:
        """Builds the environment based on the env_config field."""
        print(f"Environment: {self.env_config}")

        env = get_environment(self.env_config.copy())
        env.reset()

        batches_per_epoch = 1
        if self.environment_type == "SGD":
            print(f"Generating data for {self.env_config['dataset_name']}")
            if env.epoch_mode is False:
                num_epochs = self.env_config["num_epochs"]
                # if SGD env, translates num_batches to num_epochs
                batches_per_epoch = len(env.train_loader)
                print(f"One epoch consists of {batches_per_epoch} batches.")
                num_batches = num_epochs * batches_per_epoch
                self.env_config["cutoff"] = num_batches
            else:
                print("Currently running in epoch mode.")

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
