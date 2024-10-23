from __future__ import annotations

import json
import time
from pathlib import Path

from tap import Tap

from src.data_generator import DataGenerator, LayerwiseDataGenerator

if __name__ == "__main__":

    class DataGeneratorParser(Tap):
        benchmark: str = "SGD"
        env: str = "default"  # Config file to define the benchmark env
        num_runs: int = 1000
        seed: int = 0
        teacher: str = "step_decay"
        result_dir: Path  # path to the directory where replay_buffer and info about the replay_buffer are stored
        instance_mode: str | None = None  # Select the instance mode for SGD Benchmark
        id: str = "0"
        checkpointing_freq: int = 0  # How frequent we want to checkpoint. Default 0 means no checkpoints
        checkpoint: int = 0  # Specify which checkpoint you want to load. Default 0 means no loading
        check_if_exists: bool = False
        verbose: bool = False
        """Generate Data using a teacher in a specified environment."""

    args = DataGeneratorParser().parse_args()

    start = time.time()

    agent_name = "default" if args.id == "0" else str(args.id)
    # Read agent config from file
    teacher_config_path = Path(
        "configs",
        "agents",
        args.teacher,
        f"{args.benchmark}",
        f"{agent_name}.json",
    )
    with teacher_config_path.open() as file:
        teacher_config = json.load(file)

    # Read environment config from file
    env_config_path = Path(
        "configs", "environment", f"{args.benchmark}", f"{args.env}.json",
    )
    with env_config_path.open() as file:
        env_config = json.load(file)

    # Add initial learning rate to agent config for SGDR
    if teacher_config["type"] == "sgdr":
        teacher_config["params"]["initial_learning_rate"] = env_config[
            "initial_learning_rate"
        ]

    if teacher_config["type"] == "constant" and teacher_config["id"] == 0:
        teacher_config["params"]["learning_rate"] = env_config[
            "initial_learning_rate"
        ]
    elif teacher_config["type"] == "constant":
        env_config["initial_learning_rate"] = teacher_config["params"][
            "learning_rate"
        ]

    if (env_config["type"] == "SGD" or env_config["type"] == "LayerwiseSGD") and args.instance_mode:
        env_config["instance_mode"] = args.instance_mode

    if env_config["type"] == "LayerwiseSGD":
        generator_class = LayerwiseDataGenerator
    else:
        generator_class = DataGenerator

    generator = generator_class(
        teacher_config,
        env_config,
        args.result_dir,
        args.check_if_exists,
        args.num_runs,
        args.checkpoint,
        args.seed,
        args.verbose,
    )

    generator.generate_data(args.checkpointing_freq)
    generator.save_data()

    end = time.time()
    print(f"Took: {end-start}s to generate")
