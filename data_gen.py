import json
from pathlib import Path
import time
from tap import Tap

from src.utils.generate_data import generate_dataset

if __name__ == "__main__":

    class DataArgumentParser(Tap):
        benchmark: str = "SGD"
        env: str = "default"  # Config file to define the benchmark env
        num_runs: int = 1000
        seed: int = 0
        agent: str = "step_decay"  # Agent for data generation
        results_dir: Path = ""
        """Path to the directory where replay_buffer + info are stored"""
        instance_mode: str = None
        """Select the instance mode for the SGD Benchmark"""
        save_run_data: bool = True
        save_rep_buffer: bool = True
        id: int = 0  # Agent ID
        timeout: int = 0  # Timeout in sec. 0 -> no timeout
        checkpointing_freq: int = 0
        """How frequent we want to checkpoint. Default 0 means no checkpoints"""
        checkpoint: int = 0
        """Specify which checkpoint (run number) you want to load. Default 0 means no loading"""
        check_if_exists: bool = False

    args = DataArgumentParser().parse_args()
    start = time.time()

    agent_name = "default" if args.id == 0 else str(args.id)
    # Read agent config from file
    agent_config_path = Path(
        "configs",
        "agents",
        args.agent,
        f"{args.benchmark}",
        f"{agent_name}.json",
    )
    with agent_config_path.open() as file:
        agent_config = json.load(file)

    # Read environment config from file
    env_config_path = Path(
        "configs", "environment", f"{args.benchmark}", f"{args.env}.json"
    )
    with env_config_path.open() as file:
        env_config = json.load(file)

    # Add initial learning rate to agent config for SGDR
    if agent_config["type"] == "sgdr":
        agent_config["params"]["initial_learning_rate"] = env_config[
            "initial_learning_rate"
        ]

    if env_config["type"] == "SGD":
        if args.instance_mode:
            env_config["instance_mode"] = args.instance_mode

    generate_dataset(
        agent_config=agent_config,
        env_config=env_config,
        num_runs=args.num_runs,
        seed=args.seed,
        timeout=args.timeout,
        results_dir=args.results_dir,
        checkpointing_freq=args.checkpointing_freq,
        checkpoint=args.checkpoint,
        save_run_data=args.save_run_data,
        save_rep_buffer=args.save_rep_buffer,
        check_if_exists=args.check_if_exists,
    )

    end = time.time()
    print(f"Took: {end-start}s to generate")
