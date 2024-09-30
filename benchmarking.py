import argparse
import json
from pathlib import Path
import time

from src.utils.generate_data import generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run any agent on benchmarks"
    )
    parser.add_argument("--benchmark", type=str, default="SGD")
    parser.add_argument(
        "--env",
        type=str,
        help="Config file to define the benchmark env",
        default="default",
    )
    parser.add_argument("--seeds", type=int, default=[0], nargs="*")
    parser.add_argument(
        "--agent",
        type=str,
        help="Agent for data generation",
        default="step_decay",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="path to the directory where replay_buffer and info about the replay_buffer are stored",
    )
    parser.add_argument(
        "--instance_mode",
        type=str,
        default=None,
        help="Select the instance mode for SGD Benchmark.",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="0",
        help="Agent ID",
    )

    args = parser.parse_args()

    agent_name = "default" if args.id == "0" else str(args.id)
    # Read agent config from file
    agent_config_path = Path("configs", "agents", args.agent, f"{args.benchmark}", f"{agent_name}.json")
    with agent_config_path.open() as file:
        agent_config = json.load(file)

    # Read environment config from file
    env_config_path = Path("configs", "environment", f"{args.benchmark}", f"{args.env}.json")
    with env_config_path.open() as file:
        env_config = json.load(file)

    # Add initial learning rate to agent config for SGDR
    if agent_config["type"] == "sgdr":
        agent_config["params"]["initial_learning_rate"] = env_config["initial_learning_rate"]

    if agent_config["type"] == "constant" and agent_config["id"] == 0:
        agent_config["params"]["learning_rate"] = env_config["initial_learning_rate"]
    elif agent_config["type"] == "constant":
        env_config["initial_learning_rate"] = agent_config["params"]["learning_rate"]

    num_runs = 100
    if env_config["type"] == "SGD":
        num_runs = 5
        if args.instance_mode:
            env_config["instance_mode"] = args.instance_mode

    for seed in args.seeds:
        _ = generate_dataset(
            agent_config=agent_config,
            env_config=env_config,
            num_runs=num_runs,
            seed=seed,
            timeout=0,
            results_dir=args.results_dir,
            save_run_data=True,
            save_rep_buffer=True,
            checkpointing_freq=0,
        )
    