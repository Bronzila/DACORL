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
    parser.add_argument("--num_runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
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
        default="",
        help="Select the instance mode for SGD Benchmark.",
    )
    parser.add_argument(
        "--save_run_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save all the run data in a csv",
    )
    parser.add_argument(
        "--save_rep_buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the rep buffer",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="Agent ID",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout in sec. 0 -> no timeout",
    )

    args = parser.parse_args()
    start = time.time()

    agent_name = "default" if args.id == 0 else str(args.id)
    # Read agent config from file
    agent_config_path = Path("configs", "agents", args.agent, f"{agent_name}.json")
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

    if env_config["type"] == "SGD":
        env_config["instance_mode"] = args.instance_mode

    generate_dataset(
        agent_config=agent_config,
        env_config=env_config,
        num_runs=args.num_runs,
        seed=args.seed,
        timeout=args.timeout,
        results_dir=args.results_dir,
        save_run_data=args.save_run_data,
        save_rep_buffer=args.save_rep_buffer,
    )

    end = time.time()
    print(f"Took: {end-start}s to generate")