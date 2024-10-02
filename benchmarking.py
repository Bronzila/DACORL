import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.generate_data import generate_dataset
from src.utils.train_agent import train_agent

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
    parser.add_argument("--n_data_seeds", type=int, default=5)
    parser.add_argument("--n_train_seeds", type=int, default=5)
    parser.add_argument(
        "--teacher",
        type=str,
        help="Teacher for data generation",
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
    parser.add_argument(
        "--agent_type",
        type=str,
        default="td3_bc",
        choices=[
            "bc",
            "td3_bc",
            "cql",
            "awac",
            "edac",
            "sac_n",
            "lb_sac",
            "iql",
            "td3",
        ],
    )
    parser.add_argument(
        "--tanh_scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()

    agent_name = "default" if args.id == "0" else str(args.id)
    # Read agent config from file
    agent_config_path = Path("configs", "agents", args.teacher, f"{args.benchmark}", f"{agent_name}.json")
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

    # Experimental details
    results_dir = Path(args.results_dir)
    num_runs = 1000
    num_train_iter = 30000

    if env_config["type"] == "SGD":
        num_runs = 5
        if args.instance_mode:
            env_config["instance_mode"] = args.instance_mode

    # generate run seeds randomly
    rng = np.random.default_rng(0)

    data_gen_seeds = rng.integers(0, 2**32 - 1, size=args.n_data_seeds)

    # generate data for different seeds
    for seed in data_gen_seeds:
        _ = generate_dataset(
            agent_config=agent_config,
            env_config=env_config,
            num_runs=num_runs,
            seed=int(seed),
            timeout=0,
            results_dir=results_dir / str(seed),
            save_run_data=True,
            save_rep_buffer=True,
            checkpointing_freq=0,
            checkpoint=0,
        )


    # Train on one seed for multiple training seeds

    rng = np.random.default_rng(0)

    train_seeds = rng.integers(0, 2**32 - 1, size=args.n_train_seeds)

    # train on generated data by first data_gen_seed
    if env_config["type"] == "SGD":
        data_dir = results_dir / str(data_gen_seeds[0]) / env_config["type"] / args.teacher / str(args.id)
    elif env_config["type"] == "ToySGD":
        data_dir = results_dir / str(data_gen_seeds[0]) / env_config["type"] / args.teacher / str(args.id) / env_config["function"]

    for train_seed in train_seeds:
        _, mean = train_agent(
            data_dir=data_dir,
            agent_type=args.agent_type,
            agent_config={},
            num_train_iter=num_train_iter,
            num_eval_runs=num_runs,
            batch_size=256,
            val_freq=num_train_iter,
            seed=train_seed,
            wandb_group=None,
            timeout=0,
            debug=False,
            use_wandb=False,
            hyperparameters={},
            eval_protocol="train",
            eval_seed=0,
            tanh_scaling=args.tanh_scaling,
        )