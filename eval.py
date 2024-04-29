import json
from pathlib import Path

import torch
from src.utils.general import get_agent, get_environment, load_agent
from src.utils.test_agent import test_agent
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="path to the directory where replay_buffer and info about the replay_buffer are stored",
    )
    parser.add_argument("--agent_type", type=str, default="td3_bc")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--training_seed", type=int, default=100)
    parser.add_argument(
        "--num_train_iter",
        type=int,
        default=15000,
        help="Number of training iterations, on which the agent was trained on.",
    )
    parser.add_argument(
        "--eval_protocol", type=str, default="train", choices=["train", "interpolation"]
    )
    parser.add_argument("--eval_seed", type=int, default=123)

    args = parser.parse_args()

    agent_path = Path(
        args.data_dir, "results", args.agent_type, str(args.training_seed), f"{args.num_train_iter}"
    )
    with Path(args.data_dir, "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    # Prepare env and agent
    env = get_environment(run_info["environment"])

    state = env.reset()
    state_dim = state[0].shape[0]
    agent_config = {"state_dim": state_dim, "action_dim": 1, "max_action": 0, "min_action": -10}
    agent = load_agent(args.agent_type, agent_config, agent_path)
    print(f"Evaluating agent in {agent_path}")
    # Evaluate agent
    if args.eval_protocol == "train":
        eval_data = test_agent(
            actor=agent.actor,
            env=env,
            n_runs=args.num_runs,
            n_batches=run_info["environment"]["num_batches"],
            seed=run_info["seed"],
            starting_points=run_info["starting_points"]
        )
    elif args.eval_protocol == "interpolation":
        eval_data = test_agent(
            actor=agent.actor,
            env=env,
            n_runs=args.num_runs,
            n_batches=run_info["environment"]["num_batches"],
            seed=args.eval_seed,
        )
    # Save evaluation data
    eval_data.to_csv(agent_path / "eval_data.csv")
