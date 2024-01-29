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
    parser.add_argument("--num_eval_iter", type=int, default=1000)
    parser.add_argument(
        "--num_train_iter",
        type=int,
        default=100,
        help="Number of training iterations, on which the agent was trained on.",
    )

    args = parser.parse_args()

    agent_path = Path(
        args.data_dir, "results", args.agent_type, f"{args.num_train_iter}"
    )
    with Path(args.data_dir, "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    # Prepare env and agent
    env = get_environment(run_info["environment"])

    state = env.reset()
    state_dim = state[0].shape[0]
    agent_config = {"state_dim": state_dim, "action_dim": 1, "max_action": 1}
    agent = load_agent(args.agent_type, agent_config, agent_path)

    # Evaluate agent
    eval_data = test_agent(
        actor=agent.actor,
        env=env,
        n_episodes=args.num_eval_iter,
        seed=run_info["seed"],
    )

    # Save evaluation data
    eval_dir = Path(args.data_dir, "eval")
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True)
    eval_data.to_csv(eval_dir / "eval_data.csv")
