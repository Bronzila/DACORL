import json
from pathlib import Path

import torch
import numpy as np
from src.utils.general import get_agent, get_environment, load_agent
from src.utils.test_agent import test_agent
import time
import argparse

from tap import Tap


if __name__ == "__main__":

    class EvalParser(Tap):
        data_dir: Path
        """Path to the directory where replay_buffer and info about the replay_buffer are stored."""
        agent_type: str = "td3_bc"
        num_runs: int = 100
        training_seed: int = 100
        num_train_iter: int = 30000
        "Number of training iterations, on which the agent was trained on."
        eval_protocol: str = "train"
        gen_seed: int = 123
        """Seed used to generate random seeds."""

    args = EvalParser().parse_args()

    agent_path = Path(
        args.data_dir,
        "results",
        args.agent_type,
        str(args.training_seed),
        f"{args.num_train_iter}",
    )
    with Path(args.data_dir, "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    # Prepare env and agent
    env = get_environment(run_info["environment"])

    state = env.reset()
    state_dim = state[0].shape[0]
    agent_config = {
        "state_dim": state_dim,
        "action_dim": 1,
        "max_action": 0,
        "min_action": -10,
    }
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
            starting_points=run_info["starting_points"],
        )
    elif args.eval_protocol == "interpolation":
        rng = np.random.default_rng(args.gen_seed)
        random_eval_seed = int(rng.integers(0, 2**32 - 1, size=1)[0])
        eval_data = test_agent(
            actor=agent.actor,
            env=env,
            n_runs=args.num_runs,
            n_batches=run_info["environment"]["num_batches"],
            seed=random_eval_seed,
        )
    # Save evaluation data
    eval_data.to_csv(agent_path / f"eval_data_{args.eval_protocol}.csv")
