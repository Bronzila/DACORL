import argparse
import time
import json
from pathlib import Path

from src.utils.train_agent import train_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train any offline agent ")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="path to the directory where replay_buffer and info about the replay_buffer are stored",
    )
    parser.add_argument(
        "--agent_type", type=str, default="td3_bc", choices=["td3_bc"]
    )
    parser.add_argument(
        "--agent_config",
        default={},
        help="Not functional yet. Change configuration of the respective agent.",
    )
    parser.add_argument("--num_train_iter", type=int, default=15000)
    parser.add_argument("--num_eval_runs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--val_freq",
        type=int,
        help="how many training steps until the next validation sequence runs",
        default=15000,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Use to group certain runs in wandb.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Timeout in sec. 0 -> no timeout",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run for max. 5 iterations and don't log in wanbd.",
    )
    parser.add_argument(
        "--eval_protocol", type=str, default="train", choices=["train", "interpolation"]
    )
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hp_path", type=str)

    args = parser.parse_args()
    start = time.time()

    batch_size = args.batch_size
    if args.hp_path:
        with Path(args.hp_path).open("r") as f:
            hyperparameters = json.load(f)
        batch_size = hyperparameters["batch_size"]
    else:
        hyperparameters = {}
    hyperparameters["hidden_dim"] = args.hidden_dim
    print(hyperparameters)

    _, mean = train_agent(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        agent_config=args.agent_config,
        num_train_iter=args.num_train_iter,
        num_eval_runs=args.num_eval_runs,
        batch_size=batch_size,
        val_freq=args.val_freq,
        seed=args.seed,
        wandb_group=args.wandb_group,
        timeout=args.timeout,
        debug=args.debug,
        eval_protocol=args.eval_protocol,
        eval_seed=args.eval_seed,
        hyperparameters=hyperparameters,
    )
    print(mean)
    end = time.time()
    print(f"Took: {end-start}s to generate")
