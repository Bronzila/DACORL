import argparse
import time

from utils.train_agent import train_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train any offline agent ")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="path to the directory where replay_buffer and info about the replay_buffer are stored",
    )
    parser.add_argument(
        "--agent_type", type=str, default="td3_bc", choices=["td3_bc"]
    )
    parser.add_argument(
        "--agent_config",
        default=None,
        help="Not functional yet. Change configuration of the respective agent.",
    )
    parser.add_argument("--num_train_iter", type=int, default=5e6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--val_freq",
        type=int,
        help="how many training steps until the next validation sequence runs",
        default=250e3,
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

    args = parser.parse_args()
    start = time.time()

    train_agent(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        agent_config=args.agent_config,
        num_train_iter=args.num_train_iter,
        batch_size=args.batch_size,
        val_freq=args.val_freq,
        seed=args.seed,
        wandb_group=args.wandb_group,
        timeout=args.timeout,
    )

    end = time.time()
    print(f"Took: {end-start}s to generate")
