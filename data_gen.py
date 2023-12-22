import argparse
import time

from src.utils.generate_data import generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run any agent on the ToySGD benchmark")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent_type", type=str, default="step_decay")
    parser.add_argument("--environment_type", type=str, default="ToySGD")
    parser.add_argument("--results_dir", type=str, default="")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="path to the directory where replay_buffer and info about the replay_buffer are stored",
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
        "--timeout",
        type=int,
        default=0,
        help="Timeout in sec. 0 -> no timeout",
    )

    args = parser.parse_args()
    start = time.time()

    agent_config = {
        "gamma": 0.2,
        "step_size": 2,
    }

    generate_dataset(
        agent_type=args.agent_type,
        agent_config=agent_config,
        environment_type=args.environment_type,
        num_runs=args.num_runs,
        num_batches=args.num_batches,
        seed=args.seed,
        timeout=args.timeout,
        results_dir=args.results_dir,
        save_run_data=args.save_run_data,
        save_rep_buffer=args.save_rep_buffer,
    )

    end = time.time()
    print(f"Took: {end-start}s to generate")