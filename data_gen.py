import argparse
import json
import time

from src.utils.generate_data import generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run any agent on the ToySGD benchmark"
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent_config", type=str,
                        help="Path to the agent configuration",
                        default="configs/agents/step_decay/default.json")
    parser.add_argument("--env_config", type=str,
                        help="Path to the environment configuration",
                        default="configs/environment/Rosenbrock_default.json")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
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

    # Read agent config from file
    with open(args.agent_config, "r") as file:
        agent_config = json.load(file)

    # Read environment config from file
    with open(args.env_config, "r") as file:
        env_config = json.load(file)

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
