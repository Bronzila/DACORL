import argparse
import json
from pathlib import Path
import numpy as np
from src.utils.generate_data import generate_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher HPO")
    parser.add_argument(
        "init_lr",
        type=float,
    )

    args = parser.parse_args()

    data_dir = Path("data", "SGD_teacher_hpo_const")

    env_config_path = Path("configs", "environment", "SGD", "default.json")
    with env_config_path.open() as file:
        env_config = json.load(file)

    agent_config_path = Path("configs", "agents", "constant", "default.json")
    with agent_config_path.open() as file:
        agent_config = json.load(file)

    agent_config.update({"params": {"initial_learning_rate": args.init_lr}})
    env_config.update({"initial_learning_rate": args.init_lr})

    print(env_config)
    agg_run_data = generate_dataset(
        agent_config,
        env_config,
        num_runs=10,
        seed=0,
        results_dir=data_dir,
        timeout=0,
        save_run_data=True,
        save_rep_buffer=True,
    )

    final_evaluations = agg_run_data.groupby("run").last()
    fbests = final_evaluations["f_cur"]
    fct = env_config["function"]
    print(f"Results on {fct} with initial lr {args.init_lr}: {fbests.mean()}")
