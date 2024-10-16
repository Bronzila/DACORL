import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_generator import DataGenerator
from src.utils.combinations import combine_runs, get_homogeneous_agent_paths
from src.utils.replay_buffer import ReplayBuffer
from src.utils.train_agent import train_agent


def read_teacher(teacher_type: str, benchmark: str, teacher_name: str) -> Any:
    agent_config_path = Path("configs", "agents", teacher_type, f"{benchmark}", f"{teacher_name}.json")
    with agent_config_path.open() as file:
        return json.load(file)

def environment_agent_adjustments(env_config: dict, agent_config: dict) -> None:
    # Add initial learning rate to agent config for SGDR
    if agent_config["type"] == "sgdr":
        agent_config["params"]["initial_learning_rate"] = env_config["initial_learning_rate"]

    if agent_config["type"] == "constant" and agent_config["id"] == 0:
        agent_config["params"]["learning_rate"] = env_config["initial_learning_rate"]
    elif agent_config["type"] == "constant":
        env_config["initial_learning_rate"] = agent_config["params"]["learning_rate"]

def parse_heterogeneous_teacher_name(teacher_name: str) -> list[str]:
    # Define a dictionary that maps the abbreviations to the actual schedules
    teacher_map = {
        "ST": "step_decay",
        "SG": "sgdr",
        "C": "constant",
        "E": "exponential_decay",
    }

    teacher_codes = teacher_name.split("-")

    return [teacher_map.get(code, "Unknown") for code in teacher_codes]

def save_combined_data(path: Path, buffer: ReplayBuffer, run_info: dict, run_data: pd.DataFrame) -> None:
    path.mkdir(parents=True, exist_ok=True)
    buffer_path = path / "rep_buffer"
    run_info_path = path / "run_info.json"
    run_data_path = path / "aggregated_run_data.csv"
    buffer.save(buffer_path)
    with run_info_path.open(mode="w") as f:
        json.dump(run_info, f, indent=4)
    run_data.to_csv(run_data_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run any agent on benchmarks",
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
    parser.add_argument(
        "--combination",
        type=str,
        default="single",
        choices=[
            "homogeneous",
            "heterogeneous",
            "single",
        ],
    )

    args = parser.parse_args()

    # Read environment config from file
    env_config_path = Path("configs", "environment", f"{args.benchmark}", f"{args.env}.json")
    with env_config_path.open() as file:
        env_config = json.load(file)

    # Experimental details
    results_dir = Path(args.results_dir)
    num_runs = 100
    num_train_iter = 300

    if env_config["type"] == "SGD":
        num_runs = 5
        if args.instance_mode:
            env_config["instance_mode"] = args.instance_mode

    # generate run seeds randomly
    rng = np.random.default_rng(0)

    data_gen_seeds = rng.integers(0, 2**32 - 1, size=args.n_data_seeds)

    if args.combination == "single":
        agent_name = "default" if args.id == "0" else str(args.id)
        # Read agent config from file
        teacher_config = read_teacher(args.teacher, args.benchmark, agent_name)
        environment_agent_adjustments(env_config, teacher_config)

        # generate data for different seeds
        for seed in data_gen_seeds:
            gen = DataGenerator(
                teacher_config=teacher_config,
                env_config=env_config,
                result_dir=results_dir / str(seed),
                check_if_exists=False,
                num_runs=num_runs,
                checkpoint=0,
                seed=seed.item(),
                verbose=False,
            )
            gen.generate_data()
            gen.save_data()

    elif args.combination == "homogeneous":
        for teacher_id in ["default", "1", "2", "3", "4"]:
            teacher_config = read_teacher(args.teacher, args.benchmark, teacher_id)
            environment_agent_adjustments(env_config, teacher_config)
            gen = DataGenerator(
                teacher_config=teacher_config,
                env_config=env_config,
                result_dir=results_dir / str(data_gen_seeds[0]),
                check_if_exists=False,
                num_runs=num_runs,
                checkpoint=0,
                seed=int(data_gen_seeds[0]),
                verbose=False,
            )
            gen.generate_data()
            gen.save_data()

        data_dir = results_dir / str(data_gen_seeds[0]) / env_config["type"] / args.teacher
        paths = get_homogeneous_agent_paths(data_dir, env_config.get("function", ""))
        combined_buffer, combined_run_info, combined_run_data = combine_runs(
            paths, "concat", 3000, # buffer size not needed here as we only use concat strategy
        )
        path = data_dir / "combined"
        save_combined_data(path, combined_buffer, combined_run_info, combined_run_data)
    elif args.combination == "heterogeneous":
        agent_name = "default"
        teachers_to_combine = parse_heterogeneous_teacher_name(args.teacher)
        data_dirs = []
        for teacher_type in teachers_to_combine:
            teacher_config = read_teacher(teacher_type, args.benchmark, agent_name)
            environment_agent_adjustments(env_config, teacher_config)
            gen = DataGenerator(
                teacher_config=teacher_config,
                env_config=env_config,
                result_dir=results_dir / str(data_gen_seeds[0]),
                check_if_exists=False,
                num_runs=num_runs,
                checkpoint=0,
                seed=int(data_gen_seeds[0]),
                verbose=False,
            )
            gen.generate_data()
            gen.save_data()

            data_dirs.append(results_dir / str(data_gen_seeds[0]) / env_config["type"] / teacher_type / "0" / env_config.get("function", ""))

        final_buffer_size = (len(data_dirs) + 1) * 500 # not needed as we only concatenate here
        combined_buffer, combined_run_info, combined_run_data = combine_runs(
            data_dirs, "concat", final_buffer_size,
        )
        path = results_dir / str(data_gen_seeds[0]) / env_config["type"] / args.teacher
        save_combined_data(path, combined_buffer, combined_run_info, combined_run_data)

    # Train on one seed for multiple training seeds

    rng = np.random.default_rng(0)

    train_seeds = rng.integers(0, 2**32 - 1, size=args.n_train_seeds)

    # train on generated data by first data_gen_seed

    for train_seed in train_seeds:
        if args.combination == "single":
            if env_config["type"] == "SGD":
                data_dir = results_dir / str(data_gen_seeds[0]) / env_config["type"] / args.teacher / str(args.id)
            elif env_config["type"] == "ToySGD":
                data_dir = results_dir / str(data_gen_seeds[0]) / env_config["type"] / args.teacher / str(args.id) / env_config["function"]
            _, mean = train_agent(
                data_dir=data_dir,
                agent_type=args.agent_type,
                agent_config={},
                num_train_iter=num_train_iter,
                num_eval_runs=num_runs,
                batch_size=256,
                val_freq=num_train_iter,
                seed=train_seed,
                wandb_group="",
                timeout=0,
                debug=False,
                use_wandb=False,
                hyperparameters={},
                eval_protocol="train",
                eval_seed=0,
                tanh_scaling=args.tanh_scaling,
            )
        else:
            _, mean = train_agent(
                data_dir=path,
                agent_type=args.agent_type,
                agent_config={},
                num_train_iter=num_train_iter,
                num_eval_runs=num_runs,
                batch_size=256,
                val_freq=num_train_iter,
                seed=train_seed,
                wandb_group="",
                timeout=0,
                debug=False,
                use_wandb=False,
                hyperparameters={},
                eval_protocol="train",
                eval_seed=0,
                tanh_scaling=args.tanh_scaling,
            )