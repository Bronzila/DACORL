from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
from benchmarking import (
    environment_agent_adjustments,
    parse_heterogeneous_teacher_name,
    read_teacher,
    save_combined_data,
)

from src.data_generator import DataGenerator
from src.evaluator import Evaluator
from src.trainer import Trainer
from src.utils import combine_runs, get_homogeneous_agent_paths, get_safe_original_cwd, load_agent

if TYPE_CHECKING:
    from src.utils import HydraConfig


def generate_data(cfg: HydraConfig, env_config: dict, data_gen_seeds: list[int]):
    n_runs = cfg.n_runs

    if cfg.combination == "single":
        agent_name = "default" if cfg.id == 0 else str(cfg.id)
        teacher_config = read_teacher(cfg.teacher, cfg.benchmark, agent_name)
        environment_agent_adjustments(env_config, teacher_config)

        # Generate data for different seeds
        for seed in data_gen_seeds:
            gen = DataGenerator(
                teacher_config=teacher_config,
                env_config=env_config,
                result_dir=cfg.results_dir / str(seed),
                check_if_exists=False,
                num_runs=n_runs,
                checkpoint=0,
                seed=seed.item(),
                verbose=False,
            )
            gen.generate_data()
            gen.save_data()

    elif cfg.combination == "homogeneous":
        for teacher_id in ["default", "1", "2", "3", "4"]:
            teacher_config = read_teacher(cfg.teacher, cfg.benchmark, teacher_id)
            environment_agent_adjustments(env_config, teacher_config)
            gen = DataGenerator(
                teacher_config=teacher_config,
                env_config=env_config,
                result_dir=cfg.results_dir / str(data_gen_seeds[0]),
                check_if_exists=False,
                num_runs=n_runs,
                checkpoint=0,
                seed=int(data_gen_seeds[0]),
                verbose=False,
            )
            gen.generate_data()
            gen.save_data()

        data_dir = cfg.results_dir / str(data_gen_seeds[0]) / env_config["type"] / cfg.teacher
        paths = get_homogeneous_agent_paths(data_dir, env_config.get("function", ""))
        combined_buffer, combined_run_info, combined_run_data = combine_runs(
            paths, "concat", 3000,
        )
        path = data_dir / "combined"
        save_combined_data(path, combined_buffer, combined_run_info, combined_run_data)

    elif cfg.combination == "heterogeneous":
        agent_name = "default"
        teachers_to_combine = parse_heterogeneous_teacher_name(cfg.teacher)
        data_dirs = []
        for teacher_type in teachers_to_combine:
            teacher_config = read_teacher(teacher_type, cfg.benchmark, agent_name)
            environment_agent_adjustments(env_config, teacher_config)
            gen = DataGenerator(
                teacher_config=teacher_config,
                env_config=env_config,
                result_dir=cfg.results_dir / str(data_gen_seeds[0]),
                check_if_exists=False,
                num_runs=n_runs,
                checkpoint=0,
                seed=int(data_gen_seeds[0]),
                verbose=False,
            )
            gen.generate_data()
            gen.save_data()

            data_dirs.append(cfg.results_dir / str(data_gen_seeds[0]) / env_config["type"] / teacher_type / "0" / env_config.get("function", ""))

        final_buffer_size = (len(data_dirs) + 1) * 500
        combined_buffer, combined_run_info, combined_run_data = combine_runs(
            data_dirs, "concat", final_buffer_size,
        )
        path = cfg.results_dir / str(data_gen_seeds[0]) / env_config["type"] / cfg.teacher
        save_combined_data(path, combined_buffer, combined_run_info, combined_run_data)

def train_model(cfg: HydraConfig, env_config: dict, data_gen_seeds: list[int]):

    rng = np.random.default_rng(0)
    train_seeds = rng.integers(0, 2**32 - 1, size=cfg.n_train_seeds)

    for train_seed in train_seeds:
        if cfg.combination == "single":
            data_dir = cfg.results_dir / str(data_gen_seeds[0]) / env_config["type"] / cfg.teacher / str(cfg.id)
        else:
            data_dir = cfg.results_dir / str(data_gen_seeds[0]) / env_config["type"] / cfg.teacher

        if env_config["type"] == "ToySGD":
            data_dir = data_dir / env_config["function"]

        evaluator = Evaluator(data_dir, cfg.eval_protocol, cfg.n_runs, cfg.eval_seed)

        trainer = Trainer(
            data_dir=data_dir,
            agent_config={"tanh_scaling": cfg.tanh_scaling, "batch_size": 256},
            agent_type=cfg.agent_type,
            evaluator=evaluator,
            seed=train_seed,
        )
        _, inc_value = trainer.train(cfg.n_train_iter, cfg.n_train_iter)
        print(inc_value)

def eval_agent(cfg: HydraConfig, env_config: dict, data_gen_seeds: list[int]) -> None:
    if cfg.combination == "single":
        data_dir = cfg.results_dir / str(data_gen_seeds[0]) / env_config["type"] / cfg.teacher / str(cfg.id)
    else:
        data_dir = cfg.results_dir / str(data_gen_seeds[0]) / env_config["type"] / cfg.teacher

    if env_config["type"] == "ToySGD":
        data_dir = data_dir / env_config["function"]

    agent_path = data_dir / "results" / cfg.agent_type / str(data_gen_seeds[0]) / str(cfg.n_train_iter)
    actor = load_agent(cfg.agent_type, agent_path).actor
    evaluator = Evaluator(data_dir, cfg.eval_protocol, cfg.n_runs, cfg.eval_seed)

    eval_data = evaluator.evaluate(actor)

    eval_data.to_csv(agent_path / f"eval_data_{cfg.eval_protocol}.csv")

@hydra.main(config_path="hydra_conf", config_name="config")
def main(cfg: HydraConfig):
    """One script to rule them all,
        one script to find them,
    One script to bring them all
        and in the darkness bind them.
    """
    # Ensure results_dir is specified
    if not cfg.results_dir:
        raise ValueError("The 'results_dir' must be specified as a command-line argument or in the config file.")

    cfg.results_dir = Path(get_safe_original_cwd(), cfg.results_dir)

    start = time.time()

    # Read environment config from file
    env_config_path = Path(get_safe_original_cwd(), "configs", "environment", f"{cfg.benchmark}", f"{cfg.env}.json")
    with env_config_path.open() as file:
        env_config = json.load(file)

    if env_config["type"] == "SGD" and cfg.instance_mode:
        env_config["instance_mode"] = cfg.instance_mode

    # Generate run seeds
    rng = np.random.default_rng(0)
    data_gen_seeds = rng.integers(0, 2**32 - 1, size=cfg.n_data_seeds)

    # Execute according to the specified mode
    if cfg.mode in ["data_gen", "both"]:
        generate_data(cfg, env_config, data_gen_seeds)

    if cfg.mode in ["train", "both"]:
        train_model(cfg, env_config, data_gen_seeds)

    # Perform evaluation only separately
    if cfg.mode in ["eval"]:
        eval_agent(cfg, env_config, data_gen_seeds)

    end = time.time()
    print(f"Took: {end-start}s to generate")

if __name__ == "__main__":
    main()
