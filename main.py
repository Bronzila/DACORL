from __future__ import annotations

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

from src.data_generator import DataGenerator, LayerwiseDataGenerator
from src.evaluator import Evaluator, LayerwiseEvaluator
from src.trainer import Trainer
from src.utils import combine_runs, get_homogeneous_agent_paths, get_safe_original_cwd, load_agent

if TYPE_CHECKING:
    from src.utils import HydraConfig


def generate_data(cfg: HydraConfig, env_config: dict, seed: int):
    num_runs = env_config["num_runs"]

    GeneratorClass = LayerwiseDataGenerator if cfg.env.type == "LayerwiseSGD" else DataGenerator

    if cfg.combination == "single":
        agent_name = "default" if cfg.id == 0 else str(cfg.id)
        teacher_config = read_teacher(cfg.teacher, cfg.env.type, agent_name)
        environment_agent_adjustments(env_config, teacher_config)

        # Generate data for seed
        gen = GeneratorClass(
            teacher_config=teacher_config,
            env_config=env_config,
            result_dir=cfg.results_dir / str(seed),
            num_runs=num_runs,
            checkpoint=0,
            seed=seed,
            verbose=False,
        )
        gen.generate_data()
        gen.save_data()

    elif cfg.combination == "homogeneous":
        if not cfg.data_exists:
            for teacher_id in ["default", "1", "2", "3", "4"]:
                teacher_config = read_teacher(cfg.teacher, cfg.env.type, teacher_id)
                environment_agent_adjustments(env_config, teacher_config)
                gen = GeneratorClass(
                    teacher_config=teacher_config,
                    env_config=env_config,
                    result_dir=cfg.results_dir / str(seed),
                    num_runs=num_runs,
                    checkpoint=0,
                    seed=seed,
                    verbose=False,
                )
                gen.generate_data()
                gen.save_data()

        data_dir = cfg.results_dir / str(seed) / env_config["type"] / cfg.teacher
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
            if not cfg.data_exists:
                teacher_config = read_teacher(teacher_type, cfg.env.type, agent_name)
                environment_agent_adjustments(env_config, teacher_config)
                gen = GeneratorClass(
                    teacher_config=teacher_config,
                    env_config=env_config,
                    result_dir=cfg.results_dir / str(seed),
                    check_if_exists=False,
                    num_runs=num_runs,
                    checkpoint=0,
                    seed=seed,
                    verbose=False,
                )
                gen.generate_data()
                gen.save_data()

            data_dirs.append(cfg.results_dir / str(seed) / env_config["type"] / teacher_type / "0" / env_config.get("function", ""))

        final_buffer_size = (len(data_dirs) + 1) * 500
        combined_buffer, combined_run_info, combined_run_data = combine_runs(
            data_dirs, "concat", final_buffer_size,
        )
        path = cfg.results_dir / str(seed) / env_config["type"] / cfg.teacher
        save_combined_data(path, combined_buffer, combined_run_info, combined_run_data)

def train_model(cfg: HydraConfig, env_config: dict, seed: int):
    if cfg.combination == "single":
        data_dir = cfg.results_dir / str(seed) / env_config["type"] / cfg.teacher / str(cfg.id)
    else:
        data_dir = cfg.results_dir / str(seed) / env_config["type"] / cfg.teacher
        if cfg.combination == "homogeneous":
            data_dir = data_dir / "combined"

    if env_config["type"] == "ToySGD":
        data_dir = data_dir / env_config["function"]

    EvaluatorClass = LayerwiseEvaluator if cfg.env.type == "LayerwiseSGD" else Evaluator

    evaluator = EvaluatorClass(data_dir, cfg.eval_protocol, env_config["num_runs"], cfg.eval_seed)

    trainer = Trainer(
        data_dir=data_dir,
        agent_config={"tanh_scaling": cfg.tanh_scaling, "batch_size": 256},
        agent_type=cfg.agent_type,
        evaluator=evaluator,
        seed=seed,
    )
    _, inc_value = trainer.train(cfg.num_train_iter, cfg.num_train_iter)
    print(inc_value)

def eval_agent(cfg: HydraConfig, env_config: dict, seed: int) -> None:
    if cfg.combination == "single":
        data_dir = cfg.results_dir / str(seed) / env_config["type"] / cfg.teacher / str(cfg.id)
    else:
        data_dir = cfg.results_dir / str(seed) / env_config["type"] / cfg.teacher
        if cfg.combination == "homogeneous":
            data_dir = data_dir / "combined"


    if env_config["type"] == "ToySGD":
        data_dir = data_dir / env_config["function"]

    agent_path = data_dir / "results" / cfg.agent_type / str(seed) / str(cfg.num_train_iter)
    actor = load_agent(cfg.agent_type, agent_path).actor

    EvaluatorClass = LayerwiseEvaluator if cfg.env.type == "LayerwiseSGD" else Evaluator
    evaluator = EvaluatorClass(data_dir, cfg.eval_protocol, env_config["num_runs"], cfg.eval_seed)

    eval_data = evaluator.evaluate(actor)

    eval_data.to_csv(agent_path / f"eval_data_{cfg.eval_protocol}.csv")

@hydra.main(config_path="hydra_conf", config_name="config", version_base="1.1")
def main(cfg: HydraConfig):
    """One script to rule them all,
        one script to find them,
    One script to bring them all
        and in the darkness bind them.
    """
    # Ensure results_dir is specified
    if not cfg.results_dir:
        raise ValueError("The 'results_dir' must be specified as a command-line argument or in the config file.")
    if cfg.mode not in ["data_gen", "train", "eval", "all"]:
        raise ValueError(f"Unknown run mode: {cfg.mode}. Please use 'data_gen', 'train', 'eval' or the default 'all'.")

    cfg.results_dir = Path(get_safe_original_cwd(), cfg.results_dir)

    start = time.time()

    # Get environment config
    env_config = cfg._to_content(cfg, resolve=True, throw_on_missing=False)["env"]
    print(env_config)

    if env_config["type"] == "SGD" and cfg.instance_mode:
        env_config["instance_mode"] = cfg.instance_mode

    # Generate run seeds
    rng = np.random.default_rng(cfg.seed)
    random_seed = int(rng.integers(0, 2**32 - 1))

    # Execute according to the specified mode
    if cfg.mode in ["data_gen", "all"]:
        generate_data(cfg, env_config, random_seed)

    if cfg.mode in ["train", "all"]:
        train_model(cfg, env_config, random_seed)

    # Perform evaluation only separately
    if cfg.mode in ["eval"]:
        eval_agent(cfg, env_config, random_seed)

    end = time.time()
    print(f"Took: {end-start}s to generate")

if __name__ == "__main__":
    main()
