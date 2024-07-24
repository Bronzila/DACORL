from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import wandb

from src.utils.general import (
    get_agent,
    get_environment,
    save_agent,
    set_seeds,
    set_timeout,
)
from src.utils.replay_buffer import ReplayBuffer
from src.utils.test_agent import test_agent as test_toy
from src.utils.test_cma import test_agent as test_cma
from src.utils.test_sgd import test_agent as test_sgd

cma_es_function = {
    12: "BentCigar",
    11: "Discus",
    2: "Ellipsoid",
    23: "Katsuura",
    15: "Rastrigin",
    8: "Rosenbrock",
    17: "Schaffers",
    20: "Schwefel",
    1: "Sphere",
    16: "Weierstrass",
}


def train_agent(
    data_dir: str,
    agent_type: str,
    agent_config: dict,
    num_train_iter: int,
    batch_size: int,
    val_freq: int,
    seed: int,
    wandb_group: str,
    timeout: int,
    debug: bool,
    hyperparameters: dict,
    eval_protocol: str,
    eval_seed: int,
    tanh_scaling: bool,
    use_wandb: bool = False,
    num_eval_runs: int | None = None,
) -> None:
    if debug:
        num_train_iter = 5
        val_freq = 5

    set_timeout(timeout)
    set_seeds(seed)

    results_dir = Path(data_dir, "results", agent_type)
    with Path(data_dir, "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    env_type = run_info["environment"]["type"]

    replay_buffer = ReplayBuffer.load(Path(data_dir, "rep_buffer"))
    replay_buffer.seed(seed)
    state, _, _, _, _ = replay_buffer.sample(1)
    state_dim = state.shape[1]

    agent_config.update(
        {
            "state_dim": state_dim,
            "action_dim": 1,
            "max_action": 0 if env_type != "CMAES" else 10,
            "min_action": -10 if env_type != "CMAES" else 0,
        },
    )
    agent = get_agent(agent_type, agent_config, tanh_scaling, hyperparameters)

    # if agent_type == "bc":
    #     from CORL.algorithms.offline.any_percent_bc import (
    #         TrainConfig,
    #         keep_best_trajectories,
    #     )

    #     replay_buffer = keep_best_trajectories(
    #         replay_buffer,
    #         TrainConfig().frac,
    #         agent.discount,
    #         run_info["num_batches"],
    #     )

    if (not debug) and use_wandb:
        fct = run_info["environment"]["function"]
        teacher = run_info["agent"]["type"]
        state_version = run_info["environment"]["state_version"]
        wandb.init(  # type: ignore
            project="DAC4DL",
            entity="study_project",
            group=wandb_group,
            config=hyperparameters,
            name=f"{agent_type}-{teacher}-{fct}-{state_version}",
            tags=["agent_test", f"{agent_type}", f"{teacher}", f"{fct}"],
        )

    logs: dict = {}
    min_fbest = np.inf
    for t in range(int(num_train_iter)):
        batch = replay_buffer.sample(batch_size)
        log_dict = agent.train(batch)
        for k, v in log_dict.items():
            if k not in logs:
                logs[k] = [v]
            else:
                logs[k].append(v)

        if (not debug) and use_wandb:
            wandb.log(log_dict, agent.total_it)

        if val_freq != 0 and (t + 1) % val_freq == 0:
            if test_env == "CMAES":
                test_agent = test_cma
            elif env_type == "SGD":
                test_agent = test_sgd
            else:
                test_agent = test_toy

            with torch.random.fork_rng():
                env = get_environment(run_info["environment"])
                eval_runs = (
                    num_eval_runs
                    if num_eval_runs is not None
                    else len(run_info["starting_points"])
                )
                if eval_protocol == "train":
                    eval_data = test_agent(
                        actor=agent.actor,
                        env=env,
                        n_runs=eval_runs,
                        starting_points=run_info["starting_points"],
                        n_batches=run_info["environment"]["num_batches"],
                        seed=run_info["seed"],
                    )
                elif eval_protocol == "interpolation":
                    eval_data = test_agent(
                        actor=agent.actor,
                        env=env,
                        n_runs=eval_runs,
                        n_batches=run_info["environment"]["num_batches"],
                        seed=eval_seed,
                    )

                # Save agent early to enable continuation of pipeline
                save_agent(agent.state_dict(), results_dir, t, seed)
                eval_data.to_csv(
                    results_dir / str(seed) / f"{t + 1}" / "eval_data.csv",
                )
                with (
                    results_dir / str(seed) / f"{t + 1}" / "config.json"
                ).open("w") as f:
                    json.dump(dict(hyperparameters), f, indent=2)

                # Calculate mean performance for this checkpoint
                final_evaluations = eval_data.groupby("run").last()
                if env_type == "CMAES":
                    function_group = final_evaluations.groupby("function_id")
                    for function in function_group:
                        fbests = function[1]["f_cur"].mean()
                        target_value = function[1]["target_value"].mean()
                        fid = int(function[1]["function_id"].mean())
                        print(
                            f"Mean at iteration {t+1}: {fbests} - {target_value} at function {cma_es_function[fid]}",
                        )
                    fbest_mean = final_evaluations["f_cur"].mean()
                elif env_type == "SGD":
                    val_acc = final_evaluations["val_acc"]
                    val_acc_mean = val_acc.mean()
                    print(
                        f"Mean validation_acc at iteration {t+1}: {val_acc_mean}",
                    )
                else:
                    fbests = final_evaluations["f_cur"]
                    fbest_mean = fbests.mean()
                    print(f"Mean at iteration {t+1}: {fbest_mean}")
                min_fbest = np.min([min_fbest, fbest_mean])

    save_agent(agent.state_dict(), results_dir, t, seed)

    if (not debug) and use_wandb:
        wandb.finish()  # type: ignore

    return logs, min_fbest
