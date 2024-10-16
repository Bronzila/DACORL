from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import wandb

from src.utils.calculate_sgd_statistic import calc_mean_and_std_dev
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
) -> tuple[dict, float]:
    set_timeout(timeout)
    set_seeds(seed)
    print(f"Seed in trianing: {seed}")

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
    use_cmaes = env_type == "CMAES"
    agent = get_agent(
        agent_type,
        agent_config,
        tanh_scaling,
        hyperparameters,
        cmaes=use_cmaes,
    )

    # if agent_type == "bc":
    #     from CORL.algorithms.offline.any_percent_bc import (
    #         TrainConfig,
    #         keep_best_trajectories,

    #         replay_buffer,
    #         agent.discount,

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
    inc_value = np.inf

    print("Starting training with the following configuration...")
    print(f"Batch size: {batch_size}")
    print(f"HP config: {hyperparameters}")
    for t in range(int(num_train_iter)):
        print(f"{t}/{int(num_train_iter)}")
        batch = replay_buffer.sample(batch_size)
        log_dict = agent.train(batch)
        for k, v in log_dict.items():
            if k not in logs:
                logs[k] = [v]
            else:
                logs[k].append(v)

        if (not debug) and use_wandb:
            wandb.log(log_dict, agent.total_it)  # type: ignore

        if val_freq != 0 and (t + 1) % val_freq == 0:
            if env_type == "CMAES":
                test_agent = test_cma
            elif env_type == "SGD":
                test_agent = test_sgd
            else:
                test_agent = test_toy

            with torch.random.fork_rng():
                env = get_environment(run_info["environment"])
                if run_info["environment"]["type"] == "ToySGD":
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
                elif run_info["environment"]["type"] == "SGD":
                    # run_info["num_runs"] is not completely right here.
                    eval_runs = (
                        num_eval_runs
                        if num_eval_runs is not None
                        else run_info["num_runs"]
                    )
                    env.reset()
                    n_batches_total = run_info["environment"][
                        "num_epochs"
                    ] * len(env.train_loader)
                    if eval_protocol == "train":
                        eval_data = test_agent(
                            actor=agent.actor,
                            env=env,
                            n_runs=eval_runs,
                            n_batches=n_batches_total,
                            seed=run_info["seed"],
                        )
                    elif eval_protocol == "interpolation":
                        eval_data = test_agent(
                            actor=agent.actor,
                            env=env,
                            n_runs=eval_runs,
                            n_batches=n_batches_total,
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
                if run_info["environment"]["type"] == "ToySGD":
                    fbest_mean, _ = calc_mean_and_std_dev(eval_data, "f_cur")
                    print(f"Mean at iteration {t+1}: {fbest_mean}")
                    inc_value = (
                        fbest_mean
                        if inc_value is None
                        else np.min([inc_value, fbest_mean])
                    )
                elif run_info["environment"]["type"] == "SGD":
                    # Statistics for train set
                    train_loss_mean, train_loss_std = calc_mean_and_std_dev(
                        eval_data,
                        "train_loss",
                    )
                    print(
                        f"Mean train loss at iteration {t+1}: {train_loss_mean} +- {train_loss_std}",
                    )
                    train_acc_mean, train_acc_std = calc_mean_and_std_dev(
                        eval_data,
                        "train_acc",
                    )
                    print(
                        f"Mean train acc at iteration {t+1}: {train_acc_mean} +- {train_acc_std}",
                    )

                    # Statistics for validation set
                    valid_loss_mean, valid_loss_std = calc_mean_and_std_dev(
                        eval_data,
                        "valid_loss",
                    )
                    print(
                        f"Mean valid loss at iteration {t+1}: {valid_loss_mean} +- {valid_loss_std}",
                    )
                    valid_acc_mean, valid_acc_std = calc_mean_and_std_dev(
                        eval_data,
                        "valid_acc",
                    )
                    print(
                        f"Mean valid acc at iteration {t+1}: {valid_acc_mean} +- {valid_acc_std}",
                    )

                    # Statistics for test set
                    test_loss_mean, test_loss_std = calc_mean_and_std_dev(
                        eval_data,
                        "test_loss",
                    )
                    print(
                        f"Mean test loss at iteration {t+1}: {test_loss_mean} +- {test_loss_std}",
                    )
                    test_acc_mean, test_acc_std = calc_mean_and_std_dev(
                        eval_data,
                        "test_acc",
                    )
                    print(
                        f"Mean test acc at iteration {t+1}: {test_acc_mean} +- {test_acc_std}",
                    )

                    if inc_value is None:
                        inc_value = test_acc_mean
                    else:
                        inc_value = np.max([inc_value, test_acc_mean])
                elif env_type == "CMAES":
                    final_evaluations = eval_data.groupby("run").last()
                    function_group = final_evaluations.groupby("function_id")
                    for function in function_group:
                        fbests = function[1]["f_cur"].mean()
                        target_value = function[1]["target_value"].mean()
                        fid = int(function[1]["function_id"].mean())
                        print(
                            f"Mean at iteration {t+1}: {fbests} - {target_value} at function {cma_es_function[fid]}",
                        )
                    fbest_mean = final_evaluations["f_cur"].mean()

                if np.isnan(inc_value):
                    print(dict(hyperparameters))

    save_agent(agent.state_dict(), results_dir, t, seed)

    if (not debug) and use_wandb:
        wandb.finish()  # type: ignore

    return logs, inc_value
