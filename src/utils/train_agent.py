from __future__ import annotations

import json
from pathlib import Path

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
from src.utils.test_agent import test_agent
from typing import Optional


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
    num_eval_runs: int | None = None,
    eval_protocol: str="train",
    eval_seed: int=123,
) -> None:

    set_timeout(timeout)
    set_seeds(seed)

    results_dir = Path(data_dir, "results", agent_type)
    with Path(data_dir, "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    replay_buffer = ReplayBuffer.load(Path(data_dir, "rep_buffer"))
    state, _, _, _, _ = replay_buffer.sample(1)
    state_dim = state.shape[1]

    agent_config.update(
        {"state_dim": state_dim, "action_dim": 1,
         "max_action": 0, "min_action": -10},
    )
    agent = get_agent(agent_type, agent_config, hyperparameters)

    if not debug:
        fct = run_info["environment"]["function"]
        teacher = run_info["agent"]["type"]
        state_version = run_info["environment"]["state_version"]
        wandb.init(  # type: ignore
            project="DAC4DL",
            entity="study_project",
            group=wandb_group,
            config=hyperparameters,
            name=f"{teacher}-{fct}-{state_version}",
        )

    logs = {"actor_loss": [], "critic_loss": []}

    for t in range(int(num_train_iter)):
        batch = replay_buffer.sample(batch_size)
        log_dict = agent.train(batch)
        for k, v in log_dict.items():
            logs[k].append(v)

        if not debug:
            wandb.log(log_dict, agent.total_it)

        if val_freq != 0 and (t + 1) % val_freq == 0:
            with torch.random.fork_rng():
                env = get_environment(run_info["environment"])
                eval_runs = num_eval_runs if num_eval_runs is not None else len(run_info["starting_points"])
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
                eval_data.to_csv(results_dir/ str(seed) / f"{t + 1}" / "eval_data.csv")
                with (results_dir/ str(seed) / f"{t + 1}" / "config.json").open("w") as f:
                    json.dump(dict(hyperparameters), f, indent=2)

    save_agent(agent.state_dict(), results_dir, t, seed)

    if not debug:
        wandb.finish()  # type: ignore

    final_evaluations = eval_data.groupby("run").last()
    fbests = final_evaluations["f_cur"]
    return logs, fbests.mean()
