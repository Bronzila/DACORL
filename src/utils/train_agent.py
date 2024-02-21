from __future__ import annotations

import json
from pathlib import Path

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


def train_agent(
    data_dir: str,
    agent_type: str,
    agent_config: dict,
    num_train_iter: int,
    num_eval_runs: int,
    batch_size: int,
    val_freq: int,
    seed: int,
    wandb_group: str,
    timeout: int,
    debug: bool,
) -> None:
    if debug:
        num_train_iter = 5
        val_freq = 5

    set_timeout(timeout)
    set_seeds(seed)

    results_dir = Path(data_dir, "results", agent_type)
    with Path(data_dir, "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    replay_buffer = ReplayBuffer.load(Path(data_dir, "rep_buffer"))
    state, _, _, _, _ = replay_buffer.sample(1)
    state_dim = state.shape[1]

    agent_config.update(
        {"state_dim": state_dim, "action_dim": 1, "max_action": 1},
    )
    agent = get_agent(agent_type, agent_config)

    config = {
        "agent_config": agent_config,
        "run_info": run_info,
    }

    if not debug:
        wandb.init(  # type: ignore
            project="DAC4DL",
            entity="study_project",
            group=wandb_group,
            config=config,
        )

    for t in range(num_train_iter):
        log_dict = {}
        batch = replay_buffer.sample(batch_size)
        log_dict = agent.train(batch)

        if not debug:
            wandb.log(log_dict, agent.total_it)  # type: ignore

        if val_freq != 0 and (t + 1) % val_freq == 0:
            env = get_environment(run_info["environment"])
            eval_data = test_agent(
                actor=agent.actor,
                env=env,
                n_runs=num_eval_runs,
                n_batches=run_info["environment"]["num_batches"],
                seed=run_info["seed"],
            )

            # Save agent early to enable continuation of pipeline
            save_agent(agent.state_dict(), results_dir, t)
            eval_data.to_csv(results_dir / f"{t + 1}" / "eval_data.csv")

    if not debug:
        wandb.finish()  # type: ignore
