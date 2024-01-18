from __future__ import annotations

import json
from pathlib import Path

import wandb

from src.utils.replay_buffer import ReplayBuffer
from utils.general import get_agent, set_seeds, set_timeout


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
        {"state_dim": state_dim, "action_dim": 1, "max_action": 1},
    )
    agent = get_agent(agent_type, agent_config)

    config = {
        "agent_config": agent_config,
        "run_info": run_info,
    }
    wandb.init(
        project="DEDAC",
        entity="study_project",
        group=wandb_group,
        config=config,
    )

    for t in range(num_train_iter):
        log_dict = {}
        batch = replay_buffer.sample(batch_size)
        log_dict = agent.train(batch)

        wandb.log(log_dict, agent.total_it)

        if val_freq != 0 and (t + 1) % val_freq == 0:
            # Save agent early to enable continuation of pipeline
            path_to_save = results_dir / f"{t+1}"
            if not path_to_save.exists():
                path_to_save.mkdir(parents=True)
            agent.save(path_to_save / "agent")

    wandb.finish()
