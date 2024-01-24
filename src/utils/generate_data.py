from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.general import (
    OutOfTimeError,
    get_agent,
    get_environment,
    set_seeds,
    set_timeout,
)
from src.utils.replay_buffer import ReplayBuffer


def save_data(
    save_run_data,
    aggregated_run_data,
    save_rep_buffer,
    replay_buffer,
    results_dir,
    run_info,
    starting_points,
) -> None:
    run_info["starting_points"] = starting_points
    save_path = Path(results_dir, "rep_buffer")
    if save_rep_buffer:
        replay_buffer.save(save_path)
        with Path(results_dir, "run_info.json").open(mode="w") as f:
            json.dump(run_info, f, indent=4)

    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    if save_run_data:
        aggregated_run_data.to_csv(
            Path(results_dir, "aggregated_run_data.csv"),
        )


def generate_dataset(
    agent_config: dict,
    env_config: dict,
    num_runs: int,
    seed: int,
    results_dir: str,
    timeout: int,
    save_run_data: bool,
    save_rep_buffer: bool,
) -> None:
    set_timeout(timeout)
    set_seeds(seed)

    if not (save_run_data or save_rep_buffer):
        input("You are not saving any results. Enter a key to continue anyway.")

    # Get types
    environment_type = env_config["type"]
    agent_type = agent_config["type"]

    if results_dir == "":
        results_dir = Path("data", agent_type, environment_type)
    else:
        results_dir = Path(results_dir, agent_type, environment_type)

    num_batches = env_config["num_batches"]
    env = get_environment(env_config)
    state = env.reset()[0]
    state_dim = state.shape[0]
    buffer_size = num_runs * num_batches
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=1,
        buffer_size=buffer_size,
    )

    agent = get_agent(agent_type, agent_config, "cpu")

    aggregated_run_data = None
    run_info = {
        "agent": agent_config,
        "environment": env_config,
        "seed": seed,
        "num_runs": num_runs,
        "num_batches": num_batches,
    }

    try:
        for run in range(num_runs):
            if save_run_data:
                actions = []
                rewards = []
                f_curs = []
                x_curs = []
                states = []
                batch_indeces = []
                run_indeces = []
                starting_points = []
            state, meta_info = env.reset()
            starting_points.append(meta_info["start"])
            agent.reset()
            if save_run_data:
                actions.append(np.NaN)
                rewards.append(np.NaN)
                x_curs.append(env.x_cur.tolist())
                f_curs.append(env.objective_function(env.x_cur).numpy())
                states.append(state.numpy())
                batch_indeces.append(0)
                run_indeces.append(run)

            for batch in range(1, num_batches):
                print(
                    f"Starting batch {batch}/{num_batches} of run {run}. \
                    Total {batch + run * num_batches}/{num_runs * num_batches}",
                )

                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add_transition(
                    state,
                    action,
                    next_state,
                    reward,
                    truncated,
                )
                state = next_state
                if save_run_data:
                    actions.append(action)
                    rewards.append(reward.numpy())
                    x_curs.append(env.x_cur.tolist())
                    f_curs.append(env.objective_function(env.x_cur).numpy())
                    states.append(state.numpy())
                    batch_indeces.append(batch)
                    run_indeces.append(run)

            if save_run_data:
                run_data = pd.DataFrame(
                    {
                        "action": actions,
                        "reward": rewards,
                        "f_cur": f_curs,
                        "x_cur": x_curs,
                        "state": states,
                        "batch": batch_indeces,
                        "run": run_indeces,
                    },
                )
                if aggregated_run_data is None:
                    aggregated_run_data = run_data
                else:
                    aggregated_run_data = aggregated_run_data.append(
                        run_data,
                        ignore_index=True,
                    )
    except OutOfTimeError:
        save_data(
            save_run_data,
            aggregated_run_data,
            save_rep_buffer,
            replay_buffer,
            results_dir,
            run_info,
            starting_points,
        )
        print("Saved checkpoint, because run was about to end")

    save_data(
        save_run_data,
        aggregated_run_data,
        save_rep_buffer,
        replay_buffer,
        results_dir,
        run_info,
        starting_points,
    )

    if save_rep_buffer or save_run_data:
        msg = "Saved "
        msg += "rep_buffer " if save_rep_buffer else ""
        msg += "run_data " if save_run_data else ""
        print(f"{msg}to {results_dir}")
