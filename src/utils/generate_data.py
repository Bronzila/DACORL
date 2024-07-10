from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.general import (
    OutOfTimeError,
    get_environment,
    get_teacher,
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
    env_config["seed"] = seed

    if not (save_run_data or save_rep_buffer):
        input("You are not saving any results. Enter a key to continue anyway.")

    # Get types
    environment_type = env_config["type"]
    agent_type = agent_config["type"]

    if results_dir == "":
        results_dir: Path = Path(
            "data",
            environment_type,
            agent_type,
            str(agent_config["id"]),
        )
    else:
        results_dir = Path(
            results_dir,
            environment_type,
            agent_type,
            str(agent_config["id"]),
        )
    if environment_type == "ToySGD":
        results_dir = results_dir / env_config["function"]

    if results_dir.exists():
        print(f"Data already exists: {results_dir}")
        return

    env = get_environment(env_config)
    num_batches = env_config["num_batches"]

    env_config["action_space"] = env.action_space.shape
    state = env.reset()[0]
    state_dim = state.shape[0]

    phase = "batch"
    batches_per_epoch = 1
    if environment_type == "SGD":
        if env.epoch_mode is False:
            # if SGD env, translates num_batches to num_epochs
            batches_per_epoch = len(env.train_loader)
            print(f"One epoch consists of {batches_per_epoch} batches.")
            num_batches *= batches_per_epoch
        else:
            phase = "epoch"
            print("Currently running in epoch mode.")

    buffer_size = num_runs * num_batches
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=1,
        buffer_size=buffer_size,
        seed=seed,
    )

    agent = get_teacher(agent_type, agent_config)

    aggregated_run_data = []
    run_info = {
        "agent": agent_config,
        "environment": env_config,
        "seed": seed,
        "num_runs": num_runs,
        "num_batches": num_batches,
    }
    starting_points = []
    try:
        for run in range(num_runs):
            if save_run_data:
                actions = []
                rewards = []
                states = []
                batch_indeces = []
                run_indeces = []
                if environment_type == "ToySGD":
                    f_curs = []
                    x_curs = []
                elif environment_type == "SGD":
                    train_loss = []
                    valid_loss = []
                    test_loss = []
                elif environment_type == "CMAES":
                    lambdas = []
                    f_curs = []
            if environment_type == "CMAES":
                # Start with instance 0
                env.instance_index = -1
            state, meta_info = env.reset()
            if environment_type == "ToySGD" or "CMAES":
                starting_points.append(meta_info["start"])
            agent.reset()
            if save_run_data:
                rewards.append(np.NaN)
                states.append(state.numpy())
                batch_indeces.append(0)
                run_indeces.append(run)
                if environment_type == "ToySGD":
                    actions.append(math.log10(env.learning_rate))
                    x_curs.append(env.x_cur.tolist())
                    f_curs.append(env.objective_function(env.x_cur).numpy())
                if environment_type == "SGD":
                    actions.append(math.log10(env.learning_rate))
                    train_loss.append(env.train_loss)
                    valid_loss.append(env.validation_loss)
                    test_loss.append(env.test_losses / len(env.test_loader))
                if environment_type == "CMAES":
                    actions.append(env.es.parameters.sigma)
                    lambdas.append(env.es.parameters.lambda_)
                    f_curs.append(env.es.parameters.population.f)

            for batch in range(1, (num_batches + batches_per_epoch)):
                print(
                    f"Starting {phase} {batch}/{num_batches} of run {run}. \
                    Total {batch + run * num_batches}/{num_runs * num_batches}",
                )

                if agent_type == "csa":
                    action = agent.act(env)
                else:
                    action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add_transition(
                    state,
                    action,
                    next_state,
                    reward,
                    done,
                )
                if save_run_data:
                    actions.append(action)
                    rewards.append(reward.numpy())
                    states.append(state.numpy())
                    batch_indeces.append(batch)
                    run_indeces.append(run)
                    if environment_type == "ToySGD":
                        x_curs.append(env.x_cur.tolist())
                        f_curs.append(env.objective_function(env.x_cur).numpy())
                    if environment_type == "SGD":
                        train_loss.append(env.train_loss)
                        valid_loss.append(
                            env.validation_loss,
                        )
                        test_loss.append(env.test_losses / len(env.test_loader))
                    if environment_type == "CMAES":
                        lambdas.append(env.es.parameters.lambda_)
                        f_curs.append(env.es.parameters.population.f)

                       
                state = next_state
                if done:
                    break

            if save_run_data:
                data = {
                    "action": actions,
                    "reward": rewards,
                    "state": states,
                    "batch": batch_indeces,
                    "run": run_indeces,
                }
                if environment_type == "ToySGD":
                    data.update(
                        {
                            "f_cur": f_curs,
                            "x_cur": x_curs,
                        },
                    )
                if environment_type == "SGD":
                    data.update(
                        {
                            "train_loss": train_loss,
                            "valid_loss": valid_loss,
                            "test_loss": test_loss,
                        },
                    )
                if environment_type == "CMAES":
                    data.update(
                        {
                            "lambdas": lambdas,
                            "f_curs": f_curs,
                        },
                    )
                run_data = pd.DataFrame(data)
                aggregated_run_data.append(run_data)
    except OutOfTimeError:
        save_data(
            save_run_data,
            pd.concat(aggregated_run_data),
            save_rep_buffer,
            replay_buffer,
            results_dir,
            run_info,
            starting_points,
        )
        print("Saved checkpoint, because run was about to end")

    save_data(
        save_run_data,
        pd.concat(aggregated_run_data),
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
