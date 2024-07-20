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
    checkpoint: bool = False,
) -> None:
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    run_info["starting_points"] = starting_points
    save_path = Path(results_dir, "rep_buffer")
    if save_rep_buffer:
        if checkpoint:
            replay_buffer.checkpoint(save_path)
        else:
            replay_buffer.save(save_path)
        with Path(results_dir, "run_info.json").open(mode="w") as f:
            json.dump(run_info, f, indent=4)

    if save_run_data:
        aggregated_run_data.to_csv(
            Path(results_dir, "aggregated_run_data.csv"),
        )


def load_checkpoint(
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, ReplayBuffer, dict]:
    if not checkpoint_dir.exists():
        raise RuntimeError(
            f"The specified checkpoint does not exist: {checkpoint_dir}",
        )

    checkpoint_run_data = pd.read_csv(
        checkpoint_dir / "aggregated_run_data.csv",
    )
    rb = ReplayBuffer.load(checkpoint_dir / "rep_buffer")
    with (checkpoint_dir / "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    return checkpoint_run_data, rb, run_info


def generate_dataset(
    agent_config: dict,
    env_config: dict,
    num_runs: int,
    seed: int,
    results_dir: str,
    timeout: int,
    checkpointing_freq: int,
    checkpoint: int,
    save_run_data: bool,
    save_rep_buffer: bool,
    check_if_exists: bool = True,
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
        num_batches = env_config["num_batches"]

    if environment_type == "CMAES":
        num_batches = env_config["num_batches"]

    if check_if_exists and (results_dir.exists() and checkpoint == 0):
        print(f"Data already exists: {results_dir}")
        return None

    env = get_environment(env_config.copy())
    env.reset()
    phase = "batch"

    phase = "batch"
    batches_per_epoch = 1
    if environment_type == "SGD":
        if env.epoch_mode is False:
            num_epochs = env_config["num_epochs"]
            # if SGD env, translates num_batches to num_epochs
            batches_per_epoch = len(env.train_loader)
            print(f"One epoch consists of {batches_per_epoch} batches.")
            num_batches = num_epochs * batches_per_epoch
            env_config["cutoff"] = num_batches
        else:
            phase = "epoch"
            print("Currently running in epoch mode.")

    env = get_environment(env_config.copy())
    state = env.reset()[0]
    env.seed(
        seed,
    )  # Reseed environment here to allow for proper starting point generation
    state_dim = state.shape[0]

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

    start_run = 0

    if checkpoint != 0:
        checkpoint_dir = Path(results_dir, "checkpoints", str(checkpoint))
        checkpoint_data, replay_buffer, checkpoint_run_info = load_checkpoint(
            checkpoint_dir,
        )

        starting_points = checkpoint_run_info["starting_points"]
        start_run = checkpoint_run_info["checkpoint_info"]["run"] + 1
        if environment_type == "SGD":
            # assuming we use PCG64 as rng
            env.rng.bit_generator.state = checkpoint_run_info[
                "checkpoint_info"
            ]["rng"]
            env.instance_index = checkpoint_run_info["checkpoint_info"][
                "instance_index"
            ]

        del checkpoint_run_info["starting_points"]
        del checkpoint_run_info["checkpoint_info"]

        assert run_info == checkpoint_run_info

        aggregated_run_data.append(checkpoint_data)
    if environment_type == "CMAES":
        # Start with instance 0
        env.instance_index = -1

    try:
        for run in range(start_run, num_runs):
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
                    train_acc = []
                    valid_acc = []
                    test_loss = []
                    test_acc = []
                elif environment_type == "CMAES":
                    lambdas = []
                    f_curs = []
                    target_value = []
            state, meta_info = env.reset()
            if environment_type == ("ToySGD"):
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
                    train_acc.append(env.train_accuracy)
                    valid_acc.append(env.validation_accuracy)
                    test_loss.append(env.test_loss)
                    test_acc.append(env.test_accuracy)
                if environment_type == "CMAES":
                    actions.append(env.es.parameters.sigma)
                    lambdas.append(env.es.parameters.lambda_)
                    f_curs.append(env.es.parameters.population.f)
                    target_value.append(env.target)

            for batch in range(1, num_batches + 1):
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
                        train_acc.append(env.train_accuracy)
                        valid_acc.append(env.validation_accuracy)
                        test_loss.append(env.test_loss)
                        test_acc.append(env.test_accuracy)
                    if environment_type == "CMAES":
                        lambdas.append(env.es.parameters.lambda_)
                        f_curs.append(env.es.parameters.population.f)
                        target_value.append(env.target)

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
                            "train_acc": train_acc,
                            "valid_acc": valid_acc,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                        },
                    )
                if environment_type == "CMAES":
                    data.update(
                        {
                            "lambdas": lambdas,
                            "f_curs": f_curs,
                            "target_value": target_value,
                        },
                    )
                run_data = pd.DataFrame(data)
                aggregated_run_data.append(run_data)

            if checkpointing_freq != 0 and (run + 1) % checkpointing_freq == 0:
                # if run == 0:

                checkpoint_dir = Path(results_dir, "checkpoints", str(run))
                if not checkpoint_dir.exists():
                    checkpoint_dir.mkdir(parents=True)

                if environment_type == "ToySGD":
                    raise UserWarning(
                        "Are you sure you want to checkpoint ToySGD?",
                    )
                elif environment_type == "SGD":
                    run_info.update(
                        {
                            "checkpoint_info": {
                                "run": run,
                                "rng": env.rng.bit_generator.state,
                                "instance_index": env.instance_index,
                            },
                        },
                    )
                elif environment_type == "CMAES":
                    raise UserWarning(
                        "Are you sure you want to checkpoint ToySGD?",
                    )

                save_data(
                    save_run_data,
                    pd.concat(aggregated_run_data),
                    save_rep_buffer,
                    replay_buffer,
                    checkpoint_dir,
                    run_info,
                    starting_points,
                    checkpoint=True,
                )

                print(f"Saved checkpoint {checkpoint_dir}")
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

    return pd.concat(aggregated_run_data)
