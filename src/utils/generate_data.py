from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.experiment_data import (
    CMAESExperimentData,
    ExperimentData,
    SGDExperimentData,
    ToySGDExperimentData,
)
from src.utils.general import (
    OutOfTimeError,
    get_environment,
    get_teacher,
    set_seeds,
    set_timeout,
)
from src.utils.replay_buffer import ReplayBuffer


def save_data(
    save_run_data: bool,
    aggregated_run_data: pd.DataFrame,
    save_rep_buffer: bool,
    replay_buffer: ReplayBuffer,
    results_dir: Path,
    run_info: dict,
    starting_points: list,
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
) -> tuple[dict, ReplayBuffer, dict]:
    if not checkpoint_dir.exists():
        raise RuntimeError(
            f"The specified checkpoint does not exist: {checkpoint_dir}",
        )

    checkpoint_run_data = pd.read_csv(
        checkpoint_dir / "aggregated_run_data.csv",
    ).to_dict()
    rb = ReplayBuffer.load(checkpoint_dir / "rep_buffer")
    with (checkpoint_dir / "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    return checkpoint_run_data, rb, run_info


def generate_dataset(
    agent_config: dict,
    env_config: dict,
    num_runs: int,
    seed: int,
    results_dir: Path,
    timeout: int,
    checkpointing_freq: int,
    checkpoint: int,
    save_run_data: bool,
    save_rep_buffer: bool,
    check_if_exists: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    set_timeout(timeout)
    set_seeds(seed)
    env_config["seed"] = seed

    if not (save_run_data or save_rep_buffer):
        input("You are not saving any results. Enter a key to continue anyway.")

    print(f"Environment: {env_config}")
    print(f"Teacher: {agent_config}", flush=True)

    # Get types
    environment_type = env_config["type"]
    agent_type = agent_config["type"]

    if results_dir == Path():
        results_dir = Path(
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
        raise RuntimeError(f"Data already exists: {results_dir}")

    env = get_environment(env_config.copy())
    env.reset()
    phase = "batch"

    phase = "batch"
    batches_per_epoch = 1
    if environment_type == "SGD":
        print(f"Generating data for {env_config['dataset_name']}")
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

    agent = get_teacher(agent_config)

    exp_data: ExperimentData
    if environment_type == "ToySGD":
        exp_data = ToySGDExperimentData()
    elif environment_type == "SGD":
        exp_data = SGDExperimentData()
    elif environment_type == "CMAES":
        exp_data = CMAESExperimentData()
    else:
        raise NotImplementedError(
            f"No experiment data class for experiment {environment_type}",
        )
    run_info = {
        "agent": agent_config,
        "environment": env_config,
        "seed": seed,
        "num_runs": num_runs,
        "num_batches": num_batches,
    }

    start_run = 0
    starting_points = []

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

        exp_data.data = checkpoint_data

    if environment_type == "CMAES":
        # Start with instance 0
        env.instance_index = -1

    try:
        for run in range(start_run, num_runs):
            state, meta_info = env.reset()
            if environment_type == ("ToySGD"):
                starting_points.append(meta_info["start"])
            agent.reset()
            if save_run_data:
                exp_data.init_data(run, state, env)

            start = time.time()
            for batch in range(1, num_batches + 1):
                if verbose:
                    print(
                        f"Starting {phase} {batch}/{num_batches} of run {run}. \
                        Total {batch + run * num_batches}/{num_runs * num_batches}",
                    )

                if agent_type in ("csa", "cmaes_constant"):
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
                    exp_data.add(
                        {
                            "state": state.numpy(),
                            "action": action,
                            "reward": reward.numpy(),
                            "batch_idx": batch,
                            "run_idx": run,
                            "env": env,
                        },
                    )

                state = next_state
                if done:
                    break

            end = time.time()
            print(f"Run {run} took {end - start} sec.")
            

            if checkpointing_freq != 0 and (run + 1) % checkpointing_freq == 0:
                checkpoint_dir = Path(results_dir, "checkpoints", str(run))
                if not checkpoint_dir.exists():
                    checkpoint_dir.mkdir(parents=True)

                if environment_type == "ToySGD":
                    raise UserWarning(
                        "Are you sure you want to checkpoint ToySGD?",
                    )
                if environment_type == "SGD":
                    run_info.update(
                        {
                            "checkpoint_info": {
                                "run": run,
                                "rng": env.rng.bit_generator.state,
                                "instance_index": env.instance_index,
                            },
                        },
                    )
                if environment_type == "CMAES":
                    raise UserWarning(
                        "Are you sure you want to checkpoint ToySGD?",
                    )

                save_data(
                    save_run_data,
                    exp_data.concatenate_data(),
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
            exp_data.concatenate_data(),
            save_rep_buffer,
            replay_buffer,
            results_dir,
            run_info,
            starting_points,
        )
        print("Saved checkpoint, because run was about to end")

    save_data(
        save_run_data,
        exp_data.concatenate_data(),
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

    return exp_data.concatenate_data()
