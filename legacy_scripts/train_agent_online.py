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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    num_eval_runs: int,
    start_timesteps: int = 2560,  # 25e3  <-- Find good default value
    use_wandb: bool = False,
) -> tuple[dict, float]:
    if debug:
        num_train_iter = 5
        val_freq = 5

    set_timeout(timeout)
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    results_dir = Path(data_dir, "results", agent_type)
    with Path(data_dir, "run_info.json").open(mode="rb") as f:
        run_info = json.load(f)

    env_type = run_info["environment"]["type"]
    env = get_environment(run_info["environment"].copy())
    num_batches = (
        run_info["num_batches"]
        if env_type == "SGD"
        else run_info["environment"]["num_batches"]
    )

    state = env.reset()[0]
    state_dim = state.shape[0]

    batches_per_epoch = 1
    if env_type == "SGD":
        if env.epoch_mode is False:
            # if SGD env, translates num_batches to num_epochs
            batches_per_epoch = len(env.train_loader)
            print(f"One epoch consists of {batches_per_epoch} batches.")
        else:
            print("Currently running in epoch mode.")
    print(f"One run contains {num_batches} iterations")

    buffer_size = num_train_iter  # * num_batches
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=1,
        buffer_size=buffer_size,
        seed=seed,
    )

    action_dim = 1
    if env_type == "CMAES":
        max_action = 10
        min_action = 0
        use_cmaes = True
    else:
        max_action = 0
        min_action = -10
        use_cmaes = False
    agent_config.update(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "min_action": min_action,
            "action_space": env.action_space.shape,
        },
    )
    agent = get_agent(
        agent_type,
        agent_config,
        tanh_scaling=False,
        hyperparameters=hyperparameters,
        cmaes=use_cmaes,
    )

    if (not debug) and use_wandb:
        state_version = run_info["environment"]["state_version"]
        wandb.init(  # type: ignore
            project="DAC4DL",
            entity="study_project",
            group=wandb_group,
            config=hyperparameters,
            name=f"online_{agent_type}-{state_version}",
            tags=["online", "agent_test", f"{agent_type}"],
        )

    logs: dict = {}
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    min_fbest = np.inf
    print(f"Batch Size: {batch_size}")
    for t in range(int(num_train_iter)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = rng.uniform(min_action, max_action, 1).astype(np.float32)
        else:
            print("First Time")
            action = (
                agent.select_action(state.type(torch.float32))
                + np.random.normal(  # noqa: NPY002
                    0,
                    max_action * 0.1,
                    size=action_dim,
                ).astype(np.float32)
            ).clip(min_action, max_action)

        if env_type == "CMAES":
            action += 1e-10
        else:
            action = torch.from_numpy(action)

        # Perform action
        next_state, reward, done, _, _ = env.step(action.item())

        # Store data in replay buffer
        replay_buffer.add_transition(state, action, next_state, reward, done)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            batch = replay_buffer.sample(batch_size)
            log_dict = agent.train(batch)
            for k, v in log_dict.items():
                if k not in logs:
                    logs[k] = [v]
                else:
                    logs[k].append(v)

            log_dict.update({"episode_reward": episode_reward})

            if (not debug) and use_wandb:
                wandb.log(log_dict, agent.total_it)  # type: ignore

        # if we run out of bounds or reached max optimization iters
        if done or episode_timesteps == num_batches:
            print(
                f"Total T: {t+1}/{int(num_train_iter)} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}",
            )
            # Reset environment
            (state, _), done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if val_freq != 0 and (t + 1) % val_freq == 0:
            if env_type == "CMAES":
                test_agent = test_cma
            elif env_type == "SGD":
                test_agent = test_sgd
            else:
                test_agent = test_toy

            with torch.random.fork_rng():
                val_env = get_environment(run_info["environment"].copy())
                eval_runs = num_eval_runs
                if eval_protocol == "train":
                    eval_data = test_agent(
                        actor=agent.actor,
                        env=val_env,
                        n_runs=eval_runs,
                        starting_points=run_info["starting_points"],
                        n_batches=num_batches,
                        seed=seed,
                    )
                elif eval_protocol == "interpolation":
                    eval_data = test_agent(
                        actor=agent.actor,
                        env=val_env,
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
                            f"Mean at iteration {t+1}: {fbests} - {target_value} at function {fid}",
                        )
                    fbest_mean = final_evaluations["f_cur"].mean()
                elif env_type == "SGD":
                    val_acc = final_evaluations["valid_acc"]
                    val_acc_mean = val_acc.mean()
                    print(
                        f"Mean validation_acc at iteration {t+1}: {val_acc_mean}",
                    )
                    min_fbest = np.min([min_fbest, fbest_mean])
                else:
                    fbests = final_evaluations["f_cur"]
                    fbest_mean = fbests.mean()
                    print(f"Mean at iteration {t+1}: {fbest_mean}")
                    min_fbest = np.min([min_fbest, fbest_mean])

    save_agent(agent.state_dict(), results_dir, t, seed)

    if (not debug) and use_wandb:
        wandb.finish()  # type: ignore

    return logs, min_fbest
