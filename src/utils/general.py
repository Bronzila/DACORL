from __future__ import annotations

import json
import os
import random
import signal
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from CORL.algorithms.offline import td3_bc
from dacbench.benchmarks import ToySGD2DBenchmark
from torch import nn

from src.agents import (
    ConstantAgent,
    ExponentialDecayAgent,
    SGDRAgent,
    StepDecayAgent,
)
from src.utils.replay_buffer import ReplayBuffer


# Time out related class and function
class OutOfTimeError(Exception):
    pass


def timeouthandler(signum: Any, frame: Any) -> None:
    raise OutOfTimeError


def set_timeout(timeout: int) -> None:
    if timeout > 0:
        # conversion from hours to seconds
        timeout = timeout * 60 * 60
        signal.signal(signal.SIGALRM, timeouthandler)
        signal.alarm(timeout)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_agent(
    agent_type: str,
    agent_config: dict[str, Any],
    hyperparameters: dict[str, Any] | None = None,
    device: str = "cpu",
) -> Any:
    if agent_type == "step_decay":
        return StepDecayAgent(**agent_config["params"])
    if agent_type == "exponential_decay":
        return ExponentialDecayAgent(**agent_config["params"])
    if agent_type == "sgdr":
        return SGDRAgent(**agent_config["params"])
    if agent_type == "constant":
        return ConstantAgent(**agent_config["params"])
    if agent_type == "td3_bc":
        if hyperparameters is None:
            print("No hyperparameters specified, resorting to defaults.")
            hyperparameters = {}
        else:
            hyperparameters = dict(hyperparameters)

        config = td3_bc.TrainConfig

        state_dim = agent_config["state_dim"]
        action_dim = agent_config["action_dim"]
        max_action = agent_config["max_action"]
        min_action = agent_config["min_action"]

        alpha = hyperparameters.get("alpha", config.alpha)

        actor = td3_bc.Actor(
            state_dim,
            action_dim,
            max_action,
            get_activation(hyperparameters.get("activation", "ReLU")),
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters.get("lr_actor", 3e-4),
        )

        critic_1 = td3_bc.Critic(
            state_dim,
            action_dim,
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters.get("lr_critic", 3e-4),
        )

        critic_2 = td3_bc.Critic(
            state_dim,
            action_dim,
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=hyperparameters.get("lr_critic", 3e-4),
        )

        kwargs = {
            "max_action": max_action,
            "min_action": min_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic_1": critic_1,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2": critic_2,
            "critic_2_optimizer": critic_2_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "device": device,
            # TD3
            "policy_noise": config.policy_noise,
            "noise_clip": config.noise_clip,
            "policy_freq": config.policy_freq,
            # TD3 + BC
            "alpha": alpha,
        }
        return td3_bc.TD3_BC(**kwargs)

    raise NotImplementedError(
        f"No agent with type {agent_type} implemented.",
    )


def get_environment(env_config: dict) -> Any:
    if env_config["type"] == "ToySGD":
        # setup benchmark
        bench = ToySGD2DBenchmark()
        bench.config.cutoff = env_config["num_batches"]
        bench.config.low = env_config["low"]
        bench.config.high = env_config["high"]
        bench.config.function = env_config["function"]
        bench.config.initial_learning_rate = env_config["initial_learning_rate"]
        bench.config.state_version = env_config["state_version"]
        bench.config.reward_version = env_config["reward_version"]
        bench.config.boundary_termination = env_config["boundary_termination"]
        bench.config.seed = env_config["seed"]
        return bench.get_environment()
    else:
        raise NotImplementedError(
            f"No environment of type {env_config['type']} found.",
        )


def get_activation(activation: str) -> nn.Module:
    if activation == "ReLU":
        return nn.ReLU
    if activation == "LeakyReLU":
        return nn.LeakyReLU
    if activation == "Tanh":
        return nn.Tanh
    return None


def save_agent(state_dicts: dict, results_dir: Path, iteration: int, seed: int=0) -> None:
    filename = results_dir / str(seed) / f"{iteration + 1}"
    if not filename.exists():
        filename.mkdir(parents=True)

    for key, s in state_dicts.items():
        torch.save(s, filename / f"agent_{key}")


def load_agent(agent_type: str, agent_config: dict, agent_path: Path) -> Any:
    agent = get_agent(agent_type, agent_config)
    state_dict = agent.state_dict()
    new_state_dict = {}
    for key, _ in state_dict.items():
        s = torch.load(agent_path / f"agent_{key}")
        new_state_dict.update({key: s})

    agent.load_state_dict(new_state_dict)
    return agent

def get_homogeneous_agent_paths(root_dir: str, function: str):
    root_path = Path(root_dir)
    agent_dirs = [entry.name for entry in root_path.iterdir() if entry.is_dir() and entry.name != "combined"]
    paths = []
    for dirname in agent_dirs:
        agent_path = Path(root_dir, dirname, function)
        paths.append(agent_path)
    return paths

def combine_runs(agent_paths):
    print(agent_paths)
    combined_buffer = None
    combined_run_info = None
    combined_run_data = []
    for idx, root_path in enumerate(agent_paths):
        replay_path = Path(root_path, "rep_buffer")
        run_info_path = Path(root_path, "run_info.json")
        run_data_path = Path(root_path, "aggregated_run_data.csv")

        with run_info_path.open(mode="rb") as f:
            run_info = json.load(f)
        temp_buffer = ReplayBuffer.load(replay_path)

        if combined_buffer == None and combined_run_info == None:
            combined_buffer = temp_buffer
            combined_run_info = {"environment": run_info["environment"],
                                 "starting_points": run_info["starting_points"],
                                 "seed": run_info["seed"],
                                 "num_runs": run_info["num_runs"],
                                 "num_batches": run_info["num_batches"],
                                 "agent": {"type": run_info["agent"]["type"]}}
        else:
            combined_buffer.merge(temp_buffer)
            # combined_run_info["starting_points"].extend(run_info["starting_points"])

        df = pd.read_csv(run_data_path)
        df["run"] += idx * run_info["num_runs"]
        combined_run_data.append(df)
    return combined_buffer, combined_run_info, pd.concat(combined_run_data, ignore_index=True)

def combine_run_data(data_paths, num_runs=100):
    combined_run_data = []
    for idx, root_path in enumerate(data_paths):
        run_data_path = Path(root_path)
        df = pd.read_csv(run_data_path)
        df["run"] += idx * num_runs
        combined_run_data.append(df)
    return pd.concat(combined_run_data, ignore_index=True)

def find_lowest_values(df, column_name, n=10):
    final_evaluations = df.groupby("run").last()

    # Sort the DataFrame by the specified column in ascending order
    sorted_df = final_evaluations.sort_values(by=column_name)

    # Get the lowest n values from the sorted DataFrame
    return sorted_df.head(n)

def calc_mean_and_std_dev(df):
    final_evaluations = df.groupby("run").last()

    fbests = final_evaluations["f_cur"]
    return fbests.mean(), fbests.std()

def calculate_single_seed_statistics(calc_mean=True, calc_lowest=True, n_lowest=1, path=None,
                                    results=True, verbose=False):
    paths = []
    if results:
        for folder_path, _, _ in os.walk(path):
            paths.extend(Path(folder_path).glob("*/eval_data.csv"))
    else:
        paths.append(path)
    # Load data
    min_mean = np.inf
    min_std = np.inf
    min_iqm = np.inf
    min_iqm_std = np.inf
    min_path = ""
    lowest_vals_of_min_mean = []
    for path in paths:
        incumbent_changed = False
        df = pd.read_csv(path)
        if verbose:
            print(f"Calculating for path {path}")

        if calc_mean:
            mean, std = calc_mean_and_std_dev(df)
            mean = float(f"{mean:.3e}")
            std = float(f"{std:.3e}")
            if mean < min_mean or mean == min_mean and std < min_std:
                incumbent_changed = True
                min_mean = mean
                min_std = std
                min_path = path
                min_iqm, min_iqm_std = compute_IQM(df)
            if verbose:
                print(f"Mean +- Std {mean:.3e} ± {std:.3e}")
        if calc_lowest:
            lowest_vals = find_lowest_values(df, "f_cur", n_lowest)
            if incumbent_changed:
                lowest_vals_of_min_mean = lowest_vals["f_cur"]
            if verbose:
                print("Lowest values:")
                print(lowest_vals["f_cur"])
    return min_mean, min_std, lowest_vals_of_min_mean, min_iqm, min_iqm_std, min_path

def calculate_multi_seed_statistics(calc_mean=True, calc_lowest=True, n_lowest=1, path=None,
                                    results=True, verbose=False):
    # TODO here we currently assume, that we only have one training folder and eval file in results/td3_bc/<seed>/
    paths = []
    if results:
        for folder_path, _, _ in os.walk(path):
            paths.extend(Path(folder_path).glob("*/eval_data.csv"))
    else:
        paths.append(path)

    combined_data = combine_run_data(paths)
    if calc_mean:
            mean, std = calc_mean_and_std_dev(combined_data)
            mean = float(f"{mean:.3e}")
            std = float(f"{std:.3e}")
            iqm, iqm_std = compute_IQM(combined_data)
    if calc_lowest:
        lowest_vals = find_lowest_values(combined_data, "f_cur", n_lowest)["f_cur"]
    return mean, std, lowest_vals, iqm, iqm_std, paths[0] # path doesnt really matter here

def calculate_statistics(calc_mean=True, calc_lowest=True, n_lowest=1, path=None,
                         results=True, verbose=False, multi_seed=False):
    if multi_seed:
        return calculate_multi_seed_statistics(calc_mean, calc_lowest,
                                               n_lowest, path, results, verbose)
    else:
        return calculate_single_seed_statistics(calc_mean, calc_lowest, n_lowest,
                                                path, results, verbose)

def compute_IQM(df):
    final_evaluations = df.groupby("run").last()
    df_sorted = final_evaluations.sort_values(by="f_cur")

    # Calculate the number of rows representing 25% of the DataFrame
    num_rows = len(df_sorted)
    num_to_remove = int(0.25 * num_rows)

    # Remove the upper and lower 25% of the DataFrame
    df_trimmed = df_sorted[num_to_remove:-num_to_remove]
    fbests = df_trimmed["f_cur"]
    return fbests.mean(), fbests.std()

def compute_prob_outperformance(df_teacher, df_agent):
    final_evaluations_teacher = df_teacher.groupby("run").last()["f_cur"]
    final_evaluations_agent = df_agent.groupby("run").last()["f_cur"]

    assert len(final_evaluations_agent) == len(final_evaluations_teacher)

    p = 0
    for i in range(len(final_evaluations_agent)):
        if final_evaluations_agent[i] < final_evaluations_teacher[i]:
            p += 1

    return p / len(final_evaluations_agent)