from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from typing import Any

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
    hyperparameters: dict[str, Any] = {},
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
            get_activation(hyperparameters["activation"]),
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic_1 = td3_bc.Critic(
            state_dim,
            action_dim,
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        critic_2 = td3_bc.Critic(
            state_dim,
            action_dim,
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=hyperparameters["lr_critic"],
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
            "device": config.device,
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


def save_agent(state_dicts: dict, results_dir: Path, iteration: int) -> None:
    filename = results_dir / f"{iteration + 1}"
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
            combined_run_info["starting_points"].extend(run_info["starting_points"])

        df = pd.read_csv(run_data_path)
        df["run"] += idx * run_info["num_runs"]
        combined_run_data.append(df)
    return combined_buffer, combined_run_info, pd.concat(combined_run_data, ignore_index=True)
