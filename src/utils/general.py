from __future__ import annotations

import json
import os
import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from CORL.algorithms.offline import td3_bc
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
            activation=nn.ReLU,
            hidden_dim=hyperparameters.get("hidden_dim", 64),
            hidden_layers=hyperparameters.get("hidden_layers", 1),
            dropout_rate=hyperparameters.get("dropout_rate", 0.2),
        ).to(device)
        print(f"hidden_dim: {hyperparameters.get('hidden_dim', 64)}")
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
        print("Agent Config:")
        print(kwargs)
        return td3_bc.TD3_BC(**kwargs)

    raise NotImplementedError(
        f"No agent with type {agent_type} implemented.",
    )


def get_environment(env_config: dict) -> Any:
    from dacbench.benchmarks import SGDBenchmark, ToySGD2DBenchmark
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
    elif env_config["type"] == "SGD":
        bench = SGDBenchmark(config=env_config)
        return bench.get_benchmark()
    else:
        raise NotImplementedError(
            f"No environment of type {env_config['type']} found.",
        )



def save_agent(state_dicts: dict, results_dir: Path, iteration: int, seed: int=0) -> None:
    filename = results_dir / str(seed) / f"{iteration + 1}"
    if not filename.exists():
        filename.mkdir(parents=True)

    for key, s in state_dicts.items():
        torch.save(s, filename / f"agent_{key}")


def load_agent(agent_type: str, agent_config: dict, agent_path: Path) -> Any:
    hp_conf_path = agent_path / "config.json"
    with hp_conf_path.open("r") as f:
        hyperparameters = json.load(f)

    print("Loaded agent with following hyperparameters:")
    print(hyperparameters)

    agent = get_agent(agent_type, agent_config, hyperparameters)
    state_dict = agent.state_dict()
    new_state_dict = {}
    for key, _ in state_dict.items():
        s = torch.load(agent_path / f"agent_{key}")
        new_state_dict.update({key: s})

    agent.load_state_dict(new_state_dict)
    return agent
