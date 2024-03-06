from __future__ import annotations

import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch
from CORL.algorithms.offline import td3_bc
from dacbench.benchmarks import ToySGD2DBenchmark

from src.agents import ExponentialDecayAgent, StepDecayAgent


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
    device: str = "cpu",
) -> Any:
    if agent_type == "step_decay":
        return StepDecayAgent(**agent_config["params"])
    if agent_type == "exponential_decay":
        return ExponentialDecayAgent(**agent_config["params"])
    if agent_type == "td3_bc":
        config = td3_bc.TrainConfig

        state_dim = agent_config["state_dim"]
        action_dim = agent_config["action_dim"]
        max_action = agent_config["max_action"]

        actor = td3_bc.Actor(state_dim, action_dim, max_action).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

        critic_1 = td3_bc.Critic(state_dim, action_dim).to(device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)

        critic_2 = td3_bc.Critic(state_dim, action_dim).to(device)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

        kwargs = {
            "max_action": max_action,
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
            "policy_noise": config.policy_noise * max_action,
            "noise_clip": config.noise_clip * max_action,
            "policy_freq": config.policy_freq,
            # TD3 + BC
            "alpha": config.alpha,
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
        return bench.get_environment()
    else:
        raise NotImplementedError(
            f"No environment of type {env_config['type']} found.",
        )


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