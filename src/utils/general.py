from __future__ import annotations

import random
import signal
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from CORL.algorithms.offline import td3_bc
from dacbench.benchmarks import ToySGDBenchmark

from src.agents.step_decay import StepDecayAgent

if TYPE_CHECKING:
    from dacbench import AbstractBenchmark


# Time out related class and function
class OutOfTimeError(Exception):
    pass


def timeouthandler(signum, frame) -> None:
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
    device: str,
) -> Any:
    if agent_type == "step_decay":
        return StepDecayAgent(**agent_config)
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


def get_environment(environment_type: str) -> AbstractBenchmark:
    if environment_type == "ToySGD":
        # setup benchmark
        bench = ToySGDBenchmark()
        return bench.get_environment()

    raise NotImplementedError(
        f"No environment of type {environment_type} found.",
    )
