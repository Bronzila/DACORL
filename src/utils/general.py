from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from CORL.algorithms.offline import (
    any_percent_bc as bc,
    awac,
    cql,
    edac,
    lb_sac,
    sac_n,
    td3_bc,
)
from dacbench.benchmarks import ToySGD2DBenchmark
from torch import nn

from src.agents import (
    ConstantAgent,
    ExponentialDecayAgent,
    SGDRAgent,
    StepDecayAgent,
)
from src.utils.agent_components import (
    ConfigurableActor,
    ConfigurableCritic,
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


def get_teacher(
    teacher_type: str,
    teacher_config: dict[str, Any],
) -> Any:
    if teacher_type == "step_decay":
        return StepDecayAgent(**teacher_config["params"])
    if teacher_type == "exponential_decay":
        return ExponentialDecayAgent(**teacher_config["params"])
    if teacher_type == "sgdr":
        return SGDRAgent(**teacher_config["params"])
    if teacher_type == "constant":
        return ConstantAgent()

    raise NotImplementedError(
        f"No agent with type {teacher_type} implemented.",
    )


def get_agent(
    agent_type: str,
    agent_config: dict[str, Any],
    hyperparameters: dict[str, Any] = {},
    device: str = "cpu",
) -> Any:
    state_dim = agent_config["state_dim"]
    action_dim = agent_config["action_dim"]
    max_action = agent_config["max_action"]
    min_action = agent_config["min_action"]

    if agent_type == "td3_bc":
        config = td3_bc.TrainConfig

        actor = ConfigurableActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            max_action=max_action,
            hidden_layers=hyperparameters["hidden_layers_actor"],
            activation=get_activation(hyperparameters["activation"]),
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic_1 = ConfigurableCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hyperparameters["hidden_layers_critic"],
            hidden_dim=hyperparameters["hidden_dim"],
            activation=get_activation(hyperparameters["activation"]),
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        critic_2 = ConfigurableCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hyperparameters["hidden_layers_critic"],
            hidden_dim=hyperparameters["hidden_dim"],
            activation=get_activation(hyperparameters["activation"]),
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
            "discount": hyperparameters["discount_factor"],
            "tau": hyperparameters["target_update_rate"],
            "device": device,
            # TD3
            "policy_noise": config.policy_noise * max_action,
            "noise_clip": config.noise_clip * max_action,
            "policy_freq": config.policy_freq,
            # TD3 + BC
            "alpha": config.alpha,
        }
        return td3_bc.TD3_BC(**kwargs)

    if agent_type == "bc":
        config = bc.TrainConfig

        actor = bc.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            max_action=max_action,
            hidden_layers=hyperparameters["hidden_layers_actor"],
            activation=get_activation(hyperparameters["activation"]),
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "discount": hyperparameters["discount_factor"],
            "device": device,
        }
        return bc.BC(**kwargs)

    if agent_type == "cql":
        config = cql.TrainConfig
        actor = cql.TanhGaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            n_hidden_layers=hyperparameters["hidden_layers_actor"],
            activation=get_activation(hyperparameters["activation"]),
            log_std_multiplier=config.policy_log_std_multiplier,
            orthogonal_init=config.orthogonal_init,
            no_tanh=True,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic_1 = cql.FullyConnectedQFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            activation=get_activation(hyperparameters["activation"]),
            orthogonal_init=config.orthogonal_init,
            n_hidden_layers=hyperparameters["hidden_layers_critic"],
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        critic_2 = cql.FullyConnectedQFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            activation=get_activation(hyperparameters["activation"]),
            orthogonal_init=config.orthogonal_init,
            n_hidden_layers=hyperparameters["hidden_layers_critic"],
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        kwargs = {
            "critic_1": critic_1,
            "critic_2": critic_2,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2_optimizer": critic_2_optimizer,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "discount": hyperparameters["discount_factor"],
            "soft_target_update_rate": hyperparameters["target_update_rate"],
            "device": device,
            "max_action": max_action,
            "min_action": min_action,
            # CQL
            "target_entropy": -np.prod(agent_config["action_space"]).item(),
            "alpha_multiplier": config.alpha_multiplier,
            "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
            "backup_entropy": config.backup_entropy,
            "policy_lr": config.policy_lr,
            "qf_lr": config.qf_lr,
            "bc_steps": config.bc_steps,
            "target_update_period": config.target_update_period,
            "cql_n_actions": config.cql_n_actions,
            "cql_importance_sample": config.cql_importance_sample,
            "cql_lagrange": config.cql_lagrange,
            "cql_target_action_gap": config.cql_target_action_gap,
            "cql_temp": config.cql_temp,
            "cql_alpha": config.cql_alpha,
            "cql_max_target_backup": config.cql_max_target_backup,
            "cql_clip_diff_min": config.cql_clip_diff_min,
            "cql_clip_diff_max": config.cql_clip_diff_max,
        }

        return cql.ContinuousCQL(**kwargs)

    if agent_type == "awac":
        config = awac.TrainConfig
        actor = awac.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            activation=get_activation(hyperparameters["activation"]),
            max_action=max_action,
            min_action=min_action,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic_1 = awac.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
            activation=get_activation(hyperparameters["activation"]),
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        critic_2 = awac.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
            activation=get_activation(hyperparameters["activation"]),
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic_1": critic_1,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2": critic_2,
            "critic_2_optimizer": critic_2_optimizer,
            "gamma": hyperparameters["discount_factor"],
            "tau": hyperparameters["target_update_rate"],
            "awac_lambda": config.awac_lambda,
        }
        return awac.AdvantageWeightedActorCritic(**kwargs)

    if agent_type == "edac":
        config = edac.TrainConfig
        actor = edac.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            activation=get_activation(hyperparameters["activation"]),
            max_action=max_action,
            min_action=min_action,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=config.actor_learning_rate,
        )

        critic = edac.VectorizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
            activation=get_activation(hyperparameters["activation"]),
            num_critics=config.num_critics,
        ).to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "gamma": hyperparameters["discount_factor"],
            "tau": hyperparameters["target_update_rate"],
            "eta": config.eta,
            "alpha_learning_rate": config.alpha_learning_rate,
        }

        return edac.EDAC(**kwargs)

    if agent_type == "sac_n":
        config = sac_n.TrainConfig()
        actor = sac_n.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            activation=get_activation(hyperparameters["activation"]),
            max_action=max_action,
            min_action=min_action,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic = edac.VectorizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
            activation=get_activation(hyperparameters["activation"]),
            num_critics=config.num_critics,
        ).to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "gamma": hyperparameters["discount_factor"],
            "tau": hyperparameters["target_update_rate"],
            "alpha_learning_rate": config.alpha_learning_rate,
            "device": device,
        }

        return sac_n.SACN(**kwargs)

    if agent_type == "lb_sac":
        config = lb_sac.TrainConfig()
        actor = lb_sac.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            activation=get_activation(hyperparameters["activation"]),
            max_action=max_action,
            min_action=min_action,
            edac_init=config.edac_init,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=config.actor_learning_rate,
        )

        critic = lb_sac.VectorizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
            activation=get_activation(hyperparameters["activation"]),
            num_critics=config.num_critics,
            layernorm=config.critic_layernorm,
            edac_init=config.edac_init,
        ).to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "gamma": hyperparameters["discount_factor"],
            "tau": hyperparameters["target_update_rate"],
            "alpha_learning_rate": config.alpha_learning_rate,
            "device": device,
        }

        return lb_sac.LBSAC(**kwargs)

    # if agent_type == "iql":
    #     #     # IQL
    #     pass

    # if agent_type == "rebrac":

    # if agent_type == "dt":
    #     pass

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


def combine_runs(root_dir: str, function: str):
    combined_buffer = None
    combined_run_info = None
    combined_run_data = []
    root_path = Path(root_dir)
    agent_dirs = [
        entry.name
        for entry in root_path.iterdir()
        if entry.is_dir() and entry.name != "combined"
    ]
    for dirname in agent_dirs:
        replay_path = Path(root_dir, dirname, function, "rep_buffer")
        run_info_path = Path(root_dir, dirname, function, "run_info.json")
        run_data_path = Path(
            root_dir,
            dirname,
            function,
            "aggregated_run_data.csv",
        )

        df = pd.read_csv(run_data_path)
        combined_run_data.append(df)

        with run_info_path.open(mode="rb") as f:
            run_info = json.load(f)
        temp_buffer = ReplayBuffer.load(replay_path)

        if combined_buffer is None and combined_run_info is None:
            combined_buffer = temp_buffer
            combined_run_info = {
                "environment": run_info["environment"],
                "starting_points": run_info["starting_points"],
                "seed": run_info["seed"],
                "num_runs": run_info["num_runs"],
                "num_batches": run_info["num_batches"],
                "agent": {"type": run_info["agent"]["type"]},
            }
        else:
            combined_buffer.merge(temp_buffer)
            combined_run_info["starting_points"].extend(
                run_info["starting_points"],
            )
    return (
        combined_buffer,
        combined_run_info,
        pd.concat(combined_run_data, ignore_index=True),
    )
