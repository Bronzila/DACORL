from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from typing import Any

import ConfigSpace
import numpy as np
import torch
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    Float,
)
from CORL.algorithms.offline import (
    any_percent_bc as bc,
    awac,
    cql,
    edac,
    iql,
    lb_sac,
    sac_n,
    td3_bc,
)

from src.agents import (
    CSA,
    ConstantAgent,
    ConstantCMAES,
    ExponentialDecayAgent,
    SGDRAgent,
    StepDecayAgent,
    td3,
)
from src.utils.agent_components import (
    ConfigurableCritic,
)


# Time out related class and function
class OutOfTimeError(Exception):
    pass


def timeouthandler(_) -> None:
    raise OutOfTimeError


def set_timeout(timeout: int) -> None:
    if timeout > 0:
        # conversion from hours to seconds
        timeout = timeout * 60 * 60
        signal.signal(signal.SIGALRM, timeouthandler)
        signal.alarm(timeout)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
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
        return ConstantAgent(**teacher_config["params"])
    if teacher_type == "csa":
        return CSA(**teacher_config["params"])
    if teacher_type == "cmaes_constant":
        return ConstantCMAES()

    raise NotImplementedError(
        f"No agent with type {teacher_type} implemented.",
    )


def get_agent(
    agent_type: str,
    agent_config: dict[str, Any],
    tanh_scaling: bool,
    hyperparameters: dict[str, Any] = {},
    device: str = "cpu",
    cmaes: bool = False,
) -> Any:
    if hyperparameters.get("hidden_dim", None) is not None:
        print(
            "Warning! You are using the non reduced config space. Actor_hidden_dim and critic_hidden_dim will be equal.",
        )
        hyperparameters["actor_hidden_dim"] = hyperparameters["hidden_dim"]
        hyperparameters["critic_hidden_dim"] = hyperparameters["hidden_dim"]
    state_dim = agent_config["state_dim"]
    action_dim = agent_config["action_dim"]
    max_action = agent_config["max_action"]
    min_action = agent_config["min_action"]

    if agent_type == "td3_bc":
        config = td3_bc.TrainConfig

        actor = td3_bc.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["actor_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            dropout_rate=hyperparameters["dropout_rate"],
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
        ).to(device)
        print(f"hidden_dim: {hyperparameters.get('hidden_dim', 64)}")
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters.get("lr_actor", 3e-4),
        )

        critic_1 = ConfigurableCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hyperparameters["hidden_layers_critic"],
            hidden_dim=hyperparameters["critic_hidden_dim"],
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters.get("lr_critic", 3e-4),
        )

        critic_2 = ConfigurableCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hyperparameters["hidden_layers_critic"],
            hidden_dim=hyperparameters["critic_hidden_dim"],
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
            "discount": hyperparameters["discount_factor"],
            "tau": hyperparameters["target_update_rate"],
            "device": device,
            # TD3
            "policy_noise": config.policy_noise,
            "noise_clip": config.noise_clip,
            "policy_freq": config.policy_freq,
            # TD3 + BC
            "alpha": config.alpha,
        }
        print("Agent Config:")
        print(kwargs)
        return td3_bc.TD3_BC(**kwargs)

    if agent_type == "bc":
        config = bc.TrainConfig

        actor = bc.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["actor_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
            dropout_rate=hyperparameters["dropout_rate"],
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        kwargs = {
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
            hidden_dim=hyperparameters["actor_hidden_dim"],
            dropout_rate=hyperparameters["dropout_rate"],
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
            n_hidden_layers=hyperparameters["hidden_layers_actor"],
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
            hidden_dim=hyperparameters["critic_hidden_dim"],
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
            hidden_dim=hyperparameters["critic_hidden_dim"],
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
            # CQL
            "target_entropy": -np.prod(action_dim).item(),
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
            hidden_dim=hyperparameters["actor_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            dropout_rate=hyperparameters["dropout_rate"],
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic_1 = awac.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["critic_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        critic_2 = awac.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["critic_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
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
            hidden_dim=hyperparameters["actor_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            dropout_rate=hyperparameters["dropout_rate"],
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
            hidden_dim=hyperparameters["critic_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
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
            hidden_dim=hyperparameters["actor_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            dropout_rate=hyperparameters["dropout_rate"],
            max_action=max_action,
            min_action=min_action,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic = sac_n.VectorizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["critic_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
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
            hidden_dim=hyperparameters["actor_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_actor"],
            dropout_rate=hyperparameters["dropout_rate"],
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
            hidden_dim=hyperparameters["critic_hidden_dim"],
            hidden_layers=hyperparameters["hidden_layers_critic"],
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

    if agent_type == "iql":
        config = iql.TrainConfig
        v_network = iql.ValueFunction(
            state_dim=state_dim,
            hidden_dim=hyperparameters["critic_hidden_dim"],
            n_hidden=hyperparameters["hidden_layers_critic"],
        )
        q_network = iql.TwinQ(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["critic_hidden_dim"],
            n_hidden=hyperparameters["hidden_layers_critic"],
        )
        actor = (
            iql.DeterministicPolicy(
                state_dim,
                action_dim,
                max_action,
                min_action,
                tanh_scaling=tanh_scaling,
                action_positive=cmaes,
                hidden_dim=hyperparameters["actor_hidden_dim"],
                n_hidden=hyperparameters["hidden_layers_actor"],
                dropout=hyperparameters["dropout_rate"],
            )
            if config.iql_deterministic
            else iql.GaussianPolicy(
                state_dim,
                action_dim,
                max_action,
                min_action,
                tanh_scaling=tanh_scaling,
                action_positive=cmaes,
                hidden_dim=hyperparameters["actor_hidden_dim"],
                n_hidden=hyperparameters["hidden_layers_actor"],
                dropout_rate=hyperparameters["dropout_rate"],
            )
        ).to(device)
        v_optimizer = torch.optim.Adam(
            v_network.parameters(),
            lr=hyperparameters["lr_critic"],
        )
        q_optimizer = torch.optim.Adam(
            q_network.parameters(),
            lr=hyperparameters["lr_critic"],
        )
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            "q_optimizer": q_optimizer,
            "v_network": v_network,
            "v_optimizer": v_optimizer,
            "discount": hyperparameters["discount_factor"],
            "tau": hyperparameters["target_update_rate"],
            "device": device,
            # IQL
            "beta": config.beta,
            "iql_tau": config.iql_tau,
            "max_steps": config.max_timesteps,
        }

        # Initialize actor
        return iql.ImplicitQLearning(**kwargs)
    # if agent_type == "rebrac":

    if agent_type == "td3":
        actor = td3.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            min_action=min_action,
            dropout_rate=hyperparameters["dropout_rate"],
            hidden_dim=hyperparameters["actor_hidden_dim"],
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters["lr_actor"],
        )

        critic_1 = td3.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["critic_hidden_dim"],
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters["lr_critic"],
        )

        critic_2 = td3.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hyperparameters["critic_hidden_dim"],
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
            # TD3
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "device": device,
        }
        return td3.TD3(**kwargs)

    raise NotImplementedError(
        f"No agent with type {agent_type} implemented.",
    )


def get_environment(env_config: dict) -> Any:
    from dacbench.benchmarks import (
        CMAESBenchmark,
        FastDownwardBenchmark,
        SGDBenchmark,
        ToySGD2DBenchmark,
    )

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
    if env_config["type"] == "SGD":
        bench = SGDBenchmark(config=env_config)
        return bench.get_environment()
    if env_config["type"] == "CMAES":
        bench = CMAESBenchmark(config=env_config)
        return bench.get_environment()
    if env_config["type"] == "FastDownward":
        bench = FastDownwardBenchmark
        return bench.get_environment()
    raise NotImplementedError(
        f"No environment of type {env_config['type']} found.",
    )


def save_agent(
    state_dicts: dict,
    results_dir: Path,
    iteration: int,
    seed: int = 0,
) -> None:
    filename = results_dir / str(seed) / f"{iteration + 1}"
    if not filename.exists():
        filename.mkdir(parents=True)

    for key, s in state_dicts.items():
        torch.save(s, filename / f"agent_{key}")


def load_agent(
    agent_type: str,
    agent_config: dict,
    agent_path: Path,
    tanh_scaling: bool = False,
) -> Any:
    hp_conf_path = agent_path / "config.json"
    with hp_conf_path.open("r") as f:
        hyperparameters = json.load(f)

    print("Loaded agent with following hyperparameters:")
    print(hyperparameters)

    agent = get_agent(agent_type, agent_config, tanh_scaling, hyperparameters)
    state_dict = agent.state_dict()
    new_state_dict = {}
    for key, _ in state_dict.items():
        s = torch.load(agent_path / f"agent_{key}")
        new_state_dict.update({key: s})

    agent.load_state_dict(new_state_dict)
    return agent


def get_config_space(config_type: str) -> ConfigSpace:
    cs = ConfigurationSpace()

    if config_type == "reduced_no_arch_dropout_256":
        # General
        lr_actor = Float("lr_actor", (1e-5, 1e-2), default=3e-4, log=True)
        lr_critic = Float("lr_critic", (1e-5, 1e-2), default=3e-4, log=True)
        discount_factor = Categorical(
            "discount_factor",
            [0.9, 0.99, 0.999, 0.9999],
            default=0.99,
        )
        target_update_rate = Float(
            "target_update_rate",
            (0, 0.25),
            default=5e-3,
        )
        batch_size = Categorical(
            "batch_size",
            [2, 4, 8, 16, 32, 64, 128, 256],
            default=256,
        )

        # Arch
        hidden_layers_actor = Constant("hidden_layers_actor", 1)
        hidden_layers_critic = Constant("hidden_layers_critic", 1)
        actor_hidden_dim = Constant("actor_hidden_dim", 256)
        critic_hidden_dim = Constant("critic_hidden_dim", 256)
        # Dropout
        dropout_rate = Categorical(
            "dropout_rate",
            [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            default=0.2,
        )

    cs.add_hyperparameters(
        [
            lr_actor,
            lr_critic,
            discount_factor,
            target_update_rate,
            batch_size,
            hidden_layers_actor,
            hidden_layers_critic,
            actor_hidden_dim,
            critic_hidden_dim,
            dropout_rate,
        ],
    )
    return cs
