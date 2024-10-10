from __future__ import annotations

import json
import random
import signal
from pathlib import Path
from types import FrameType
from typing import Any, Union

import ConfigSpace
import numpy as np
import torch
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant as ConstantHP,
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
    SGDR,
    Constant,
    ConstantCMAES,
    ExponentialDecay,
    StepDecay,
    td3,
)

ActorType = Union[
    bc.Actor,
    td3_bc.Actor,
    td3.Actor,
    edac.Actor,
    sac_n.Actor,
    lb_sac.Actor,
]

CriticType = Union[
    td3_bc.Critic,
    td3.Critic,
    awac.Critic,
    cql.FullyConnectedQFunction,
    edac.VectorizedCritic,
    sac_n.VectorizedCritic,
    lb_sac.VectorizedCritic,
    iql.TwinQ,
]


# Time out related class and function
class OutOfTimeError(Exception):
    pass


def timeouthandler(_signum: int, _frame: FrameType | None) -> Any:
    raise OutOfTimeError


def set_timeout(timeout: int) -> None:
    if timeout > 0:
        # conversion from hours to seconds
        timeout = timeout * 60 * 60
        signal.signal(signal.SIGALRM, timeouthandler)
        signal.alarm(timeout)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)


def get_teacher(
    teacher_config: dict[str, Any],
) -> Any:
    if teacher_config["type"] == "step_decay":
        return StepDecay(**teacher_config["params"])
    if teacher_config["type"] == "exponential_decay":
        return ExponentialDecay(**teacher_config["params"])
    if teacher_config["type"] == "sgdr":
        return SGDR(**teacher_config["params"])
    if teacher_config["type"] == "constant":
        return Constant(**teacher_config["params"])
    if teacher_config["type"] == "csa":
        return CSA(**teacher_config["params"])
    if teacher_config["type"] == "cmaes_constant":
        return ConstantCMAES()

    raise NotImplementedError(
        f"No agent with type {teacher_config['type']} implemented.",
    )


def get_agent(
    agent_type: str,
    agent_config: dict[str, Any],
    tanh_scaling: bool,
    hyperparameters: dict[str, Any] = {},
    device: str = "cpu",
    cmaes: bool = False,
) -> Any:
    if hyperparameters.get("hidden_dim") is not None:
        print(
            "Warning! You are using the non reduced config space. Actor_hidden_dim and critic_hidden_dim will be equal.",
        )
        hyperparameters["actor_hidden_dim"] = hyperparameters["hidden_dim"]
        hyperparameters["critic_hidden_dim"] = hyperparameters["hidden_dim"]
    state_dim = agent_config["state_dim"]
    action_dim = agent_config["action_dim"]
    max_action = agent_config["max_action"]
    min_action = agent_config["min_action"]

    actor: torch.nn.Module
    critic: CriticType
    critic_1: CriticType
    critic_2: CriticType

    actor_hidden_dim = hyperparameters.get("actor_hidden_dim", 256)
    actor_hidden_layers = hyperparameters.get("hidden_layers_actor", 1)
    critic_hidden_dim = hyperparameters.get("critic_hidden_dim", 256)
    critic_hidden_layers = hyperparameters.get("hidden_layers_critic", 1)
    dropout_rate = hyperparameters.get("dropout_rate", 0.2)
    lr_actor = hyperparameters.get("lr_actor", 3e-4)
    lr_critic = hyperparameters.get("lr_critic", 3e-4)
    discount_factor = hyperparameters.get("discount_factor", 0.99)
    tau = hyperparameters.get("target_update_rate", 5e-3)

    if agent_type == "td3_bc":
        td3_bc_config = td3_bc.TrainConfig

        actor = td3_bc.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            hidden_layers=actor_hidden_layers,
            dropout_rate=dropout_rate,
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
        ).to(device)

        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr_actor,
        )

        critic_1 = td3_bc.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=critic_hidden_layers,
            hidden_dim=critic_hidden_dim,
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=lr_critic,
        )

        critic_2 = td3_bc.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=critic_hidden_layers,
            hidden_dim=critic_hidden_dim,
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=lr_critic,
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
            "discount": discount_factor,
            "tau": tau,
            "device": device,
            # TD3
            "policy_noise": td3_bc_config.policy_noise,
            "noise_clip": td3_bc_config.noise_clip,
            "policy_freq": td3_bc_config.policy_freq,
            # TD3 + BC
            "alpha": td3_bc_config.alpha,
        }
        print("Agent Config:")
        print(kwargs)
        return td3_bc.TD3_BC(**kwargs)

    if agent_type == "bc":
        actor = bc.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            hidden_layers=actor_hidden_layers,
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
            dropout_rate=dropout_rate,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr_actor,
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "discount": discount_factor,
            "device": device,
        }
        return bc.BC(**kwargs)

    if agent_type == "cql":
        cql_config = cql.TrainConfig
        actor = cql.TanhGaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            dropout_rate=dropout_rate,
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
            n_hidden_layers=actor_hidden_layers,
            log_std_multiplier=cql_config.policy_log_std_multiplier,
            orthogonal_init=cql_config.orthogonal_init,
            no_tanh=True,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr_actor,
        )

        critic_1 = cql.FullyConnectedQFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            orthogonal_init=cql_config.orthogonal_init,
            n_hidden_layers=critic_hidden_layers,
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=lr_critic,
        )

        critic_2 = cql.FullyConnectedQFunction(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            orthogonal_init=cql_config.orthogonal_init,
            n_hidden_layers=critic_hidden_layers,
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=lr_critic,
        )

        kwargs = {
            "critic_1": critic_1,
            "critic_2": critic_2,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2_optimizer": critic_2_optimizer,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "discount": discount_factor,
            "soft_target_update_rate": tau,
            "device": device,
            # CQL
            "target_entropy": -np.prod(action_dim).item(),
            "alpha_multiplier": cql_config.alpha_multiplier,
            "use_automatic_entropy_tuning": cql_config.use_automatic_entropy_tuning,
            "backup_entropy": cql_config.backup_entropy,
            "policy_lr": cql_config.policy_lr,
            "qf_lr": cql_config.qf_lr,
            "bc_steps": cql_config.bc_steps,
            "target_update_period": cql_config.target_update_period,
            "cql_n_actions": cql_config.cql_n_actions,
            "cql_importance_sample": cql_config.cql_importance_sample,
            "cql_lagrange": cql_config.cql_lagrange,
            "cql_target_action_gap": cql_config.cql_target_action_gap,
            "cql_temp": cql_config.cql_temp,
            "cql_alpha": cql_config.cql_alpha,
            "cql_max_target_backup": cql_config.cql_max_target_backup,
            "cql_clip_diff_min": cql_config.cql_clip_diff_min,
            "cql_clip_diff_max": cql_config.cql_clip_diff_max,
        }

        return cql.ContinuousCQL(**kwargs)

    if agent_type == "awac":
        awac_config = awac.TrainConfig
        actor = awac.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            hidden_layers=actor_hidden_layers,
            dropout_rate=dropout_rate,
            max_action=max_action,
            min_action=min_action,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr_actor,
        )

        critic_1 = awac.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            hidden_layers=critic_hidden_layers,
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=lr_critic,
        )

        critic_2 = awac.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            hidden_layers=critic_hidden_layers,
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=lr_critic,
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic_1": critic_1,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2": critic_2,
            "critic_2_optimizer": critic_2_optimizer,
            "gamma": discount_factor,
            "tau": tau,
            "awac_lambda": awac_config.awac_lambda,
        }
        return awac.AdvantageWeightedActorCritic(**kwargs)

    if agent_type == "edac":
        edac_config = edac.TrainConfig
        actor = edac.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            hidden_layers=actor_hidden_layers,
            dropout_rate=dropout_rate,
            max_action=max_action,
            min_action=min_action,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=edac_config.actor_learning_rate,
        )

        critic = edac.VectorizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            hidden_layers=critic_hidden_layers,
            num_critics=edac_config.num_critics,
        ).to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=lr_critic,
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "gamma": discount_factor,
            "tau": tau,
            "eta": edac_config.eta,
            "alpha_learning_rate": edac_config.alpha_learning_rate,
        }

        return edac.EDAC(**kwargs)

    if agent_type == "sac_n":
        sac_n_config = sac_n.TrainConfig()
        actor = sac_n.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            hidden_layers=actor_hidden_layers,
            dropout_rate=dropout_rate,
            max_action=max_action,
            min_action=min_action,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr_actor,
        )

        critic = sac_n.VectorizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            hidden_layers=critic_hidden_layers,
            num_critics=sac_n_config.num_critics,
        ).to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=lr_critic,
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "gamma": discount_factor,
            "tau": tau,
            "alpha_learning_rate": sac_n_config.alpha_learning_rate,
            "device": device,
        }

        return sac_n.SACN(**kwargs)

    if agent_type == "lb_sac":
        lb_sac_config = lb_sac.TrainConfig()
        actor = lb_sac.Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            hidden_layers=actor_hidden_layers,
            dropout_rate=dropout_rate,
            max_action=max_action,
            min_action=min_action,
            edac_init=lb_sac_config.edac_init,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lb_sac_config.actor_learning_rate,
        )

        critic = lb_sac.VectorizedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            hidden_layers=critic_hidden_layers,
            num_critics=lb_sac_config.num_critics,
            layernorm=lb_sac_config.critic_layernorm,
            edac_init=lb_sac_config.edac_init,
        ).to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=lr_critic,
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "gamma": discount_factor,
            "tau": tau,
            "alpha_learning_rate": lb_sac_config.alpha_learning_rate,
            "device": device,
        }

        return lb_sac.LBSAC(**kwargs)

    if agent_type == "iql":
        iql_config = iql.TrainConfig
        v_network = iql.ValueFunction(
            state_dim=state_dim,
            hidden_dim=critic_hidden_dim,
            n_hidden=critic_hidden_layers,
        )
        q_network = iql.TwinQ(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            n_hidden=critic_hidden_layers,
        )
        actor = (
            iql.DeterministicPolicy(
                state_dim,
                action_dim,
                max_action,
                min_action,
                tanh_scaling=tanh_scaling,
                hidden_dim=actor_hidden_dim,
                n_hidden=actor_hidden_layers,
                dropout_rate=dropout_rate,
            )
            if iql_config.iql_deterministic
            else iql.GaussianPolicy(
                state_dim,
                action_dim,
                max_action,
                min_action,
                tanh_scaling=tanh_scaling,
                action_positive=cmaes,
                hidden_dim=actor_hidden_dim,
                n_hidden=actor_hidden_layers,
                dropout_rate=dropout_rate,
            )
        ).to(device)
        v_optimizer = torch.optim.Adam(
            v_network.parameters(),
            lr=lr_critic,
        )
        q_optimizer = torch.optim.Adam(
            q_network.parameters(),
            lr=lr_critic,
        )
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr_actor,
        )

        kwargs = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            "q_optimizer": q_optimizer,
            "v_network": v_network,
            "v_optimizer": v_optimizer,
            "discount": discount_factor,
            "tau": tau,
            "device": device,
            # IQL
            "beta": iql_config.beta,
            "iql_tau": iql_config.iql_tau,
            "max_steps": iql_config.max_timesteps,
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
            dropout_rate=dropout_rate,
            hidden_dim=actor_hidden_dim,
            tanh_scaling=tanh_scaling,
            action_positive=cmaes,
        ).to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=lr_actor,
        )

        critic_1 = td3.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=lr_critic,
        )

        critic_2 = td3.Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=lr_critic,
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
            "discount": discount_factor,
            "tau": tau,
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
    from DACBench.dacbench.benchmarks import (
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
        hidden_layers_actor = ConstantHP("hidden_layers_actor", 1)
        hidden_layers_critic = ConstantHP("hidden_layers_critic", 1)
        actor_hidden_dim = ConstantHP("actor_hidden_dim", 256)
        critic_hidden_dim = ConstantHP("critic_hidden_dim", 256)
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
