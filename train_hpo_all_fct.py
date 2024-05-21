import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch.nn as nn
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Constant,
    Float,
    Integer,
)
from matplotlib import pyplot as plt
from smac import (
    HyperbandFacade,
    HyperparameterOptimizationFacade as HPOFacade,
    Scenario,
)

from src.utils.general import set_seeds
from src.utils.train_agent import train_agent

warnings.filterwarnings("ignore")

mean_mappings = {
    "Ackley": {
        "exponential_decay": 6.953,
        "step_decay": 6.917,
        "sgdr": 6.601,
        "constant": 6.930
    },
    "Rastrigin": {
        "exponential_decay": 17.6,
        "step_decay": 17.49,
        "sgdr": 16.81,
        "constant": 17.84
    },
    "Rosenbrock": {
        "exponential_decay": 3.122,
        "step_decay": 3.095,
        "sgdr": 2.865,
        "constant": 2.895
    },
    "Sphere": {
        "exponential_decay": 1.875e-3,
        "step_decay": 2.796e-4,
        "sgdr": 8.441e-3,
        "constant": 4.066e-4
    }
}

class TD3BC_Optimizee:
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        debug: bool,
        budget: int,
        eval_protocol: str,
        eval_seed: int,
    ) -> None:
        self.data_dir = data_dir
        self.agent_type = agent_type
        self.debug = debug
        self.budget = budget
        self.eval_protocol = eval_protocol
        self.eval_seed = eval_seed

        with Path(self.data_dir, "Ackley", "run_info.json").open(mode="rb") as f:
            self.run_info = json.load(f)

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        lr_actor = Float("lr_actor", (1e-5, 1e-2), default=3e-4)
        lr_critic = Float("lr_critic", (1e-5, 1e-2), default=3e-4)
        # hidden_layers_actor = Integer("hidden_layers_actor", (0, 5), default=1)
        # hidden_layers_critic = Integer(
        #     "hidden_layers_critic", (0, 5), default=1
        # )
        activation = Constant(
            "activation", "ReLU"
        )
        batch_size = Categorical(
            "batch_size", [2, 4, 8, 16, 32, 64, 128, 256], default=64
        )
        # discount_factor = Float("discount_factor", (0, 1), default=0.99)
        # target_update_rate = Float("target_update_rate", (0, 1), default=5e-3)
        # Add the parameters to configuration space
        cs.add_hyperparameters(
            [
                lr_actor,
                lr_critic,
                # hidden_layers_actor,
                # hidden_layers_critic,
                activation,
                batch_size,
                # discount_factor,
                # target_update_rate,
            ],
        )
        return cs

    def train(
        self, config: Configuration, seed: int = 0
    ) -> float:
        # results = []
        # for _seed in range(10)
        #     log_dict, eval_mean = train_agent(
        #         data_dir=self.data_dir,
        #         agent_type=self.agent_type,
        #         agent_config={},
        #         num_train_iter=self.budget,
        #         batch_size=config["batch_size"],
        #         val_freq=int(self.budget),
        #         seed=_seed,
        #         wandb_group=None,
        #         timeout=0,
        #         hyperparameters=config,
        #         debug=self.debug,
        #         eval_protocol=self.eval_protocol,
        #         eval_seed=self.eval_seed,
        #     )
        #     results.append(eval_mean)

        # return np.array(results).mean()

        print(seed)
        fct_results = []
        for function in ["Ackley", "Rastrigin", "Rosenbrock", "Sphere"]:
            teacher_type = self.run_info["agent"]["type"]
            data_dir = Path(self.data_dir, function)
            log_dict, eval_mean = train_agent(
                data_dir=data_dir,
                agent_type=self.agent_type,
                agent_config={},
                num_train_iter=self.budget,
                batch_size=config["batch_size"],
                val_freq=int(self.budget),
                seed=seed,
                wandb_group=None,
                timeout=0,
                hyperparameters=config,
                debug=self.debug,
                eval_protocol=self.eval_protocol,
                eval_seed=self.eval_seed,
            )
            print(f"Results for function {function}: {eval_mean}")
            normalized_mean = eval_mean / mean_mappings[function][teacher_type]
            print(f"Normalized mean: {normalized_mean}")
            fct_results.append(normalized_mean)

        print(f"Mean over all functions: {np.array(fct_results).mean()}")
        return np.array(fct_results).mean()

    @property
    def configspace_arch(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        lr_actor = Float("lr_actor", (1e-5, 1e-2), default=3e-4)
        lr_critic = Float("lr_critic", (1e-5, 1e-2), default=3e-4)
        hidden_layers = Integer("hidden_layers", (0, 5), default=1)
        hidden_dim = Categorical(
            "hidden_dim", [16, 32, 64, 128, 256], default=64
        )
        # hidden_layers_critic = Integer(
        #     "hidden_layers_critic", (0, 5), default=1
        # )
        activation = Constant(
            "activation", "ReLU"
        )
        batch_size = Categorical(
            "batch_size", [2, 4, 8, 16, 32, 64, 128, 256], default=64
        )
        # discount_factor = Float("discount_factor", (0, 1), default=0.99)
        # target_update_rate = Float("target_update_rate", (0, 1), default=5e-3)
        # Add the parameters to configuration space
        cs.add_hyperparameters(
            [
                lr_actor,
                lr_critic,
                hidden_layers,
                hidden_dim,
                # hidden_layers_critic,
                activation,
                batch_size,
                # discount_factor,
                # target_update_rate,
            ],
        )
        return cs


def plot_trajectory(facade: HPOFacade, output_path) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facade.scenario.objectives)

    X, Y = [], []
    for item in facade.intensifier.trajectory:
        # Single-objective optimization
        assert len(item.config_ids) == 1
        assert len(item.costs) == 1

        y = item.costs[0]
        x = item.walltime

        X.append(x)
        Y.append(y)

    print(X)
    print(Y)
    plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
    plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.savefig(output_path / "traj.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPO for any agent")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="path to the directory where replay_buffer and info about the replay_buffer are stored",
    )
    parser.add_argument(
        "--agent_type", type=str, default="td3_bc", choices=["td3_bc"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_limit", type=int, default=30)
    parser.add_argument("--budget", type=int, default=15000)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run for max. 5 iterations and don't log in wanbd.",
    )
    parser.add_argument(
        "--arch_cs",
        action="store_true",
        help="Use architecture config space.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path where optimization logs are saved",
        default="smac"
    )
    parser.add_argument(
        "--eval_protocol", type=str, default="train", choices=["train", "interpolation"]
    )
    parser.add_argument("--eval_seed", type=int, default=123)

    args = parser.parse_args()
    set_seeds(args.seed)

    optimizee = TD3BC_Optimizee(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        debug=args.debug,
        budget=args.budget,
        eval_protocol=args.eval_protocol,
        eval_seed=args.eval_seed,
    )
    output_path = Path(args.output_path)
    cs = optimizee.configspace_arch if args.arch_cs else optimizee.configspace
    scenario = Scenario(
        cs,
        output_directory=output_path,
        walltime_limit=60 * 60 * args.time_limit,  # convert 10 hours into seconds
        n_trials=800,
        n_workers=1,
        deterministic=False,
    )

    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=10)
    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        optimizee.train,
        initial_design=initial_design,
        overwrite=True,
        intensifier=intensifier,
        logging_level=20,
    )
    incumbent = smac.optimize()

    print("Incumbent:")
    print(incumbent)
    print(f"Final score: {smac.validate(incumbent)}")

    plot_trajectory(smac, output_path)

    with (output_path / "inc.json").open("w") as f:
        json.dump(dict(incumbent), f)