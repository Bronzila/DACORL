from pathlib import Path
import warnings
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Categorical,
    Integer,
    Constant,
)
from matplotlib import pyplot as plt
from smac import HyperbandFacade, MultiFidelityFacade as MFFacade, Scenario
import torch.nn as nn
import argparse
from utils.general import get_environment
from utils.test_agent import test_agent
import json
import numpy as np

from utils.train_agent import train_agent

from check_fbest import calc_mean_and_std_dev

warnings.filterwarnings("ignore")


class Optimizee:
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        debug: bool,
    ) -> None:
        self.data_dir = data_dir
        self.agent_type = agent_type
        self.debug = debug

        with Path(self.data_dir, "run_info.json").open(mode="rb") as f:
            self.run_info = json.load(f)

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        lr_actor = Float("lr_actor", (1e-5, 1e-2), default=3e-4)
        lr_critic = Float("lr_critic", (1e-5, 1e-2), default=3e-4)
        hidden_layers_actor = Integer("hidden_layers_actor", (0, 5), default=1)
        hidden_layers_critic = Integer(
            "hidden_layers_critic", (0, 5), default=1
        )
        hidden_dim = Categorical("hidden_dim", [32, 64, 128, 256, 512], default=256)
        activation = Categorical(
            "activation", ["ReLU", "LeakyReLU", "Tanh"], default="ReLU"
        )
        batch_size = Categorical(
            "batch_size", [2, 4, 8, 16, 32, 64, 128, 256], default=64
        )
        discount_factor = Float("discount_factor", (0, 1), default=0.99)
        target_update_rate = Float("target_update_rate", (0, 1), default=5e-3)
        # Add the parameters to configuration space
        cs.add_hyperparameters(
            [
                lr_actor,
                lr_critic,
                hidden_layers_actor,
                hidden_layers_critic,
                hidden_dim,
                activation,
                batch_size,
                discount_factor,
                target_update_rate,
            ]
        )
        return cs

    @property
    def configspace_reduced(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        lr_actor = Float("lr_actor", (1e-5, 1e-2), default=3e-4)
        lr_critic = Float("lr_critic", (1e-5, 1e-2), default=3e-4)
        hidden_layers_actor = Constant("hidden_layers_actor", 1)
        hidden_layers_critic = Constant("hidden_layers_critic", 1)
        hidden_dim = Constant("hidden_dim", 256)
        activation = Constant("activation", "ReLU")
        batch_size = Categorical(
            "batch_size", [2, 4, 8, 16, 32, 64, 128, 256], default=64
        )
        discount_factor = Float("discount_factor", (0, 1), default=0.99)
        target_update_rate = Float("target_update_rate", (0, 1), default=5e-3)
        # Add the parameters to configuration space
        cs.add_hyperparameters(
            [
                lr_actor,
                lr_critic,
                hidden_layers_actor,
                hidden_layers_critic,
                hidden_dim,
                activation,
                batch_size,
                discount_factor,
                target_update_rate,
            ]
        )
        return cs

    def train(
        self, config: Configuration, seed: int = 0, budget: int = 25
    ) -> float:
        log_dict = train_agent(
            data_dir=self.data_dir,
            agent_type=self.agent_type,
            agent_config={},
            num_train_iter=budget,
            num_eval_runs=0,
            batch_size=config["batch_size"],
            val_freq=budget + 1,
            seed=seed,
            wandb_group=None,
            timeout=0,
            hyperparameters=config,
            debug=self.debug,
        )

        return np.mean(log_dict["actor_loss"])


def plot_trajectory(facade: MFFacade) -> None:
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
    plt.show()


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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="If set, architectural parameters will be constant",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run for max. 5 iterations and don't log in wanbd.",
    )

    args = parser.parse_args()

    optimizee = Optimizee(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        batch_size=args.batch_size,
        debug=args.debug,
    )

    scenario = Scenario(
        (
            optimizee.configspace_reduced
            if args.reduced
            else optimizee.configspace
        ),
        walltime_limit=10 * 60 * 60,  # convert 10 hour into seconds
        n_trials=100,  # Evaluate max 500 different trials
        min_budget=1000,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
        max_budget=10000,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
        n_workers=8,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        optimizee.train,
        initial_design=initial_design,
        overwrite=True,
        # logging_level=0,
    )
    incumbent = smac.optimize()

    print(smac.validate(incumbent))

    plot_trajectory(smac)
