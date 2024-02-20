from pathlib import Path
import warnings
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Categorical,
    Integer,
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


class TD3BC_Optimizee:
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        batch_size: int,
        debug: bool,
    ) -> None:
        self.data_dir = data_dir
        self.agent_type = agent_type
        self.batch_size = batch_size
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
        activation = Categorical(
            "activation", ["ReLU", "LeakyReLU", "Tanh"], default="ReLU"
        )
        # Add the parameters to configuration space
        cs.add_hyperparameters(
            [
                lr_actor,
                lr_critic,
                hidden_layers_actor,
                hidden_layers_critic,
                activation,
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
            batch_size=self.batch_size,
            val_freq=budget + 1,
            seed=seed,
            wandb_group=None,
            timeout=0,
            hyperparameters=config,
            debug=self.debug,
        )

        # env = get_environment(self.run_info["environment"])
        # results = test_agent(
        #     actor=agent.actor,
        #     env=env,
        #     n_runs=50,
        #     n_batches=self.run_info["environment"]["num_batches"],
        #     seed=seed,
        # )
        # mean, std = calc_mean_and_std_dev(results)
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
        "--debug",
        action="store_true",
        help="Run for max. 5 iterations and don't log in wanbd.",
    )

    args = parser.parse_args()

    optimizee = TD3BC_Optimizee(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        batch_size=args.batch_size,
        debug=args.debug,
    )

    scenario = Scenario(
        optimizee.configspace,
        walltime_limit=60 * 60,  # convert 1 hour into seconds
        n_trials=50,  # Evaluate max 500 different trials
        min_budget=25,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
        max_budget=100,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
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
    )
    incumbent = smac.optimize()

    plot_trajectory(smac)
