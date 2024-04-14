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
    Float,
    Integer,
    Constant
)
from matplotlib import pyplot as plt
from smac import (
    HyperbandFacade,
    MultiFidelityFacade as MFFacade,
    Scenario,
)

from src.utils.general import set_seeds
from src.utils.train_agent import train_agent

warnings.filterwarnings("ignore")


class TD3BC_Optimizee:
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        debug: bool,
        eval_protocol: str,
        eval_seed: int,
    ) -> None:
        self.data_dir = data_dir
        self.agent_type = agent_type
        self.debug = debug
        self.eval_protocol = eval_protocol
        self.eval_seed = eval_seed

        with Path(self.data_dir, "run_info.json").open(mode="rb") as f:
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
        self, config: Configuration, seed: int = 0, budget: int = 25
    ) -> float:
        print(seed)
        log_dict, eval_mean = train_agent(
            data_dir=self.data_dir,
            agent_type=self.agent_type,
            agent_config={},
            num_train_iter=budget,
            batch_size=config["batch_size"],
            val_freq=int(budget),
            seed=seed,
            wandb_group=None,
            timeout=0,
            hyperparameters=config,
            debug=self.debug,
            eval_protocol=self.eval_protocol,
            eval_seed=self.eval_seed,
        )

        return eval_mean


def plot_trajectory(facade: MFFacade, output_path) -> None:
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run for max. 5 iterations and don't log in wanbd.",
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
        eval_protocol=args.eval_protocol,
        eval_seed=args.eval_seed,
    )
    output_path = Path(args.output_path)
    scenario = Scenario(
        optimizee.configspace,
        output_directory=output_path,
        walltime_limit=60 * 60 * 10,  # convert 10 hours into seconds
        n_trials=5000,
        min_budget=1000,
        max_budget=15000,
        n_workers=1,
        deterministic=False,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        optimizee.train,
        initial_design=initial_design,
        overwrite=True,
        logging_level=20,
    )
    incumbent = smac.optimize()

    print("Incumbent:")
    print(incumbent)
    print(f"Final score: {smac.validate(incumbent)}")

    plot_trajectory(smac, output_path)

    with (output_path / "inc.json").open("w") as f:
        json.dump(dict(incumbent), f)