import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

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

from src.utils.general import get_config_space, set_seeds
from src.utils.train_agent import train_agent as train_offline
from src.utils.train_agent_online import train_agent as train_online

warnings.filterwarnings("ignore")


class Optimizee:
    def __init__(
        self,
        data_dir: str,
        agent_type: str,
        debug: bool,
        budget: int,
        eval_protocol: str,
        eval_seed: int,
        hidden_dim: int,
        tanh_scaling: bool,
    ) -> None:
        self.data_dir = data_dir
        self.agent_type = agent_type
        self.debug = debug
        self.budget = budget
        self.eval_protocol = eval_protocol
        self.eval_seed = eval_seed
        self.hidden_dim = hidden_dim
        self.tanh_scaling = tanh_scaling

        if agent_type == "td3":
            self.train_agent = train_online
        else:
            self.train_agent = train_offline
        with Path(self.data_dir, "run_info.json").open(mode="rb") as f:
            self.run_info = json.load(f)

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        lr_actor = Float("lr_actor", (1e-5, 1e-2), default=3e-4, log=True)
        lr_critic = Float("lr_critic", (1e-5, 1e-2), default=3e-4, log=True)
        activation = Constant(
            "activation", "ReLU"
        )
        batch_size = Categorical(
            "batch_size", [2, 4, 8, 16, 32, 64, 128, 256], default=64
        )

        # Add the parameters to configuration space
        cs.add_hyperparameters(
            [
                lr_actor,
                lr_critic,
                activation,
                batch_size,
            ],
        )
        return cs

    def train(
        self, config: Configuration, seed: int = 0
    ) -> float:
        print(seed)
        config = dict(config)
        config["hidden_dim"] = self.hidden_dim
        log_dict, eval_mean = train_agent(
            data_dir=self.data_dir,
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
            use_wandb=False,
            tanh_scaling=self.tanh_scaling,
        )
        print(f"Results for seed {seed}: {eval_mean}")
        return eval_mean

    @property
    def configspace_arch(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        lr_actor = Float("lr_actor", (1e-5, 1e-2), default=3e-4, log=True)
        lr_critic = Float("lr_critic", (1e-5, 1e-2), default=3e-4, log=True)
        hidden_layers = Integer("hidden_layers", (0, 5), default=1)
        hidden_dim = Categorical(
            "hidden_dim", [16, 32, 64, 128, 256], default=64
        )
        activation = Constant(
            "activation", "ReLU"
        )
        batch_size = Categorical(
            "batch_size", [2, 4, 8, 16, 32, 64, 128, 256], default=64
        )
        # Add the parameters to configuration space
        cs.add_hyperparameters(
            [
                lr_actor,
                lr_critic,
                hidden_layers,
                hidden_dim,
                activation,
                batch_size,
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
        "--output_path",
        type=str,
        help="Path where optimization logs are saved",
        default="smac",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="td3_bc",
        choices=["bc", "td3_bc", "cql", "awac", "edac", "sac_n", "lb_sac", "iql", "td3"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_limit", type=int, default=30)
    parser.add_argument("--budget", type=int, default=30000)
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="If set, architectural parameters will be constant",
    )
    parser.add_argument(
        "--save_incumbent",
        action="store_true",
        default=True,
        help="Flag if we save the incumbent configuration",
    )
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
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--tanh_scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument(
        "--cs_type",
        type=str,
        help="Which config space to use",
        default="reduced_dropout"
    )

    args = parser.parse_args()
    set_seeds(args.seed)

    optimizee = Optimizee(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        debug=args.debug,
        budget=args.budget,
        eval_protocol=args.eval_protocol,
        eval_seed=args.eval_seed,
        hidden_dim=args.hidden_dim,
        tanh_scaling=args.tanh_scaling,
    )

    output_path = Path(args.output_path)
    cs = get_config_space(args.cs_type)
    scenario = Scenario(
        cs,
        output_directory=output_path,
        walltime_limit=60 * 60 * args.time_limit,  # convert 10 hours into seconds
        n_trials=400,
        n_workers=1,
        deterministic=False,
    )

    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=10)
    # We want to run five random configurations before starting the optimization.
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5, additional_configs=[
        cs.get_default_configuration()
    ])

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

    if args.save_incumbent:
        save_config_dir = Path(args.data_dir) / "results" / args.agent_type
        save_config_dir.mkdir(exist_ok=True)
        path = save_config_dir / "incumbent.json"
        print(f"Saving incumbent to : {path}")
        with path.open("w") as f:
            json.dump(incumbent.get_dictionary(), f, indent=4)

    print(smac.validate(incumbent))

    with (output_path / "inc.json").open("w") as f:
        json.dump(dict(incumbent), f)