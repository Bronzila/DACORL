import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch.nn as nn

from tap import Tap
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Categorical,
    Constant,
)
from smac import (
    HyperparameterOptimizationFacade as HPOFacade,
    Scenario,
)

from src.utils.general import set_seeds
from src.utils.generate_data import generate_dataset

warnings.filterwarnings("ignore")


class TD3BC_Optimizee:
    def __init__(
        self,
        data_dir: Path,
        agent_type: str,
        env: str,
    ) -> None:
        self.data_dir = data_dir
        self.agent_type = agent_type
        self.env_configs = []
        if env:
            env_config_path = Path("configs", "environment", "SGD", env + ".json")
            with env_config_path.open() as file:
                self.env_configs.append(json.load(file))
        else:
            for filename in ["Ackley_extended_vel.json",
                            "Rastrigin_extended_vel.json", "Rosenbrock_extended_vel.json",
                            "Sphere_extended_vel.json"]:
                env_config_path = Path("configs", "environment", filename)
                with env_config_path.open() as file:
                    env_config = json.load(file)
                self.env_configs.append(env_config)

    @property
    def configspace(self) -> ConfigurationSpace:
        # batches_per_epoch
        bpe = 450 #375

        cs = ConfigurationSpace()
        if self.agent_type == "exponential_decay":
            decay_steps = Categorical("decay_steps", [int(bpe/2), bpe, int(bpe*1.5), bpe*2], default=bpe)
            decay_rate = Categorical("decay_rate", [0.8, 0.85, 0.9, 0.95], default=0.9)
            # Add the parameters to configuration space
            cs.add_hyperparameters(
                [
                    decay_steps,
                    decay_rate,
                ],
            )
        elif self.agent_type == "step_decay":
            step_size = Categorical("step_size", [int(bpe/2), bpe, int(bpe*1.5), bpe*2], default=bpe)
            gamma = Categorical("gamma", [0.8, 0.85, 0.9, 0.95], default=0.9)
            # Add the parameters to configuration space
            cs.add_hyperparameters(
                [
                    step_size,
                    gamma,
                ],
            )
        elif self.agent_type == "sgdr":
            T_i = Categorical("T_i", [1,2,3,4], default=1)
            T_mult = Categorical("T_mult", [1,2,3], default=2)
            batches_per_epoch = Constant("batches_per_epoch", bpe)
            # Add the parameters to configuration space
            cs.add_hyperparameters(
                [
                    T_i,
                    T_mult,
                    batches_per_epoch,
                ],
            )
        elif self.agent_type == "constant":
            learning_rate = Categorical("learning_rate", [0.0025, 0.002, 0.0015, 0.00125, 0.001, 0.00075, 0.0005, 0.0001], default=0.001)
            cs.add_hyperparameters(
                [
                    learning_rate
                ]
            )
        return cs


    def train(
        self, config: Configuration, seed: int = 0
    ) -> float:
        seed = 0
        config = dict(config)
        print(config)
        agent_config = {
            "params": config,
            "id": 0,
            "type": self.agent_type,
        }
        results = []
        for env_config in self.env_configs:
            if self.agent_type == "sgdr":
                agent_config["params"]["initial_learning_rate"] = env_config["initial_learning_rate"]
            if self.agent_type == "constant":
                env_config["initial_learning_rate"] = config["learning_rate"]
            agg_run_data = generate_dataset(agent_config, env_config, num_runs=1, seed=seed,
                             results_dir=self.data_dir, save_run_data=True, timeout=0,
                             save_rep_buffer=True, checkpoint=0, checkpointing_freq=0, check_if_exists=False)

            final_evaluations = agg_run_data.groupby("run").last()
            train_loss = final_evaluations["train_loss"]
            valid_loss = final_evaluations["valid_loss"]
            test_loss = final_evaluations["test_loss"]
            train_acc = final_evaluations["train_acc"]
            valid_acc = final_evaluations["valid_acc"]
            test_acc = final_evaluations["test_acc"]
            print(f"Train Loss: {train_loss.mean()}")
            print(f"Valid Loss: {valid_loss.mean()}")
            print(f"Test Loss: {test_loss.mean()}")
            print(f"Train Acc: {train_acc.mean()}")
            print(f"Valid Acc: {valid_acc.mean()}")
            print(f"Test Acc: {test_acc.mean()}")
            results.append(valid_loss.mean())

        print(f"Mean result: {np.mean(results)}")
        return np.mean(results)


if __name__ == "__main__":
    class TeacherHPOParser(Tap):
        data_dir: Path # Path to the RelayBuffer
        agent_type: str = "exponential_decay"
        seed: int = 0
        env_config: str # Environment to tune HPs on, if None, tune for all environments.
        output_path: Path = "smac"

    args = TeacherHPOParser().parse_args()
    set_seeds(args.seed)

    optimizee = TD3BC_Optimizee(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        env=args.env_config,
    )
    scenario = Scenario(
        optimizee.configspace,
        output_directory=args.output_path,
        n_trials=30,
        n_workers=1,
        deterministic=False,
    )

    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=1)

    # Create our SMAC object and pass the scenario and the train method
    smac = HPOFacade(
        scenario,
        optimizee.train,
        intensifier=intensifier,
        overwrite=True,
        logging_level=20,
    )
    incumbent = smac.optimize()

    print("Incumbent:")
    print(incumbent)
    print(f"Final score: {smac.validate(incumbent)}")

    with (args.output_path / "inc.json").open("w") as f:
        json.dump(dict(incumbent), f)

    lowest_val_confs = smac.runhistory.get_configs(sort_by="cost")[:15]
    lowest_val_path = args.output_path / "lowest"
    lowest_val_path.mkdir(exist_ok=True)
    for id, config in enumerate(lowest_val_confs):
        with (lowest_val_path / f"{id}.json").open("w") as f:
            json.dump(dict(config), f)

    rejected_path = args.output_path / "rejected_incs"
    rejected_path.mkdir(exist_ok=True)
    for id, config in enumerate(smac.intensifier.get_rejected_configs()):
        with (rejected_path / f"{id + 1}.json").open("w") as f:
            json.dump(dict(config), f)