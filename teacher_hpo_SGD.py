import json
import hydra
import warnings
import time
from pathlib import Path

import numpy as np
from ConfigSpace import (
    Float,
    Integer,
    Categorical,
    Configuration,
    ConfigurationSpace,
    Constant,
)
from smac import (
    MultiFidelityFacade as MFFacade,
    Scenario,
)
from smac.intensifier import Hyperband

from src.data_generator import DataGenerator, LayerwiseDataGenerator
from src.utils import HydraConfig, get_safe_original_cwd
from src.utils.general import set_seeds

from hydra.utils import get_original_cwd

warnings.filterwarnings("ignore")


class Optimizee:
    def __init__(self, cfg: HydraConfig, num_seeds: int = 3) -> None:
        self.hydra_config = cfg
        self.env_config = cfg._to_content(
            cfg, resolve=True, throw_on_missing=False
        )["env"]
        self.env_config["dataset_path"] = str(
            Path(get_original_cwd(), cfg.dataset_path)
        )
        self.num_seeds = num_seeds

    @property
    def configspace(self) -> ConfigurationSpace:
        # batches_per_epoch
        bpe: int
        if self.env_config["dataset_name"] == "MNIST":
            bpe = 12
        elif self.env_config["dataset_name"] == "FashionMNIST":
            bpe = 1200
        elif self.env_config["dataset_name"] == "CIFAR10":
            bpe = 97

        cs = ConfigurationSpace()
        initial_learning_rate = Float(
            "initial_learning_rate", (0.0001, 1.0), default=0.001, log=True
        )
        if self.hydra_config.teacher == "exponential_decay":
            decay_steps = Categorical(
                "decay_steps",
                [int(bpe / 2), bpe, int(bpe * 1.5), bpe * 2],
                default=bpe,
            )
            decay_rate = Float("decay_rate", (0.7, 0.99), default=0.9)
            # Add the parameters to configuration space
            cs.add_hyperparameters(
                [
                    initial_learning_rate,
                    decay_steps,
                    decay_rate,
                ],
            )
        elif self.hydra_config.teacher == "step_decay":
            step_size = Categorical(
                "step_size",
                [int(bpe / 2), bpe, int(bpe * 1.5), bpe * 2],
                default=bpe,
            )
            gamma = Float("gamma", (0.7, 0.99), default=0.9)
            # Add the parameters to configuration space
            cs.add_hyperparameters(
                [
                    initial_learning_rate,
                    step_size,
                    gamma,
                ],
            )
        elif self.hydra_config.teacher == "sgdr":
            T_i = Integer("T_i", (1, 4), default=1)
            T_mult = Integer("T_mult", (1, 3), default=2)
            batches_per_epoch = Constant("batches_per_epoch", bpe)
            # Add the parameters to configuration space
            cs.add_hyperparameters(
                [
                    initial_learning_rate,
                    T_i,
                    T_mult,
                    batches_per_epoch,
                ],
            )
        elif self.hydra_config.teacher == "constant":
            cs.add_hyperparameters([initial_learning_rate])

        return cs

    def train(
        self, config: Configuration, seed: int = 0, budget: int = 1
    ) -> float:
        rng = np.random.default_rng(seed)
        run_seeds = rng.integers(0, 2**32 - 1, 2)

        initial_learning_rate = config["initial_learning_rate"]

        results = []
        for _seed in run_seeds:
            print("###########################1")
            GeneratorClass = (
                LayerwiseDataGenerator
                if self.hydra_config.env.type == "LayerwiseSGD"
                else DataGenerator
            )

            teacher_name = (
                "default"
                if self.hydra_config.id == 0
                else str(self.hydra_config.id)
            )

            env_config = self.env_config.copy()
            env_config["initial_learning_rate"] = initial_learning_rate
            env_config["num_epochs"] = int(budget)
            # fix num_runs to 1 since we anyway average over n run_seeds
            env_config["num_runs"] = 1

            # since Step and Exp. decay don't feature init LR, drop it
            if (
                self.hydra_config.teacher in ["step_decay", "exponential_decay"]
                and config.get("initial_learning_rate") is not None
            ):
                config = dict(config)
                del config["initial_learning_rate"]

            teacher_config = {
                "type": self.hydra_config.teacher,
                "id": teacher_name,
                "params": config,
            }

            # Generate data for seed
            gen = GeneratorClass(
                teacher_config=teacher_config,
                env_config=env_config,
                result_dir=self.hydra_config.results_dir,
                checkpoint=0,
                seed=_seed,
                verbose=True,
            )
            gen.generate_data()
            agg_run_data = gen.exp_data.concatenate_data()

            final_evaluations = agg_run_data.groupby("run_idx").last()
            train_loss = final_evaluations["train_loss"]
            valid_loss = final_evaluations["validation_loss"]
            test_loss = final_evaluations["test_loss"]
            train_acc = final_evaluations["train_accuracy"]
            valid_acc = final_evaluations["validation_accuracy"]
            test_acc = final_evaluations["test_accuracy"]
            print(f"Train Loss: {train_loss.mean()}")
            print(f"Valid Loss: {valid_loss.mean()}")
            print(f"Test Loss: {test_loss.mean()}")
            print(f"Train Acc: {train_acc.mean()}")
            print(f"Valid Acc: {valid_acc.mean()}")
            print(f"Test Acc: {test_acc.mean()}")

            results.append(valid_loss.mean())
        print(f"Average on config {np.mean(results)} ({config})")
        return np.mean(results)


@hydra.main(config_path="hydra_conf", config_name="config", version_base="1.1")
def main(cfg: HydraConfig):
    if not cfg.results_dir:
        raise ValueError(
            "The 'results_dir' must be specified as a command-line argument or in the config file."
        )

    cfg.results_dir = Path(get_safe_original_cwd(), cfg.results_dir)

    start = time.time()

    # Get environment config
    env_config = cfg._to_content(cfg, resolve=True, throw_on_missing=False)[
        "env"
    ]
    env_config["dataset_path"] = str(
        Path(get_safe_original_cwd(), cfg.dataset_path)
    )

    print(env_config)

    optimizee = Optimizee(cfg)

    scenario = Scenario(
        optimizee.configspace,
        output_directory=cfg.results_dir,
        n_trials=200,
        min_budget=1,
        max_budget=env_config["num_epochs"],
        n_workers=1,
        deterministic=False,
        seed=cfg.seed,  #
    )

    # Incumbent is selected only based on the highest budget
    intensifier = Hyperband(
        scenario, n_seeds=1, incumbent_selection="highest budget"
    )

    smac = MFFacade(
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

    with (cfg.results_dir / "inc.json").open("w") as f:
        json.dump(dict(incumbent), f)

    lowest_val_confs = smac.runhistory.get_configs(sort_by="cost")[:15]
    lowest_val_path = cfg.results_dir / "lowest"
    lowest_val_path.mkdir(exist_ok=True)
    for id, config in enumerate(lowest_val_confs):
        with (lowest_val_path / f"{id}.json").open("w") as f:
            json.dump(dict(config), f)

    rejected_path = cfg.results_dir / "rejected_incs"
    rejected_path.mkdir(exist_ok=True)
    for id, config in enumerate(smac.intensifier.get_rejected_configs()):
        with (rejected_path / f"{id + 1}.json").open("w") as f:
            json.dump(dict(config), f)

    end = time.time()
    print(f"Took: {end-start}s to generate")


if __name__ == "__main__":
    main()
