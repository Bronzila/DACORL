import argparse
from pathlib import Path
from smac import Scenario, HyperparameterOptimizationFacade as HPOFacade
from utils.general import set_seeds

from train_hpo import Optimizee


def run_hpo(data_dir: Path) -> tuple:
    set_seeds(0)

    optimizee = Optimizee(
        data_dir=data_dir,
        agent_type="td3_bc",
        debug=False,
        eval_protocol="train",
        eval_seed=0,
    )

    output_path = Path("data/test_smac")
    scenario = Scenario(
        optimizee.configspace,
        output_directory=output_path,
        n_trials=50,
        n_workers=1,
        deterministic=False,
    )

    intensifier = HPOFacade.get_intensifier(scenario, max_config_calls=5)
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

    return incumbent, smac.validate(incumbent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPO for any agent")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
    )
    args = parser.parse_args()

    incumbent1, result1 = run_hpo(data_dir=args.data_dir)
    incumbent2, result2 = run_hpo(data_dir=args.data_dir)

    print(f"Result 1: {result1}, Result 2: {result2}")
    print(incumbent1)
    print(incumbent2)