import argparse
from pathlib import Path

from src.utils.run_statistics import calculate_statistics
from tap import Tap

if __name__ == "__main__":

    class CalculateStatisticsParser(Tap):
        csv_path: Path
        n_iterations: int  # Select training budget
        calc_lowest: bool = False
        calc_mean: bool = False
        calc_auc: bool = False
        results: bool = False
        verbose: bool = False
        multi_seed: bool = False
        num_runs: int = 100
        objective: str = "f_cur"  # Name of the column to analyze

    args = CalculateStatisticsParser().parse_args()

    min_mean, min_std, lowest_vals_of_min_mean, min_path = calculate_statistics(
        args.path,
        args.calc_mean,
        args.calc_lowest,
        args.calc_auc,
        args.n_iterations,
        args.results,
        args.verbose,
        args.multi_seed,
        args.num_runs,
        args.objective,
    )

    print("Overall lowest statistics:")
    print(f"Found for path {min_path}")
    print(
        "Mean +- Std {mean:.3e} ± {std:.3e}".format(mean=min_mean, std=min_std)
    )
    print("Lowest values:")
    print(lowest_vals_of_min_mean)
