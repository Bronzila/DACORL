import argparse
import re
from pathlib import Path

from markdown_table_generator import generate_markdown, table_from_string_list

from src.utils.general import calculate_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument(
        "--path", type=str, help="Base path",
        default="data/ToySGD"
    )
    parser.add_argument(
        "--baseline", type=str, help="Baseline path",
        default=""
    )
    parser.add_argument(
        "--custom_path", type=str, help="Base path",
    )
    parser.add_argument(
        "--verbose",
        help="Verbose output",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--agents",
        help="Specify which agents to generate the table for",
        type=str,
        nargs="*",
        default=["exponential_decay", "step_decay", "sgdr", "constant"],
    )
    parser.add_argument(
        "--functions",
        help="Specify which functions to generate the table for",
        type=str,
        nargs="*",
        default=["Ackley", "Rastrigin", "Rosenbrock", "Sphere"],
    )
    parser.add_argument(
        "--id",
        help="Specify which ids to generate tables for",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--num_runs",
        help="Number of runs used for evaluation. Needed for multi-seed results in order to adjust indeces correctly",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--interpolation",
        help="Whether interpolation protocol has been used to evaluate agents",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--lowest",
        help="Calculate for lowest",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--mean",
        help="Calculate for mean",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--iqm",
        help="Calculate for iqm",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--auc",
        help="Calculate for AuC",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    hetero = False
    base_path = Path(args.path)
    if args.custom_path:
        hetero = True
        base_path = Path(args.custom_path)

    if args.baseline:
        baseline = Path(args.baseline)
    else:
        baseline = base_path

    mean_better_counter = 0
    iqm_better_counter = 0
    lowest_better_counter = 0
    auc_better_counter = 0

    general_counter = 0

    for teacher in args.agents:
        for function in args.functions:
            if hetero:
                agent_path = base_path / teacher / function / "results"
                baseline_path = baseline / teacher / function / "aggregated_run_data.csv"
            else:
                agent_path = base_path / teacher / str(args.id) / function / "results"
                baseline_path = baseline / teacher / str(args.id) / function / "aggregated_run_data.csv"

            mean, std, lowest, iqm, iqm_std, _, auc, auc_std = calculate_statistics(path=agent_path, results=True, verbose=args.verbose, multi_seed=True, num_runs=args.num_runs, interpolation=args.interpolation)
            lowest = lowest.to_numpy()[0]
            baseline_mean, baseline_std, baseline_lowest, baseline_iqm, baseline_iqm_std, _, baseline_auc, baseline_auc_std = calculate_statistics(path=baseline_path, results=False, verbose=args.verbose, multi_seed=False, num_runs=args.num_runs, interpolation=args.interpolation)
            baseline_lowest = baseline_lowest.to_numpy()[0]
            print(f"{teacher}/{function}")
            print("Agent")
            print(mean)
            print("Baseline")
            print(baseline_mean)
            if args.mean:
                if (mean < baseline_mean) or (mean == baseline_mean and std < baseline_std):
                    mean_better_counter += 1
            
            if args.iqm:
                if (iqm < baseline_iqm) or (iqm == baseline_iqm and iqm_std < baseline_iqm_std):
                    iqm_better_counter += 1

            if args.lowest:
                if lowest < baseline_lowest:
                    lowest_better_counter += 1

            if args.auc:
                if (auc < baseline_auc) or (auc == baseline_auc and auc_std < baseline_auc_std):
                    auc_better_counter += 1
            
            general_counter += 1
    
    if args.mean:
        print(f"Mean better for {mean_better_counter}/{general_counter}:")
        print(f"{(mean_better_counter / general_counter) * 100}%")

    if args.iqm:
        print(f"IQM better for {iqm_better_counter}/{general_counter}:")
        print(f"{(iqm_better_counter / general_counter) * 100}%")

    if args.lowest:
        print(f"Lowest better for {lowest_better_counter}/{general_counter}:")
        print(f"{(lowest_better_counter / general_counter) * 100}%")

    if args.auc:
        print(f"AuC better for {auc_better_counter}/{general_counter}:")
        print(f"{(auc_better_counter / general_counter) * 100}%")