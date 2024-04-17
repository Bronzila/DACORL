import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def find_lowest_values(df, column_name, n=10):
    final_evaluations = df.groupby("run").last()

    # Sort the DataFrame by the specified column in ascending order
    sorted_df = final_evaluations.sort_values(by=column_name)

    # Get the lowest n values from the sorted DataFrame
    return sorted_df.head(n)

def calc_mean_and_std_dev(df):
    final_evaluations = df.groupby("run").last()

    fbests = final_evaluations["f_cur"]
    return fbests.mean(), fbests.std()

def calculate_statistics(calc_mean=True, calc_lowest=True, n_lowest=1, path=None, results=True, verbose=False):
    paths = []
    if results:
        for folder_path, _, _ in os.walk(path):
            paths.extend(Path(folder_path).glob("*/eval_data.csv"))
    else:
        paths.append(path)
    # Load data
    min_mean = np.inf
    min_std = np.inf
    min_path = ""
    lowest_vals_of_min_mean = []
    for path in paths:
        incumbent_changed = False
        df = pd.read_csv(path)
        if verbose:
            print(f"Calculating for path {path}")

        if calc_mean:
            mean, std = calc_mean_and_std_dev(df)
            mean = float(f"{mean:.3e}")
            std = float(f"{std:.3e}")
            if mean < min_mean or mean == min_mean and std < min_std:
                incumbent_changed = True
                min_mean = mean
                min_std = std
                min_path = path
            if verbose:
                print(f"Mean +- Std {mean:.3e} ± {std:.3e}")
        if calc_lowest:
            lowest_vals = find_lowest_values(df, "f_cur", n_lowest)
            if incumbent_changed:
                lowest_vals_of_min_mean = lowest_vals["f_cur"]
            if verbose:
                print("Lowest values:")
                print(lowest_vals["f_cur"])
    return min_mean, min_std, lowest_vals_of_min_mean, min_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data from a CSV file.")
    parser.add_argument(
        "--path", type=str, help="Path to csv",
    )
    parser.add_argument(
        "--lowest",
        help="Get top n fbest (default: True)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="f_cur",
        help="Name of the column to analyze (default: 'f_cur')",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of fbest values to retrieve (default: 10)",
    )

    parser.add_argument(
        "--mean",
        help="Get mean and std deviation of final solutions (default: True)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--results",
        help="Find results subfolders",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose",
        help="Verbose output",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    min_mean, min_std, lowest_vals_of_min_mean, min_path = calculate_statistics(args.mean, args.lowest, args.n, args.path, args.results, args.verbose)

    print("Overall lowest statistics:")
    print(f"Found for path {min_path}")
    print("Mean +- Std {mean:.3e} ± {std:.3e}".format(mean=min_mean, std=min_std))
    print("Lowest values:")
    print(lowest_vals_of_min_mean)
