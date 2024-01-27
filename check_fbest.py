import argparse
import os
import re

import numpy as np
import pandas as pd


def find_lowest_values(df, column_name, n=10):
    final_evaluations = df[df["batch"] == 99]

    # Sort the DataFrame by the specified column in ascending order
    sorted_df = final_evaluations.sort_values(by=column_name)

    # Get the lowest n values from the sorted DataFrame
    return sorted_df.head(n)

def calc_mean_and_std_dev(df):
    final_evaluations = df[df["batch"] == 99]

    fbests = final_evaluations["f_cur"]
    return fbests.mean(), fbests.std()


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
        default=10,
        help="Number of fbest values to retrieve (default: 10)",
    )

    parser.add_argument(
        "--mean",
        help="Get mean and std deviation of final solutions (default: True)",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.path)

    if args.mean:
        mean, std = calc_mean_and_std_dev(df)
        print("Mean +- Std {mean:.3e} ± {std:.3e}".format(mean=mean, std=std))
    if args.lowest:
        lowest_vals = find_lowest_values(df, "f_cur")
        print("Lowest values:")
        print(lowest_vals[args.column_name])