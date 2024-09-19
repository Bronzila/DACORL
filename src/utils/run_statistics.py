from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.replay_buffer import ReplayBuffer


def combine_runs(agent_paths):
    combined_buffer = None
    combined_run_info = None
    combined_run_data = []
    for idx, root_path in enumerate(agent_paths):
        replay_path = Path(root_path, "rep_buffer")
        run_info_path = Path(root_path, "run_info.json")
        run_data_path = Path(root_path, "aggregated_run_data.csv")

        with run_info_path.open(mode="rb") as f:
            run_info = json.load(f)
        temp_buffer = ReplayBuffer.load(replay_path)

        if combined_buffer is None and combined_run_info is None:
            combined_buffer = temp_buffer
            combined_run_info = {
                "environment": run_info["environment"],
                "starting_points": run_info["starting_points"],
                "seed": run_info["seed"],
                "num_runs": run_info["num_runs"],
                "num_batches": run_info["num_batches"],
                "agent": {"type": run_info["agent"]["type"]},
            }
        else:
            combined_buffer.merge(temp_buffer)

        df = pd.read_csv(run_data_path)
        df["run"] += idx * run_info["num_runs"]
        combined_run_data.append(df)
    return (
        combined_buffer,
        combined_run_info,
        pd.concat(combined_run_data, ignore_index=True),
    )


def combine_run_data(
    data_paths: list[str],
    num_runs: int = 100,
) -> pd.DataFrame:
    combined_run_data = []
    for idx, root_path in enumerate(data_paths):
        print(root_path)
        run_data_path = Path(root_path)
        try:
            df = pd.read_csv(run_data_path)
        except pd.errors.EmptyDataError:
            warnings.warn(
                f"The following data is corrupted and could not be loaded: {run_data_path}",
            )
            continue
        df["run"] += idx * num_runs
        combined_run_data.append(df)
    return pd.concat(combined_run_data, ignore_index=True)


def find_lowest_values(df, column_name, n=10):
    final_evaluations = df.groupby("run").last()

    # Sort the DataFrame by the specified column in ascending order
    sorted_df = final_evaluations.sort_values(by=column_name)

    # Get the lowest n values from the sorted DataFrame
    return sorted_df.head(n)


def calc_mean_and_std_dev(df: pd.DataFrame, objective: str) -> float:
    final_evaluations = df.groupby("run").last()

    fbests = final_evaluations[objective]
    return fbests.mean(), fbests.std()


def calculate_single_seed_statistics(
    objective: str,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    calc_auc: bool = True,
    n_lowest: int = 1,
    path: str | None | None = None,
    results: bool = True,
    verbose: bool = False,
) -> tuple[float, float, float, float, float, Path]:
    paths = []
    if results:
        for folder_path, _, _ in os.walk(path):
            paths.extend(Path(folder_path).glob("*/eval_data.csv"))
    else:
        paths.append(path)
    # Load data
    min_mean = np.inf
    min_std = np.inf
    min_iqm = np.inf
    min_iqm_std = np.inf
    min_auc = np.inf
    min_auc_std = np.inf
    min_path = ""
    lowest_vals_of_min_mean = []
    assert (
        len(paths) == 1
    ), "There are more than 1 paths given. Are you sure, you want to only use the lowest checkpoint?"
    for path in paths:
        incumbent_changed = False
        df = pd.read_csv(path)
        if verbose:
            print(f"Calculating for path {path}")

        if calc_mean:
            mean, std = calc_mean_and_std_dev(df, objective)
            mean = float(f"{mean:.3e}")
            std = float(f"{std:.3e}")
            if mean < min_mean or mean == min_mean and std < min_std:
                incumbent_changed = True
                min_mean = mean
                min_std = std
                min_path = path
                min_iqm, min_iqm_std = compute_iqm(df, objective)
                if calc_auc:
                    min_auc, min_auc_std = compute_AuC(df, objective)
            if verbose:
                print(f"Mean +- Std {mean:.3e} ± {std:.3e}")
        if calc_lowest:
            lowest_vals = find_lowest_values(df, objective, n_lowest)
            if incumbent_changed:
                lowest_vals_of_min_mean = lowest_vals[objective]
            if verbose:
                print("Lowest values:")
                print(lowest_vals[objective])
    return (
        min_mean,
        min_std,
        lowest_vals_of_min_mean,
        min_iqm,
        min_iqm_std,
        min_path,
        min_auc,
        min_auc_std,
    )


def calculate_multi_seed_statistics(
    objective: str,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    calc_auc: bool = True,
    n_iterations: int = 15000,
    n_lowest: int = 1,
    path: str | None = None,
    results: bool = True,
    verbose: bool = False,
    num_runs: int = 100,
) -> tuple[float, float, float, float, float, Path]:
    # TODO here we currently assume, that we only have one training
    # folder and eval file in results/td3_bc/<seed>/
    paths = []
    if results:
        for seed_path in Path(path).rglob(f"*{n_iterations}/eval_data.csv"):
            paths.append(seed_path)
    else:
        paths.append(path)

    combined_data = combine_run_data(paths, num_runs=num_runs)
    if calc_mean:
        mean, std = calc_mean_and_std_dev(combined_data, objective)
        mean = float(f"{mean:.3e}")
        std = float(f"{std:.3e}")
        iqm, iqm_std = compute_iqm(combined_data, objective)
    if calc_auc:
        auc, auc_std = compute_AuC(combined_data)
    if calc_lowest:
        lowest_vals = find_lowest_values(combined_data, objective, n_lowest)[
            objective
        ]
    return (
        mean,
        std,
        lowest_vals,
        iqm,
        iqm_std,
        paths[0],
        auc,
        auc_std,
    )  # path doesnt really matter here


def calculate_statistics(
    calc_mean: bool = True,
    calc_lowest: bool = True,
    n_iterations: int = 15000,
    n_lowest: int = 1,
    path: str | None = None,
    results: bool = True,
    verbose: bool = False,
    multi_seed: bool = False,
    num_runs: int = 100,
    objective: str = "f_cur",
) -> tuple[float, float, float, float, float, Path]:
    if multi_seed:
        return calculate_multi_seed_statistics(
            objective,
            calc_mean,
            calc_lowest,
            n_iterations,
            n_lowest,
            path,
            results,
            verbose,
            num_runs,
        )
    else:
        return calculate_single_seed_statistics(
            objective,
            calc_mean,
            calc_lowest,
            n_lowest,
            path,
            results,
            verbose,
        )


def compute_iqm(df: pd.DataFrame, objective: str):
    final_evaluations = df.groupby("run").last()
    df_sorted = final_evaluations.sort_values(by=objective)

    # Calculate the number of rows representing 25% of the DataFrame
    num_rows = len(df_sorted)
    num_to_remove = int(0.25 * num_rows)

    # Remove the upper and lower 25% of the DataFrame
    df_trimmed = df_sorted[num_to_remove:-num_to_remove]
    fbests = df_trimmed[objective]
    return fbests.mean(), fbests.std()


def compute_AuC(df: pd.DataFrame, objective: str) -> tuple[float, float]:
    def fill_missing_values(group):
        last_value = group[objective].iloc[-1]

        # Create full run
        all_steps = pd.DataFrame({"batch": range(101)})

        # Merge group with full run
        filled_group = pd.merge(all_steps, group, on="batch", how="left")

        filled_group["run"] = group["run"].iloc[0]
        filled_group[objective] = filled_group[objective].fillna(last_value)

        return filled_group

    required_fields_df = df[[objective, "run", "batch"]]

    df_filled = (
        required_fields_df.groupby("run")
        .apply(fill_missing_values)
        .reset_index(drop=True)
    )

    # Sort by run and batch to ensure order
    df_filled = df_filled.sort_values(by=["run", "batch"]).reset_index(
        drop=True,
    )

    def calculate_auc(run):
        auc = np.trapz(run[objective], run["batch"])
        return pd.Series({"run": run["run"].iloc[0], "auc": auc})

    auc_per_run = (
        df_filled.groupby("run").apply(calculate_auc).reset_index(drop=True)
    )

    mean_auc = auc_per_run["auc"].mean()
    std_auc = auc_per_run["auc"].std()

    return mean_auc, std_auc
