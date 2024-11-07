from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def combine_run_data(
    data_paths: list[Path],
    num_runs: int = 100,
) -> pd.DataFrame:
    combined_run_data = []
    for idx, root_path in enumerate(data_paths):
        run_data_path = Path(root_path)
        df = pd.read_csv(run_data_path)
        df["run_idx"] += idx * num_runs
        combined_run_data.append(df)
    return pd.concat(combined_run_data, ignore_index=True)


def find_highest_values(
    df: pd.DataFrame,
    column_name: str,
    n: int = 10,
) -> pd.DataFrame:
    final_evaluations = df.groupby("run_idx").last()

    # Sort the DataFrame by the specified column in ascending order
    sorted_df = final_evaluations.sort_values(by=column_name, ascending=False)

    # Get the lowest n values from the sorted DataFrame
    return sorted_df.head(n)


def calc_mean_and_std_dev(
    df: pd.DataFrame,
    metric: str = "test_accuracy",
) -> tuple[float, float]:
    final_evaluations = (
        df.sort_values(by=["run_idx", "batch_idx"]).groupby("run_idx").last()
    )

    fbests = final_evaluations[metric]
    return fbests.mean(), fbests.std()


def calculate_single_seed_statistics(
    path: Path,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    calc_auc: bool = True,
    n_lowest: int = 1,
    results: bool = True,
    interpolation: bool = False,
    metric: str = "test_accuracy",
    verbose: bool = False,
) -> tuple[float, float, list[float], float, float, Path, float, float]:
    paths: list[Path] = []
    filename = (
        "eval_data_interpolation.csv" if interpolation else "eval_data.csv"
    )
    if results:
        for folder_path, _, _ in os.walk(path):
            paths.extend(Path(folder_path).glob(f"*/{filename}"))
    else:
        paths.append(path)
    # Load data
    max_mean = -1.0
    max_std = -1.0
    max_iqm = -1.0
    max_iqm_std = -1.0
    max_auc = -1.0
    max_auc_std = -1.0
    max_path = Path()
    lowest_vals_of_max_mean: list[float] = []
    assert (
        len(paths) == 1
    ), "There are more than 1 paths given. Are you sure, you want to only use the lowest checkpoint?"
    for path in paths:
        incumbent_changed = False
        df = pd.read_csv(path)
        if verbose:
            print(f"Calculating for path {path}")
        if calc_mean:
            mean, std = calc_mean_and_std_dev(df, metric)
            mean = float(f"{mean:.3e}")
            std = float(f"{std:.3e}")
            if mean > max_mean or mean == max_mean and std < max_std:
                incumbent_changed = True
                max_mean = mean
                max_std = std
                max_path = path
                max_iqm, max_iqm_std = compute_iqm(df, metric)
                max_iqm = float(f"{max_iqm:.3e}")
                max_iqm_std = float(f"{max_iqm_std:.3e}")
                if calc_auc:
                    max_auc, max_auc_std = compute_auc(df, metric)
                    max_auc = float(f"{max_auc:.3e}")
                    max_auc_std = float(f"{max_auc_std:.3e}")
                else:
                    max_auc, max_auc_std = (0, 0)
            if verbose:
                print(f"Mean +- Std {mean:.3e} ± {std:.3e}")
        if calc_lowest:
            lowest_vals = find_highest_values(df, metric, n_lowest)
            if incumbent_changed:
                lowest_vals_of_max_mean = list(lowest_vals[metric])
            if verbose:
                print("Lowest values:")
                print(lowest_vals[metric])
    return (
        max_mean,
        max_std,
        lowest_vals_of_max_mean,
        max_iqm,
        max_iqm_std,
        max_path,
        max_auc,
        max_auc_std,
    )


def calculate_multi_seed_statistics(
    path: Path,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    calc_auc: bool = True,
    n_lowest: int = 1,
    results: bool = True,
    num_runs: int = 100,
    interpolation: bool = False,
    metric: str = "test_accuracy",
    verbose: bool = False,
) -> tuple[float, float, pd.Series, float, float, Path, float, float]:
    seed_dirs = set()
    filename = (
        "eval_data_interpolation.csv" if interpolation else "eval_data.csv"
    )
    if results:
        for eval_file in Path(path).rglob(filename):
            # Extract the seed directory
            seed_dir = eval_file.parents[1]
            seed_dirs.add(seed_dir)
    else:
        seed_dirs.add(path)

    if len(seed_dirs) < 5:
        print("Some runs failed")
    print(seed_dirs)
    best_iterations_paths = []
    # This is only used if there are multiple checkpoints in the seed directory --> choose the best one
    for seed_dir in seed_dirs:
        (
            max_mean,
            max_std,
            _,
            _,
            _,
            max_path,
            _,
            _,
        ) = calculate_single_seed_statistics(
            seed_dir,
            calc_mean,
            calc_lowest,
            calc_auc,
            n_lowest,
            results,
            interpolation,
            metric,
            verbose,
        )
        if verbose:
            print(f"Minimum mean {max_mean} +- {max_std} for path {max_path}")
        best_iterations_paths.append(max_path)
    combined_data = combine_run_data(best_iterations_paths, num_runs=num_runs)
    if calc_mean:
        mean, std = calc_mean_and_std_dev(combined_data, metric)
        mean = float(f"{mean:.3e}")
        std = float(f"{std:.3e}")
        iqm, iqm_std = compute_iqm(combined_data, metric)
        iqm = float(f"{iqm:.3e}")
        iqm_std = float(f"{iqm_std:.3e}")
    if calc_lowest:
        lowest_vals = find_highest_values(combined_data, metric, n_lowest)[
            metric
        ]
    if calc_auc:
        auc, auc_std = compute_auc(combined_data, metric)
        auc = float(f"{auc:.3e}")
        auc_std = float(f"{auc_std:.3e}")
    else:
        auc, auc_std = (0, 0)
    return (
        mean,
        std,
        lowest_vals,
        iqm,
        iqm_std,
        Path(),
        auc,
        auc_std,
    )  # path doesnt really matter here


def calculate_statistics(
    path: Path,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    calc_auc: bool = True,
    n_lowest: int = 1,
    results: bool = True,
    multi_seed: bool = False,
    num_runs: int = 100,
    interpolation: bool = False,
    metric: str = "test_accuracy",
    verbose: bool = False,
) -> tuple:
    if multi_seed:
        return calculate_multi_seed_statistics(
            path,
            calc_mean,
            calc_lowest,
            calc_auc,
            n_lowest,
            results,
            num_runs,
            interpolation,
            metric,
            verbose,
        )
    return calculate_single_seed_statistics(
        path,
        calc_mean,
        calc_lowest,
        calc_auc,
        n_lowest,
        results,
        interpolation,
        metric,
        verbose,
    )


def compute_iqm(
    df: pd.DataFrame,
    metric: str = "test_accuracy",
) -> tuple[float, float]:
    final_evaluations = df.groupby("run_idx").last()
    df_sorted = final_evaluations.sort_values(by=metric)

    # Calculate the number of rows representing 25% of the DataFrame
    num_rows = len(df_sorted)
    num_to_remove = int(0.25 * num_rows)

    # Remove the upper and lower 25% of the DataFrame
    df_trimmed = df_sorted[num_to_remove:-num_to_remove]
    fbests = df_trimmed[metric]
    return fbests.mean(), fbests.std()


def compute_auc(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    def fill_missing_values(group: pd.DataFrame) -> pd.DataFrame:
        last_value = group[metric].iloc[-1]

        # Create full run
        all_steps = pd.DataFrame({"batch_idx": range(101)})

        # Merge group with full run
        filled_group = all_steps.merge(group, on="batch_idx", how="left")

        filled_group["run_idx"] = group["run_idx"].iloc[0]
        filled_group[metric] = filled_group[metric].fillna(last_value)

        return filled_group

    # Metric has to be validationation accuracy as test accuracy is only computed at end of epoch
    metric = "validation_accuracy"
    required_fields_df = df[[metric, "run_idx", "batch_idx"]]

    # No need to fill values for SGD since we do not terminate early
    df_filled = required_fields_df

    # Sort by run and batch to ensure order
    df_filled = df_filled.sort_values(by=["run_idx", "batch_idx"]).reset_index(
        drop=True,
    )

    def calculate_auc(run: pd.DataFrame) -> Any:
        auc = np.trapz(run[metric], run["batch_idx"])
        return pd.Series({"run_idx": run["run_idx"].iloc[0], "auc": auc})

    auc_per_run = (
        df_filled.groupby("run_idx").apply(calculate_auc).reset_index(drop=True)
    )

    mean_auc = auc_per_run["auc"].mean()
    std_auc = auc_per_run["auc"].std()

    return mean_auc, std_auc


def compute_prob_outperformance(
    df_teacher: pd.DataFrame,
    df_agent: pd.DataFrame,
) -> float:
    final_evaluations_teacher = df_teacher.groupby("run_idx").last()["f_cur"]
    final_evaluations_agent = df_agent.groupby("run_idx").last()["f_cur"]

    assert len(final_evaluations_agent) == len(final_evaluations_teacher)

    p = 0
    for i in range(len(final_evaluations_agent)):
        if final_evaluations_agent[i] < final_evaluations_teacher[i]:
            p += 1

    return p / len(final_evaluations_agent)
