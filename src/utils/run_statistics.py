from __future__ import annotations

from pathlib import Path

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
        df["run"] += idx * num_runs
        combined_run_data.append(df)
    return pd.concat(combined_run_data, ignore_index=True)


def find_lowest_values(
    df: pd.DataFrame,
    column_name: str,
    n: int = 10,
) -> pd.DataFrame:
    final_evaluations = df.groupby("run").last()

    # Sort the DataFrame by the specified column in ascending order
    sorted_df = final_evaluations.sort_values(by=column_name)

    # Get the lowest n values from the sorted DataFrame
    return sorted_df.head(n)


def calc_mean_and_std_dev(
    df: pd.DataFrame,
    metric: str = "f_cur",
) -> tuple[float, float]:
    final_evaluations = (
        df.sort_values(by=["run", "batch"]).groupby("run").last()
    )

    fbests = final_evaluations[metric]
    return fbests.mean(), fbests.std()


def calculate_single_seed_statistics(
    path: Path,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    n_lowest: int = 1,
    results: bool = True,
    verbose: bool = False,
    interpolation: bool = False,
    calc_auc: bool = True,
    metric: str = "f_cur",
) -> tuple:
    filename = (
        "eval_data_interpolation.csv" if interpolation else "eval_data.csv"
    )
    paths = list(path.rglob(f"*/{filename}")) if results else [path]
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
            mean, std = calc_mean_and_std_dev(df, metric)
            mean = float(f"{mean:.2e}")
            std = float(f"{std:.2e}")
            if mean < min_mean or mean == min_mean and std < min_std:
                incumbent_changed = True
                min_mean = mean
                min_std = std
                min_path = path
                min_iqm, min_iqm_std = compute_iqm(df, metric)
                min_iqm = float(f"{min_iqm:.2e}")
                min_iqm_std = float(f"{min_iqm_std:.2e}")
                if calc_auc:
                    min_auc, min_auc_std = compute_auc(df, metric)
                    min_auc = float(f"{min_auc:.2e}")
                    min_auc_std = float(f"{min_auc_std:.2e}")
                else:
                    min_auc = 0
                    min_auc_std = 0
            if verbose:
                print(f"Mean +- Std {mean:.2e} ± {std:.2e}")
        if calc_lowest:
            lowest_vals = find_lowest_values(df, metric, n_lowest)
            if incumbent_changed:
                lowest_vals_of_min_mean = lowest_vals[metric]
            if verbose:
                print("Lowest values:")
                print(lowest_vals[metric])
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
    path: Path,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    n_lowest: int = 1,
    results: bool = True,
    verbose: bool = False,
    num_runs: int = 100,
    interpolation: bool = False,
    calc_auc: bool = True,
    metric: str = "f_cur",
) -> tuple:
    seed_dirs: set[Path] = set()
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

    print(seed_dirs)
    best_iterations_paths = []
    # This is only used if there are multiple checkpoints in the seed directory --> choose the best one
    for seed_dir in seed_dirs:
        (
            min_mean,
            min_std,
            _,
            _,
            _,
            min_path,
            _,
            _,
        ) = calculate_single_seed_statistics(
            seed_dir,
            calc_mean,
            calc_lowest,
            n_lowest,
            results,
            verbose,
            interpolation,
            calc_auc,
            metric,
        )
        if verbose:
            print(f"Minimum mean {min_mean} +- {min_std} for path {min_path}")
        best_iterations_paths.append(min_path)
    combined_data = combine_run_data(best_iterations_paths, num_runs=num_runs)
    if calc_mean:
        mean, std = calc_mean_and_std_dev(combined_data, metric)
        mean = float(f"{mean:.2e}")
        std = float(f"{std:.2e}")
        iqm, iqm_std = compute_iqm(combined_data, metric)
        iqm = float(f"{iqm:.2e}")
        iqm_std = float(f"{iqm_std:.2e}")
    if calc_lowest:
        lowest_vals = find_lowest_values(combined_data, metric, n_lowest)[
            metric
        ]
    if calc_auc:
        auc, auc_std = compute_auc(combined_data, metric)
        auc = float(f"{auc:.2e}")
        auc_std = float(f"{auc_std:.2e}")
    else:
        auc = 0
        auc_std = 0
    return (
        mean,
        std,
        lowest_vals,
        iqm,
        iqm_std,
        0,
        auc,
        auc_std,
    )  # path doesnt really matter here


def calculate_statistics(
    path: Path,
    calc_mean: bool = True,
    calc_lowest: bool = True,
    n_lowest: int = 1,
    results: bool = True,
    verbose: bool = False,
    multi_seed: bool = False,
    num_runs: int = 100,
    interpolation: bool = False,
    calc_auc: bool = True,
    metric: str = "f_cur",
) -> tuple:
    if multi_seed:
        return calculate_multi_seed_statistics(
            path,
            calc_mean,
            calc_lowest,
            n_lowest,
            results,
            verbose,
            num_runs,
            interpolation,
            calc_auc,
            metric,
        )
    return calculate_single_seed_statistics(
        path,
        calc_mean,
        calc_lowest,
        n_lowest,
        results,
        verbose,
        interpolation,
        calc_auc,
        metric,
    )


def compute_iqm(df: pd.DataFrame, metric: str = "f_cur") -> tuple[float, float]:
    final_evaluations = df.groupby("run").last()
    df_sorted = final_evaluations.sort_values(by=metric)

    # Calculate the number of rows representing 25% of the DataFrame
    num_rows = len(df_sorted)
    num_to_remove = int(0.25 * num_rows)

    # Remove the upper and lower 25% of the DataFrame
    df_trimmed = df_sorted[num_to_remove:-num_to_remove]
    fbests = df_trimmed[metric]
    return fbests.mean(), fbests.std()


def compute_auc(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    def fill_missing_values(group: pd.Series) -> pd.Series:
        last_value = group[metric].iloc[-1]

        # Create full run
        all_steps = pd.DataFrame({"batch": range(101)})

        # Merge group with full run
        filled_group = all_steps.merge(group, on="batch", how="left")

        filled_group["run"] = group["run"].iloc[0]
        filled_group[metric] = filled_group[metric].fillna(last_value)

        return filled_group

    required_fields_df = df[[metric, "run", "batch"]]

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
        auc = np.trapz(run[metric], run["batch"])
        return pd.Series({"run": run["run"].iloc[0], "auc": auc})

    auc_per_run = (
        df_filled.groupby("run").apply(calculate_auc).reset_index(drop=True)
    )

    mean_auc = auc_per_run["auc"].mean()
    std_auc = auc_per_run["auc"].std()

    return mean_auc, std_auc


def compute_prob_outperformance(
    df_teacher: pd.DataFrame,
    df_agent: pd.DataFrame,
) -> float:
    final_evaluations_teacher = df_teacher.groupby("run").last()["f_cur"]
    final_evaluations_agent = df_agent.groupby("run").last()["f_cur"]

    assert len(final_evaluations_agent) == len(final_evaluations_teacher)

    p = 0
    for i in range(len(final_evaluations_agent)):
        if final_evaluations_agent[i] < final_evaluations_teacher[i]:
            p += 1

    return p / len(final_evaluations_agent)
