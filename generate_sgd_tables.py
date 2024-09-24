import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.calculate_sgd_statistic import calculate_multi_seed_statistics, calculate_single_seed_statistics, calculate_statistics

teacher_mapping = {
    "constant": "Constant",
    "exponential_decay": "Exp. Decay",
    "step_decay": "Step Decay",
    "sgdr": "SGDR"
}

def format_percentage(num):
    return f"{num*100:.2f}"

def format_number(num):
    if 1 <= np.abs(num) <= 999:
        return f"{num:.2f}"
    elif 0.1 <= np.abs(num) < 1:
        return f"{num:.3f}"
    else:
        formatted_num = f"{num:.2e}"
        return formatted_num.replace('e-0', 'e-').replace('e+0', 'e')

def generate_table(rows: list, format: str, metric_min: list) -> str:
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(rows[1:], columns=rows[0])
    if format == "markdown":
        return df.to_markdown(index=False)
    elif format == "latex":
        for i, j, _ in metric_min:
            df.iloc[i, j+1] = f"\\cellcolor{{highlight}}{df.iloc[i, j+1]}"
        latex_str = df.to_latex(index=False, escape=False)
        latex_str = latex_str.replace('{llll}', '{lccc}', 1)
        latex_str = latex_str.replace('{lllll}', '{lcccc}', 1)
        return latex_str
    
def generate_file_path(base_path: Path, metric: str, agent_id: str | int, format: str):
    if format == "markdown":
        suffix = "md"
    elif format == "latex":
        suffix = "tex"

    table_dir = base_path / "tables"
    table_dir.mkdir(exist_ok=True)

    return table_dir / f"{metric}_{agent_id}.{suffix}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument("--base_path", type=Path, help="Base path", default="data/SGD")
    parser.add_argument("--lowest", help="Get fbest table", action="store_true")
    parser.add_argument("--mean", help="Get mean and std deviation table", action="store_true")
    parser.add_argument("--auc", help="Get mean and std deviation table of AuC", action="store_true", default=True)
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument("--teacher", help="Specify which agents to generate the table for", nargs="+", default=["constant", "exponential_decay", "step_decay", "sgdr"])
    parser.add_argument("--agents", help="Specify which agents to generate the table for", nargs="+", default=["bc", "td3_bc", "cql", "awac", "edac", "sac_n", "lb_sac", "iql", "td3"])
    parser.add_argument("--hpo_budget", help="HPO budget used", type=int, default=30000)
    parser.add_argument("--ids", help="Specify which ids to generate tables for", nargs="+", default=[0])
    parser.add_argument("--format", help="Output format: markdown or latex", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--num_runs", help="Number of runs used for evaluation. Needed for multi-seed results in order to adjust indices correctly", type=int, default=1000)
    args = parser.parse_args()


    args.agents.insert(0, "teacher")

    pm = "$\pm$" if args.format == "latex" else "±"

    for agent_id in args.ids:
        header = [" "]
        header.extend([teacher_mapping[teacher] for teacher in args.teacher])
        rows_mean = [header]
        rows_iqm = [header]
        rows_auc = [header]
        rows_lowest = [header]

        mean_min = {j: [] for j in range(len(args.teacher))}
        iqm_min = {j: [] for j in range(len(args.teacher))}
        lowest_min = {j: [] for j in range(len(args.teacher))}
        auc_min = {j: [] for j in range(len(args.teacher))}

        teacher_mean = [np.inf] * len(args.teacher)
        teacher_iqm = [np.inf] * len(args.teacher)
        teacher_lowest = [np.inf] * len(args.teacher)
        teacher_auc = [np.inf] * len(args.teacher)

        for i, agent in enumerate(args.agents):
            agent_str = f"\\acrshort{{{agent}}}" if agent != "teacher" else agent
            row_mean = [agent_str]
            row_iqm = [agent_str]
            row_auc = [agent_str]
            row_lowest = [agent_str]

            for j, teacher in enumerate(args.teacher):
                main_path = args.base_path / teacher / str(agent_id) 
                if agent == "teacher":
                    path = main_path / "aggregated_run_data.csv"
                    mean, std, lowest, iqm, iqm_std, min_path, auc, auc_std = (
                        calculate_single_seed_statistics(
                            path=path, results=False, verbose=args.verbose, calc_auc=args.auc
                        )
                    )
                    teacher_mean[j] = mean
                    teacher_iqm[j] = iqm
                    teacher_lowest[j] = lowest.to_numpy()[0]
                    teacher_auc[j] = auc
                else:
                    path = main_path / "results" / agent
                    mean, std, lowest, iqm, iqm_std, min_path, auc, auc_std = (
                        calculate_multi_seed_statistics(
                            path=path,
                            n_iterations=args.hpo_budget,
                            results=True,
                            verbose=args.verbose,
                            num_runs=args.num_runs,
                            calc_auc=args.auc,
                        )
                    )

                if args.mean:
                    mean_str = f"{format_percentage(mean)} {pm} {format_percentage(std)}"
                    if mean > teacher_mean[j]:
                        mean_str = f"\\textbf{{{mean_str}}}"
                    row_mean.append(mean_str)
                    if mean > mean_min[j][0][2] if mean_min[j] else -np.inf:
                        mean_min[j] = [[i, j, mean]]
                    elif mean == mean_min[j][0][2]:
                        mean_min[j].append([i, j, mean])

                    iqm_str = f"{format_percentage(iqm)} {pm} {format_percentage(iqm_std)}"
                    if iqm > teacher_iqm[j]:
                        iqm_str = f"\\textbf{{{iqm_str}}}"
                    row_iqm.append(iqm_str)
                    if iqm > iqm_min[j][0][2] if iqm_min[j] else -np.inf:
                        iqm_min[j] = [[i, j, iqm]]
                    elif iqm == iqm_min[j][0][2]:
                        iqm_min[j].append([i, j, iqm])

                if args.lowest:
                    lowest_val = lowest.to_numpy()[0]
                    lowest_str = f"{format_percentage(lowest_val)}"
                    if lowest_val > teacher_lowest[j]:
                        lowest_str = f"\\textbf{{{lowest_str}}}"
                    row_lowest.append(lowest_str)
                    if lowest_val > lowest_min[j][0][2] if lowest_min[j] else -np.inf:
                        lowest_min[j] = [[i, j, lowest_val]]
                    elif lowest_val == lowest_min[j][0][2]:
                        lowest_min[j].append([i, j, lowest_val])

                if args.auc:
                    auc_str = f"{format_number(auc)} {pm} {format_number(auc_std)}"
                    if auc > teacher_auc[j]:
                        auc_str = f"\\textbf{{{auc_str}}}"
                    row_auc.append(auc_str)
                    if auc > auc_min[j][0][2] if auc_min[j] else -np.inf:
                        auc_min[j] = [[i, j, auc]]
                    elif auc == auc_min[j][0][2]:
                        auc_min[j].append([i, j, auc])

            if args.mean:
                rows_mean.append(row_mean)
                rows_iqm.append(row_iqm)
            if args.lowest:
                rows_lowest.append(row_lowest)
            if args.auc:
                rows_auc.append(row_auc)

        if args.mean:
            # Regular mean
            table_result_path = generate_file_path(args.base_path, "mean", agent_id, args.format)
            table_content = generate_table(rows_mean, args.format, [item for sublist in mean_min.values() for item in sublist])
            with table_result_path.open("w") as f:
                f.write(table_content)
            # IQM
            table_result_path = generate_file_path(args.base_path, "iqm", agent_id, args.format)
            table_content = generate_table(rows_iqm, args.format, [item for sublist in iqm_min.values() for item in sublist])
            with table_result_path.open("w") as f:
                f.write(table_content)
        if args.lowest:
            table_result_path = generate_file_path(args.base_path, "lowest", agent_id, args.format)
            table_content = generate_table(rows_lowest, args.format, [item for sublist in lowest_min.values() for item in sublist])
            with table_result_path.open("w") as f:
                f.write(table_content)
        if args.auc:
            table_result_path = generate_file_path(args.base_path, "auc", agent_id, args.format)
            table_content = generate_table(rows_auc, args.format, [item for sublist in auc_min.values() for item in sublist])
            with table_result_path.open("w") as f:
                f.write(table_content)
