import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.general import (
    calculate_multi_seed_statistics,
    calculate_single_seed_statistics,
)
from markdown_table_generator import generate_markdown, table_from_string_list

def generate_table(rows: list, format: str):
    if format == "markdown":
        table = table_from_string_list(rows)
        return generate_markdown(table)
    elif format == "latex":
        df = pd.DataFrame(rows[1:], columns=rows[0])
        filepath = Path('folder/subfolder/out.csv') 
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)  
        return df.style.to_latex(index=False, escape=False)

def generate_file_path(base_path: Path, metric: str, function: str, agent_id: str | int, format: str):
    if format == "markdown":
        suffix = "md"
    elif format == "latex":
        suffix = "tex"

    table_dir = base_path / "tables"
    table_dir.mkdir(exist_ok=True)

    return table_dir / f"{metric}_{function}_{agent_id}.{suffix}"

def calculate_percentage_change(reference, current):
    return ((current - reference) / abs(reference)) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument(
        "--path", type=str, help="Base path", default="data/ToySGD"
    )
    parser.add_argument(
        "--lowest",
        help="Get fbest table",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--mean",
        help="Get mean and std deviation table",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose",
        help="Verbose output",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--teacher",
        help="Specify which agents to generate the table for",
        nargs="+",
        default=["exponential_decay", "step_decay", "sgdr", "constant"],
    )
    parser.add_argument(
        "--agents",
        help="Specify which agents to generate the table for",
        nargs="+",
        default=["bc", "td3_bc", "cql", "awac", "edac", "sac_n", "lb_sac", "iql", "td3"],
    )
    parser.add_argument(
        "--functions",
        help="Specify which functions to generate the table for",
        nargs="+",
        default=["Ackley", "Rastrigin", "Rosenbrock", "Sphere"],
    )
    parser.add_argument(
        "--hpo_budget",
        help="HPO budget used",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--ids",
        help="Specify which ids to generate tables for",
        nargs="+",
        default=[0],
    )
    parser.add_argument(
        "--format",
        help="Output format: markdown or latex",
        choices=["markdown", "latex"],
        default="markdown",
    )
    parser.add_argument(
        "--num_runs",
        help="Number of runs used for evaluation. Needed for multi-seed results in order to adjust indeces correctly",
        type=int,
        default=1000
    )
    args = parser.parse_args()

    args.agents.insert(0, "teacher")
    base_path = Path(args.path)
    if base_path.name == "ToySGD":
        objective = "f_cur"
    elif base_path.name == "SGD":
        objective = "valid_acc"
    elif base_path.name == "CMAES":
        objective = "f_cur"
    else:
        raise NotImplementedError(f"Currently the benchmark {base_path.name} is not implemented.")

    for function in args.functions:
        for agent_id in args.ids:
            header = [None]
            header.extend(args.teacher)
            rows_mean = [header]
            rows_iqm = [header]
            rows_lowest = [header]
            for i, agent in enumerate(args.agents):
                row_mean = [agent]
                row_iqm = [agent]
                row_lowest = [agent]
                for j, teacher in enumerate(args.teacher):
                    if agent == "teacher":
                        path = base_path / teacher / str(agent_id) / function / "aggregated_run_data.csv"
                        mean, std, lowest, iqm, iqm_std, min_path = (
                            calculate_single_seed_statistics(
                                objective=objective, path=path, results=False, verbose=args.verbose
                            )
                        )
                    else:
                        path = base_path / teacher / str(agent_id) / function / "results" / agent
                        mean, std, lowest, iqm, iqm_std, min_path = (
                            calculate_multi_seed_statistics(
                                objective=objective,
                                path=path,
                                n_iterations=args.hpo_budget,
                                results=True,
                                verbose=args.verbose,
                                num_runs=args.num_runs,
                            )
                        )

                    if args.mean:
                        if mean is None:
                            row_mean.append(" ")
                            continue
                        row_mean.append(f"{mean:.3e} ± {std:.3e}")
                        if iqm is None:
                            row_iqm.append(" ")
                            continue
                        row_iqm.append(f"{iqm:.3e} ± {iqm_std:.3e}")
                    if args.lowest:
                        if lowest is None:
                            row_lowest.append(" ")
                            continue
                        lowest = lowest
                        row_lowest.append(f"{lowest.to_numpy()[0]:.3e}")
                if args.mean:
                    rows_mean.append(row_mean)
                    rows_iqm.append(row_iqm)
                if args.lowest:
                    rows_lowest.append(row_lowest)

            
            if args.mean:
                # Regular mean
                table_result_path = generate_file_path(base_path, "mean", function, agent_id, args.format)
                table_content = generate_table(rows_mean, args.format)
                with table_result_path.open("w") as f:
                    f.write(table_content)
                # IQM
                table_result_path = generate_file_path(base_path, "iqm", function, agent_id, args.format)
                table_content = generate_table(rows_iqm, args.format)
                with table_result_path.open("w") as f:
                    f.write(table_content)
            if args.lowest:
                table_result_path = generate_file_path(base_path, "lowest", function, agent_id, args.format)
                table_content = generate_table(rows_lowest, args.format)
                with table_result_path.open("w") as f:
                    f.write(table_content)
