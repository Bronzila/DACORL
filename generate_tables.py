import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from check_fbest import calculate_statistics
from markdown_table_generator import generate_markdown, table_from_string_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument(
        "--path", type=str, help="Base path",
        default="data/ToySGD"
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
        nargs='+',
        default=["exponential_decay", "step_decay", "sgdr", "constant"],
    )
    parser.add_argument(
        "--agents",
        help="Specify which agents to generate the table for",
        nargs='+',
        default=["bc", "td3_bc", "cql", "awac", "edac", "sac_n", "lb_sac"],
    )
    parser.add_argument(
        "--functions",
        help="Specify which functions to generate the table for",
        nargs='+',
        default=["Ackley", "Rastrigin", "Rosenbrock", "Sphere"],
    )
    parser.add_argument(
        "--hpo_budget",
        help="HPO budget used",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--ids",
        help="Specify which ids to generate tables for",
        nargs='+', 
        default=[0],
    )
    args = parser.parse_args()

    args.agents.insert(0, "teacher")
    base_path = Path(args.path)
    for function in args.functions:
        for agent_id in args.ids:
            header = [None]
            header.extend(args.teacher)
            rows_mean = [header]
            rows_lowest = [header]
            for i, agent in enumerate(args.agents):
                row_mean = [agent]
                row_lowest = [agent]
                for j, teacher in enumerate(args.teacher):
                    if agent == "teacher":
                        run_data_path = base_path / teacher/ str(agent_id) / function / "aggregated_run_data.csv"
                    else:
                        run_data_path = base_path / teacher / str(agent_id) / function / "results" / agent / f"{args.hpo_budget}" / "eval_data.csv"
                    mean, std, lowest, min_path = calculate_statistics(path=run_data_path, results=False, verbose=args.verbose)

                    if args.mean:
                        row_mean.append(f"{mean:.3e} ± {std:.3e}")
                    if args.lowest:
                        lowest = lowest.to_numpy()[0]
                        row_lowest.append(f"{lowest:.3e}")
                if args.mean:
                    rows_mean.append(row_mean)
                if args.lowest:
                    rows_lowest.append(row_lowest)

            table_dir = base_path / "tables"
            table_dir.mkdir(exist_ok=True)
            if args.mean:
                table_result_path = table_dir / f"mean_{function}_{agent_id}.md"
                table = table_from_string_list(rows_mean)
                markdown = generate_markdown(table)
                with table_result_path.open("w") as f:
                    f.write(markdown)
            if args.lowest:
                table_result_path = table_dir / f"lowest_{function}_{agent_id}.md"
                table = table_from_string_list(rows_lowest)
                markdown = generate_markdown(table)
                with table_result_path.open("w") as f:
                    f.write(markdown)