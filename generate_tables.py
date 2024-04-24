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
        "--custom_path", type=str, help="Base path",
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
        "--results",
        help="Find results subfolders",
        action=argparse.BooleanOptionalAction,
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
        "--ids",
        help="Specify which ids to generate tables for",
        type=int,
        nargs="*",
        default=[0],
    )
    parser.add_argument(
        "--multi_seed",
        help="Calculate table for multi-seed",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    if args.custom_path:
        base_path = Path(args.custom_path)
    for agent_id in args.ids:
        header = [None]
        header.extend(args.functions)
        rows_mean = [header]
        rows_iqm = [header]
        rows_lowest = [header]
        for agent in args.agents:
            row_mean = [agent]
            row_iqm = [agent]
            row_lowest = [agent]
            for function in args.functions:
                if args.results:
                    run_data_path = base_path / agent / str(agent_id) / function / "results"
                else:
                    run_data_path = base_path / agent/ str(agent_id) / function / "aggregated_run_data.csv"
                if args.custom_path:                    
                    # run_data_path = base_path / agent / function / "results" if args.results else base_path / agent / function / "aggregated_run_data.csv"
                    run_data_path = base_path / function / "results" if args.results else base_path / function / "aggregated_run_data.csv"
                mean, std, lowest, iqm, iqm_std, min_path = calculate_statistics(path=run_data_path, results=args.results, verbose=args.verbose)
                pattern = r"(\d+)"
                train_steps = int(re.findall(pattern, str(min_path))[-1]) if args.results else 0
                if args.mean:
                    row_mean.append(f"{mean:.3e} ± {std:.3e}, {train_steps}")
                    row_iqm.append(f"{iqm:.3e} ± {iqm_std:.3e}, {train_steps}")
                if args.lowest:
                    lowest = lowest.to_numpy()[0]
                    row_lowest.append(f"{lowest:.3e}")
            if args.mean:
                rows_mean.append(row_mean)
                rows_iqm.append(row_iqm)
            if args.lowest:
                rows_lowest.append(row_lowest)

        table_dir = base_path / "tables"
        table_dir.mkdir(exist_ok=True)
        agent_or_teacher = "agent" if args.results else "teacher"
        if args.mean:
            table_result_path = table_dir / f"mean_{agent_or_teacher}_{agent_id}.md"
            table = table_from_string_list(rows_mean)
            markdown = generate_markdown(table)
            with table_result_path.open("w") as f:
                f.write(markdown)

            table_result_path = table_dir / f"iqm_{agent_or_teacher}_{agent_id}.md"
            table = table_from_string_list(rows_iqm)
            markdown = generate_markdown(table)
            with table_result_path.open("w") as f:
                f.write(markdown)
        if args.lowest:
            table_result_path = table_dir / f"lowest_{agent_or_teacher}_{agent_id}.md"
            table = table_from_string_list(rows_lowest)
            markdown = generate_markdown(table)
            with table_result_path.open("w") as f:
                f.write(markdown)