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
        type=list,
        default=["exponential_decay", "step_decay", "sgdr", "constant"],
    )
    parser.add_argument(
        "--functions",
        help="Specify which functions to generate the table for",
        type=list,
        default=["Ackley", "Rastrigin", "Rosenbrock", "Sphere"],
    )
    parser.add_argument(
        "--ids",
        help="Specify which ids to generate tables for",
        type=list,
        default=[0],
    )
    args = parser.parse_args()

    base_path = Path(args.path)
    for agent_id in args.ids:
        header = [None]
        header.extend(args.functions)
        rows_mean = [header]
        rows_lowest = [header]
        for agent in args.agents:
            row_mean = [agent]
            row_lowest = [agent]
            for function in args.functions:
                if args.results:
                    run_data_path = base_path / agent / str(agent_id) / function / "results"
                else:
                    run_data_path = base_path / agent/ str(agent_id) / function / "aggregated_run_data.csv"
                mean, std, lowest, min_path = calculate_statistics(path=run_data_path, results=args.results)
                pattern = r"(\d+)"
                train_steps = int(re.findall(pattern, str(min_path))[-1])
                if args.mean:
                    row_mean.append(f"{mean:.3e} ± {std:.3e}, {train_steps}")
                if args.lowest:
                    lowest = lowest.to_numpy()[0]
                    row_lowest.append(f"{lowest:.3e}")
            if args.mean:
                rows_mean.append(row_mean)
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
        if args.lowest:
            table_result_path = table_dir / f"lowest_{agent_or_teacher}_{agent_id}.md"
            table = table_from_string_list(rows_lowest)
            markdown = generate_markdown(table)
            with table_result_path.open("w") as f:
                f.write(markdown)