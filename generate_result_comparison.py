import re
import numpy as np
import argparse
from pathlib import Path
import pandas as pd


def read_markdown_table(file_path):
    with file_path.open("r") as file:
        lines = file.readlines()
        # Extract the teacher
        teacher = lines[0].strip().split("|")[1:-1]
        # Extract the rows
        rows = [line.strip().split("|")[1:-1] for line in lines[2:]]
    return teacher, rows


def parse_value(value):
    # Handle values with ± symbol
    if "±" in value:
        mean, std = value.split("±")
        return float(mean), float(std)
    else:
        return float(value), None


def create_dataframe(teachers, rows):
    data = {teacher: [] for teacher in teachers}
    for row in rows:
        for teacher, value in zip(teachers, row):
            data[teacher].append(value)
    return pd.DataFrame(data)


def calculate_percentage_change(val1, val2):
    if val2 != 0:
        return ((val1 - val2) / abs(val2 + 10e-5)) * 100
    return float("inf")


def compare_experiments(exp1_path: Path, exp2_path: Path, id1: str, id2: str):
    experiments_paths = [exp1_path, exp2_path]
    tables_path = "tables"

    statistics_types = ["iqm", "mean", "lowest"]
    functions = ["Ackley", "Rastrigin", "Rosenbrock"]#, "Sphere"]
    agent_list = ["td3_bc", "cql", "awac", "edac", "sac_n", "lb_sac","bc", "iql", "td3"]

    results = {
        stat_type: {func: {} for func in functions}
        for stat_type in statistics_types
    }

    for stat_type in statistics_types:
        for func in functions:
            file_name = [f"{stat_type}_{func}_{id}.md" for id in [id1, id2]]
            file_paths = [
                exp / "ToySGD" / tables_path / file_name[i]
                for i, exp in enumerate(experiments_paths)
            ]

            if all(file_path.exists() for file_path in file_paths):
                teachers_1, rows_1 = read_markdown_table(file_paths[0])
                teachers_2, rows_2 = read_markdown_table(file_paths[1])

                df1 = create_dataframe(teachers_1, rows_1)
                df2 = create_dataframe(teachers_2, rows_2)

                # Clean agent names by stripping whitespaces
                df1.iloc[:, 0] = df1.iloc[:, 0].str.strip()
                df2.iloc[:, 0] = df2.iloc[:, 0].str.strip()

                # Find common agents excluding 'teacher'
                common_agents = set(df1.iloc[:, 0]).intersection(
                    set(df2.iloc[:, 0])
                ) - {"teacher"}

                if common_agents:
                    for agent in common_agents:
                        row1 = df1[df1.iloc[:, 0] == agent].iloc[0]
                        row2 = df2[df2.iloc[:, 0] == agent].iloc[0]

                        for teacher in teachers_1[1:]:
                            val1, std1 = parse_value(row1[teacher])
                            val2, std2 = parse_value(row2[teacher])

                            if std1 is not None and std2 is not None:
                                val_change = calculate_percentage_change(
                                    val1, val2
                                )
                                std_change = calculate_percentage_change(
                                    std1, std2
                                )
                                results[stat_type][func].setdefault(
                                    agent, {}
                                ).setdefault(teacher, []).append(val_change)
                                results[stat_type][func].setdefault(
                                    "overall", {}
                                ).setdefault(teacher, []).append(val_change)
                            else:
                                val_change = calculate_percentage_change(
                                    val1, val2
                                )
                                results[stat_type][func].setdefault(
                                    agent, {}
                                ).setdefault(teacher, []).append(val_change)
                                results[stat_type][func].setdefault(
                                    "overall", {}
                                ).setdefault(teacher, []).append(val_change)
                else:
                    print(
                        f"No common agents found for {stat_type.upper()} - {func} excluding 'teacher'\n"
                    )
            else:
                for file_path in file_paths:
                    if not file_path.exists():
                        print(f"Warning: File {file_path} does not exist.\n")

    # Calculate and print mean relative changes
    for stat_type, funcs in results.items():
        print(f"\n{stat_type.upper()}:")

        # Mean change for each agent over multiple functions
        agent_changes = {}
        for func, agents in funcs.items():
            for agent, teachers in agents.items():
                if agent != "overall" and agent in agent_list:
                    for teacher, changes in teachers.items():
                        agent_changes.setdefault(agent, []).extend(changes)

        print("\nMean change for each agent:")
        for agent, changes in agent_changes.items():
            mean_change = np.mean(changes)
            print(f"  Agent: {agent} | Mean Change: {mean_change:.2f}%")

        # Mean change for each function over multiple agents
        func_changes = {func: [] for func in functions}
        for func, agents in funcs.items():
            for agent, teachers in agents.items():
                if agent != "overall" and agent in agent_list:
                    for teacher, changes in teachers.items():
                        func_changes[func].extend(changes)

        print("\nMean change for each function:")
        for func, changes in func_changes.items():
            mean_change = np.mean(changes)
            print(f"  Function: {func} | Mean Change: {mean_change:.2f}%")

        # Mean change for each teacher over multiple functions and agents
        teacher_changes = {}
        for func, agents in funcs.items():
            for agent, teachers in agents.items():
                if agent in agent_list:
                    for teacher, changes in teachers.items():
                        teacher_changes.setdefault(teacher, []).extend(changes)

        print("\nMean change for each Teacher:")
        for teacher, changes in teacher_changes.items():
            mean_change = np.mean(changes)
            print(f"  Teacher: {teacher} | Mean Change: {mean_change:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare experiment results between two experiments."
    )
    parser.add_argument(
        "exp1_path", type=Path, help="Path to the first experiments folder."
    )
    parser.add_argument(
        "exp2_path", type=Path, help="Path to the second experiments folder."
    )
    parser.add_argument(
        "id1",
        nargs="?",
        type=str,
        help="ID which teacher configuration was used in the first experiment.",
        default="0",
    )
    parser.add_argument(
        "id2",
        nargs="?",
        type=str,
        help="ID which teacher instantiation was used in the second experiment.",
        default="0",
    )
    args = parser.parse_args()

    compare_experiments(args.exp1_path, args.exp2_path, args.id1, args.id2)
