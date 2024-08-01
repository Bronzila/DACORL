import argparse
from pathlib import Path
from src.utils.general import calculate_statistics

single_name_mapping = {
    "teacher": {
        "constant": "Constant",
        "exponential_decay": "Exp. Decay",
        "step_decay": "Step Decay",
        "sgdr": "SGDR"
    },
    "agent": {
        "constant": "TD3+BC-C",
        "exponential_decay": "TD3+BC-E",
        "step_decay": "TD3+BC-ST",
        "sgdr": "TD3+BC-SG"
    }
}
heterogeneous_name_mapping = {
    "teacher": {
        "combined": "All",
        "combined_e_c": "Exp + Const",
        "combined_e_sg": "Exp + SGDR",
        "combined_e_sg_c": "Exp + SGDR + Const",
        "combined_e_st": "Exp + Step",
        "combined_e_st_c": "Exp + Step + Const",
        "combined_e_st_sg": "Exp + Step + SGDR",
        "combined_sg_c": "SGDR + Const",
        "combined_st_c": "Step + Const",
        "combined_st_sg": "Step + SGDR",
        "combined_st_sg_c": "Step + SGDR + Const"
    },
    "agent": {
        "combined": "TD3+BC-All",
        "combined_e_c": "TD3+BC-E-C",
        "combined_e_sg": "TD3+BC-E-SG",
        "combined_e_sg_c": "TD3+BC-E-SG-C",
        "combined_e_st": "TD3+BC-E-ST",
        "combined_e_st_c": "TD3+BC-E-ST-C",
        "combined_e_st_sg": "TD3+BC-E-ST-SG",
        "combined_sg_c": "TD3+BC-SG-C",
        "combined_st_c": "TD3+BC-ST-C",
        "combined_st_sg": "TD3+BC-ST-SG",
        "combined_st_sg_c": "TD3+BC-ST-SG-C"
    }
}

def format_number(num):
    if 1 <= num <= 999:
        return f"{num:.2f}".rstrip('0').rstrip('.')
    elif 0.1 <= num < 1:
        return f"{num:.3f}".rstrip('0').rstrip('.')
    else:
        formatted_num = f"{num:.2e}"
        return formatted_num.replace('e-0', 'e-').replace('e+0', 'e')

def generate_latex(table, caption, label):
    latex = "{ \\renewcommand{\\arraystretch}{1.4} % Adjust the row height only within this group\n"
    latex += "\\begin{table}[h]\n\\fontsize{10}{12}\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\begin{tabular}{l " + "c " * (len(table[0]) - 1) + "}\n\\toprule\n"
    for i, row in enumerate(table):
        if i == 0:
            latex += " & ".join(row) + " \\\\\n\\midrule\n"
        elif i == len(table) - 1:
            latex += " & ".join(row) + " \\\\\n\\bottomrule\n"
        else:
            latex += " & ".join(row) + " \\\\\n"
    latex += "\\end{tabularx}\n\\end{table}\n}"
    return latex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument(
        "--path", type=str, help="Base path",
        default="data/ToySGD"
    )
    parser.add_argument(
        "--teacher_base_path", type=str, help="Teacher base path"
    )
    parser.add_argument(
        "--heterogeneous",
        help="Print heterogeneous results",
        action=argparse.BooleanOptionalAction,
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
        "--auc",
        help="Get mean and std deviation table of AuC",
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
        default=["constant", "exponential_decay", "step_decay", "sgdr"],
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
        type=str,
        nargs="*",
        default=[0],
    )
    parser.add_argument(
        "--multi_seed",
        help="Calculate table for multi-seed",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--num_runs",
        help="Number of runs used for evaluation. Needed for multi-seed results in order to adjust indices correctly",
        type=int,
        default=100
    )
    parser.add_argument(
        "--interpolation",
        help="Whether interpolation protocol has been used to evaluate agents",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    if args.heterogeneous:
        name_mapping = heterogeneous_name_mapping
    else:
        name_mapping = single_name_mapping

    base_path = Path(args.path)
    teacher_base_path = Path(args.teacher_base_path) if args.teacher_base_path else base_path

    for agent_id in args.ids:
        header = [""]
        header.extend(args.functions)
        rows_mean = [header]
        rows_iqm = [header]
        rows_auc = [header]
        rows_lowest = [header]

        teacher_values = {agent: {} for agent in args.agents}

        # Process teachers
        for agent in args.agents:
            row_mean = [name_mapping["teacher"][agent]]
            row_iqm = [name_mapping["teacher"][agent]]
            row_auc = [name_mapping["teacher"][agent]]
            row_lowest = [name_mapping["teacher"][agent]]
            for function in args.functions:
                run_data_path = teacher_base_path / agent / str(agent_id) / function / "aggregated_run_data.csv"
                if args.heterogeneous:
                    run_data_path = teacher_base_path / agent / function / "aggregated_run_data.csv"
                mean, std, lowest, iqm, iqm_std, min_path, auc, auc_std = calculate_statistics(
                    path=run_data_path, results=False, verbose=args.verbose, multi_seed=args.multi_seed, num_runs=args.num_runs, interpolation=args.interpolation)

                teacher_values[agent][function] = (mean, std, iqm, iqm_std, auc, auc_std, lowest)

                if args.mean:
                    row_mean.append(f"{format_number(mean)} $\pm$ {format_number(std)}")
                    row_iqm.append(f"{format_number(iqm)} $\pm$ {format_number(iqm_std)}")
                if args.lowest:
                    lowest = lowest.to_numpy()[0]
                    row_lowest.append(f"{format_number(lowest)}")
                if args.auc:
                    row_auc.append(f"{format_number(auc)} $\pm$ {format_number(auc_std)}")

            if args.mean:
                rows_mean.append(row_mean)
                rows_iqm.append(row_iqm)
            if args.lowest:
                rows_lowest.append(row_lowest)
            if args.auc:
                rows_auc.append(row_auc)

        # Process agents
        best_values = {function: {'mean': float('inf'), 'iqm': float('inf'), 'auc': float('inf'), 'lowest': float('inf')} for function in args.functions}
        best_agents = {function: {'mean': None, 'iqm': None, 'auc': None, 'lowest': None} for function in args.functions}

        for agent in args.agents:
            row_mean = [name_mapping["agent"][agent]]
            row_iqm = [name_mapping["agent"][agent]]
            row_auc = [name_mapping["agent"][agent]]
            row_lowest = [name_mapping["agent"][agent]]
            for function in args.functions:
                run_data_path = base_path / agent / str(agent_id) / function / "results"
                if args.heterogeneous:
                    run_data_path = base_path / agent / function / "results"
                mean, std, lowest, iqm, iqm_std, min_path, auc, auc_std = calculate_statistics(
                    path=run_data_path, results=True, verbose=args.verbose, multi_seed=args.multi_seed, num_runs=args.num_runs, interpolation=args.interpolation)

                teacher_mean, teacher_std, teacher_iqm, teacher_iqm_std, teacher_auc, teacher_auc_std, teacher_lowest = teacher_values[agent][function]

                lowest = lowest.to_numpy()[0]  # Extract the scalar value from the Series
                teacher_lowest = teacher_lowest.to_numpy()[0]  # Extract the scalar value from the Series

                # Initialize the value variables
                mean_value = f"{format_number(mean)} $\pm$ {format_number(std)}"
                iqm_value = f"{format_number(iqm)} $\pm$ {format_number(iqm_std)}"
                auc_value = f"{format_number(auc)} $\pm$ {format_number(auc_std)}"
                lowest_value = f"{format_number(lowest)}"

                # Check if the agent outperforms the teacher
                if mean < teacher_mean or (mean == teacher_mean and std < teacher_std):
                    mean_value = f"\\textbf{{{mean_value}}}"
                if iqm < teacher_iqm or (iqm == teacher_iqm and iqm_std < teacher_iqm_std):
                    iqm_value = f"\\textbf{{{iqm_value}}}"
                if auc < teacher_auc or (auc == teacher_auc and auc_std < teacher_auc_std):
                    auc_value = f"\\textbf{{{auc_value}}}"
                if lowest < teacher_lowest:
                    lowest_value = f"\\textbf{{{lowest_value}}}"

                # Check if the agent is the best overall
                if mean < best_values[function]['mean']:
                    best_values[function]['mean'] = mean
                    best_agents[function]['mean'] = (agent, 'agent')
                if iqm < best_values[function]['iqm']:
                    best_values[function]['iqm'] = iqm
                    best_agents[function]['iqm'] = (agent, 'agent')
                if auc < best_values[function]['auc']:
                    best_values[function]['auc'] = auc
                    best_agents[function]['auc'] = (agent, 'agent')
                if lowest < best_values[function]['lowest']:
                    best_values[function]['lowest'] = lowest
                    best_agents[function]['lowest'] = (agent, 'agent')

                row_mean.append(mean_value)
                row_iqm.append(iqm_value)
                row_auc.append(auc_value)
                row_lowest.append(lowest_value)

            rows_mean.append(row_mean)
            rows_iqm.append(row_iqm)
            rows_auc.append(row_auc)
            rows_lowest.append(row_lowest)

        # Add teacher values to best value comparison
        for agent in args.agents:
            for function in args.functions:
                mean, std, iqm, iqm_std, auc, auc_std, lowest = teacher_values[agent][function]

                lowest = lowest.to_numpy()[0]  # Ensure lowest is a scalar value

                if mean < best_values[function]['mean']:
                    best_values[function]['mean'] = mean
                    best_agents[function]['mean'] = (agent, 'teacher')
                if iqm < best_values[function]['iqm']:
                    best_values[function]['iqm'] = iqm
                    best_agents[function]['iqm'] = (agent, 'teacher')
                if auc < best_values[function]['auc']:
                    best_values[function]['auc'] = auc
                    best_agents[function]['auc'] = (agent, 'teacher')
                if lowest < best_values[function]['lowest']:
                    best_values[function]['lowest'] = lowest
                    best_agents[function]['lowest'] = (agent, 'teacher')

        def highlight_best(rows, best_agents, metric):
            for row in rows:
                for i, function in enumerate(args.functions, 1):
                    agent, role = best_agents[function][metric]
                    if role == 'teacher':
                        best_name = name_mapping['teacher'][agent]
                        if row[0] == best_name:
                            row[i] = f"\\cellcolor{{highlight}}{row[i]}"
                    else:
                        best_name = name_mapping['agent'][agent]
                        if row[0] == best_name:
                            if '\\textbf{' in row[i]:
                                row[i] = f"\\cellcolor{{highlight}}{row[i]}"
                            else:
                                row[i] = f"\\cellcolor{{highlight}}\\textbf{{{row[i]}}}"
            return rows

        rows_mean = highlight_best(rows_mean, best_agents, 'mean')
        rows_iqm = highlight_best(rows_iqm, best_agents, 'iqm')
        rows_auc = highlight_best(rows_auc, best_agents, 'auc')
        rows_lowest = highlight_best(rows_lowest, best_agents, 'lowest')

        table_name = "tables_interpolation" if args.interpolation else "tables"
        table_dir = base_path / table_name
        table_dir.mkdir(exist_ok=True)
        if args.mean:
            table_result_path = table_dir / f"mean_{agent_id}.tex"
            table = rows_mean
            latex = generate_latex(table, "Mean Results", f"tab:mean_{agent_id}")
            with table_result_path.open("w") as f:
                f.write(latex)

            table_result_path = table_dir / f"iqm_{agent_id}.tex"
            table = rows_iqm
            latex = generate_latex(table, "IQM Results", f"tab:iqm_{agent_id}")
            with table_result_path.open("w") as f:
                f.write(latex)
        if args.lowest:
            table_result_path = table_dir / f"lowest_{agent_id}.tex"
            table = rows_lowest
            latex = generate_latex(table, "Lowest Results", f"tab:lowest_{agent_id}")
            with table_result_path.open("w") as f:
                f.write(latex)
        if args.auc:
            table_result_path = table_dir / f"auc_{agent_id}.tex"
            table = rows_auc
            latex = generate_latex(table, "AUC Results", f"tab:auc_{agent_id}")
            with table_result_path.open("w") as f:
                f.write(latex)
