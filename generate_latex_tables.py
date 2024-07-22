import argparse
import re
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
        "combined": "Combined all",
        "combined_e_c": "Exp + Const",
        "combined_e_sg": "Exp + SGDR",
        "combined_e_sg_c": "Exp + SGDR + Const",
        "combined_e_st": "Exp + Step",
        "combined_e_st_c": "Exp + Step + Const",
        "combined_e_st_sg": "Exp + Step + SGDR",
        "combined_sg_c": "SGDR + Const",
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

def generate_latex(table):
    latex = "{ \\renewcommand{\\arraystretch}{1.4} % Adjust the row height only within this group\n"
    latex += "\\begin{table}[h]\n\\centering\n\\caption{Your caption here}\n\\label{tab:your_label}\n\\begin{tabularx}{\\textwidth}{l " + "c " * (len(table[0]) - 1) + "}\n\\toprule\n"
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
        help="Number of runs used for evaluation. Needed for multi-seed results in order to adjust indeces correctly",
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
        name_mapping = heterogeneous_name_mapping["agent"] if args.results else heterogeneous_name_mapping["teacher"]
    else:
        name_mapping = single_name_mapping["agent"] if args.results else single_name_mapping["teacher"]

    base_path = Path(args.path)
    for agent_id in args.ids:
        header = [""]
        header.extend(args.functions)
        rows_mean = [header]
        rows_iqm = [header]
        rows_auc = [header]
        rows_lowest = [header]
        for agent in args.agents:
            row_mean = [name_mapping[agent]]
            row_iqm = [name_mapping[agent]]
            row_auc = [name_mapping[agent]]
            row_lowest = [name_mapping[agent]]
            for function in args.functions:
                if args.results:
                    run_data_path = base_path / agent / str(agent_id) / function / "results"
                else:
                    run_data_path = base_path / agent / str(agent_id) / function / "aggregated_run_data.csv"
                if args.heterogeneous:                    
                    run_data_path = base_path / agent / function / "results" if args.results else base_path / agent / function / "aggregated_run_data.csv"
                mean, std, lowest, iqm, iqm_std, min_path, auc, auc_std = calculate_statistics(path=run_data_path, results=args.results, verbose=args.verbose, multi_seed=args.multi_seed, num_runs=args.num_runs, interpolation=args.interpolation)
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

        table_name = "tables_interpolation" if args.interpolation else "tables"
        table_dir = base_path / table_name
        table_dir.mkdir(exist_ok=True)
        agent_or_teacher = "agent" if args.results else "teacher"
        if args.mean:
            table_result_path = table_dir / f"mean_{agent_or_teacher}_{agent_id}.tex"
            table = rows_mean
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)

            table_result_path = table_dir / f"iqm_{agent_or_teacher}_{agent_id}.tex"
            table = rows_iqm
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)
        if args.lowest:
            table_result_path = table_dir / f"lowest_{agent_or_teacher}_{agent_id}.tex"
            table = rows_lowest
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)
        if args.auc:
            table_result_path = table_dir / f"auc_{agent_or_teacher}_{agent_id}.tex"
            table = rows_auc
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)