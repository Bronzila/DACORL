import argparse
import re
from pathlib import Path

from src.utils.calculate_sgd_statistic import calculate_statistics

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

def format_percentage(num):
    return f"{num*100:.2f}"

def format_number(num):
    if 1 <= num <= 999:
        return f"{num:.2f}".rstrip('0').rstrip('.')
    elif 0.1 <= num < 1:
        return f"{num:.3f}".rstrip('0').rstrip('.')
    else:
        formatted_num = f"{num:.2e}"
        return formatted_num.replace('e-0', 'e-').replace('e+0', 'e')

def generate_latex(table):
    latex = "\\begin{table}[htb]\n\\centering\n\\caption{Your caption here}\n\\label{tab:your_label}\n\\begin{tabular}{l " + "c " * (len(table[0]) - 1) + "}\n\\toprule\n"
    for i, row in enumerate(table):
        if i == 0:
            latex += " & ".join(row) + " \\\\\n\\midrule\n"
        elif i == len(table) - 1:
            latex += " & ".join(row) + " \\\\\n\\bottomrule\n"
        else:
            latex += " & ".join(row) + " \\\\\n"
    latex += "\\end{tabular}\n\\end{table}\n"
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
        "--ids",
        help="Specify which ids to generate tables for",
        type=str,
        nargs="*",
        default=[0],
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

    # Only need teacher names as agents are columns
    if args.heterogeneous:
        name_mapping = heterogeneous_name_mapping["teacher"]
    else:
        name_mapping = single_name_mapping["teacher"]

    base_path = Path(args.path)
    for agent_id in args.ids:
        header = [""]
        header.extend(["Teacher", "TD3+BC"])
        rows_mean = [header]
        rows_iqm = [header]
        rows_auc = [header]
        rows_lowest = [header]
        for agent in args.agents:
            # First fill teacher/agent names in first column
            row_mean = [name_mapping[agent]]
            row_iqm = [name_mapping[agent]]
            row_auc = [name_mapping[agent]]
            row_lowest = [name_mapping[agent]]

            
            results_path = base_path / agent / str(agent_id) / "results"
            if args.teacher_base_path:
                teacher_path = Path(args.teacher_base_path) / agent / str(agent_id) / "aggregated_run_data.csv"
            else:
                teacher_path = base_path / agent / str(agent_id) / "aggregated_run_data.csv"

            if args.heterogeneous:                    
                results_path = base_path / agent / "results"
                if args.teacher_base_path:
                    teacher_path = Path(args.teacher_base_path) / agent / "aggregated_run_data.csv"

            # Calculate agent performance
            a_mean, a_std, a_lowest, a_iqm, a_iqm_std, a_min_path, a_auc, a_auc_std = calculate_statistics(path=results_path, results=True, verbose=args.verbose, multi_seed=True, num_runs=args.num_runs, interpolation=args.interpolation, metric="test_acc")

            # Calculate teacher performance
            t_mean, t_std, t_lowest, t_iqm, t_iqm_std, t_min_path, t_auc, t_auc_std = calculate_statistics(path=teacher_path, results=False, verbose=args.verbose, multi_seed=False, num_runs=args.num_runs, interpolation=args.interpolation, metric="test_acc")
            
            if args.mean:
                row_mean.append(f"{format_percentage(t_mean)} $\pm$ {format_percentage(t_std)}")
                row_mean.append(f"{format_percentage(a_mean)} $\pm$ {format_percentage(a_std)}")
                row_iqm.append(f"{format_percentage(t_iqm)} $\pm$ {format_percentage(t_iqm_std)}")
                row_iqm.append(f"{format_percentage(a_iqm)} $\pm$ {format_percentage(a_iqm_std)}")

                rows_mean.append(row_mean)
                rows_iqm.append(row_iqm)
            if args.lowest:
                t_lowest = t_lowest.to_numpy()[0]
                a_lowest = a_lowest.to_numpy()[0]
                row_lowest.append(f"{format_percentage(t_lowest)}")
                row_lowest.append(f"{format_percentage(a_lowest)}")
                rows_lowest.append(row_lowest)
            if args.auc:
                row_auc.append(f"{format_number(t_auc)} $\pm$ {format_number(t_auc_std)}")
                row_auc.append(f"{format_number(a_auc)} $\pm$ {format_number(a_auc_std)}")
                rows_auc.append(row_auc)    

        table_name = "tables_interpolation" if args.interpolation else "tables"
        table_dir = base_path / table_name
        table_dir.mkdir(exist_ok=True)
        if args.mean:
            table_result_path = table_dir / f"mean_{agent_id}.tex"
            table = rows_mean
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)

            table_result_path = table_dir / f"iqm_{agent_id}.tex"
            table = rows_iqm
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)
        if args.lowest:
            table_result_path = table_dir / f"lowest_{agent_id}.tex"
            table = rows_lowest
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)
        if args.auc:
            table_result_path = table_dir / f"auc_{agent_id}.tex"
            table = rows_auc
            latex = generate_latex(table)
            with table_result_path.open("w") as f:
                f.write(latex)
