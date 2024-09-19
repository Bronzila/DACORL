import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.run_statistics import (
    calc_mean_and_std_dev,
    compute_AuC,
    compute_iqm,
    find_lowest_values,
    combine_run_data
)

cma_es_function = {
    12: "BentCigar",
    11: "Discus",
    2: "Ellipsoid",
    23: "Katsuura",
    15: "Rastrigin",
    8: "Rosenbrock",
    17: "Schaffers",
    20: "Schwefel",
    1: "Sphere",
    16: "Weierstrass",
}

unimodal_functions = ["BentCigar", "Discus", "Ellipsoid", "Sphere"]
multimodal_functions_1 = ["Katsuura", "Rastrigin", "Rosenbrock"]
multimodal_functions_2 = ["Schaffers", "Schwefel", "Weierstrass"]

def get_grouped_df(paths: Path | list[Path], num_runs: int) -> pd.DataFrame:
    if isinstance(paths, Path):
        df = pd.read_csv(paths)
    elif isinstance(paths, list):
        df = combine_run_data(paths, num_runs=num_runs)
    return df

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
        for col in metric_min:
            i, j, _ = col
            df.iloc[i, j] = f"\\cellcolor{{highlight}}{df.iloc[i, j]}"
        latex_str = df.to_latex(index=False, escape=False)
        latex_str = latex_str.replace('{llll}', '{lccc}', 1)
        latex_str = latex_str.replace('{lllll}', '{lcccc}', 1)
        return latex_str

def generate_file_path(base_path: Path, metric: str, function_group: str, format: str):
    suffix = "md" if format == "markdown" else "tex"
    table_dir = base_path / "tables"
    table_dir.mkdir(exist_ok=True)
    return table_dir / f"{metric}_{function_group}.{suffix}"

def calculate_percentage_change(reference, current):
    return ((current - reference) / abs(reference)) * 100

def compute_metrics_for_agent(agent, agent_path, functions, objective, num_runs):
    results = {func_name: {} for func_name in functions.values()}
    grouped_agent = get_grouped_df(agent_path, num_runs=num_runs)
    for function_id, function_name in functions.items():
        agent_results = grouped_agent[grouped_agent["function_id"] == function_id]
        mean_a, std_a = calc_mean_and_std_dev(agent_results, objective)
        iqm_mean_a, iqm_std_a = compute_iqm(agent_results, objective)
        auc_mean_a, auc_std_a = compute_AuC(agent_results, objective)
        lowest_a = find_lowest_values(agent_results, objective, n=1)[objective].item()

        results[function_name] = {
            "mean": (mean_a, std_a),
            "iqm": (iqm_mean_a, iqm_std_a),
            "auc": (auc_mean_a, auc_std_a),
            "lowest": lowest_a
        }
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument("--path", type=str, help="Base path", default="data/ToySGD")
    parser.add_argument("--lowest", help="Get fbest table", action="store_true")
    parser.add_argument("--mean", help="Get mean and std deviation table", action="store_true")
    parser.add_argument("--auc", help="Get mean and std deviation table of AuC", action="store_true", default=True)
    parser.add_argument("--verbose", help="Verbose output", action="store_true")
    parser.add_argument("--teacher", help="Specify which agents to generate the table for", nargs="+", default=["csa", "cmaes_constant"])
    parser.add_argument("--agents", help="Specify which agents to generate the table for", nargs="+", default=["bc", "td3_bc", "cql", "awac", "edac", "sac_n", "lb_sac", "iql"])
    parser.add_argument("--hpo_budget", help="HPO budget used", type=int, default=30000)
    parser.add_argument("--ids", help="Specify which ids to generate tables for", nargs="+", default=[0])
    parser.add_argument("--format", help="Output format: markdown or latex", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--num_runs", help="Number of runs used for evaluation. Needed for multi-seed results in order to adjust indices correctly", type=int, default=1000)
    args = parser.parse_args()

    base_path = Path(args.path)
    if base_path.name == "ToySGD":
        objective = "f_cur"
    elif base_path.name == "SGD":
        objective = "valid_acc"
    elif base_path.name == "CMAES":
        objective = "f_cur"
    else:
        raise NotImplementedError(f"Currently the benchmark {base_path.name} is not implemented.")

    pm = "$\pm$" if args.format == "latex" else "±"

    for teacher in args.teacher:
        header_unimodal = ["Agent"] + unimodal_functions
        header_multimodal_1 = ["Agent"] + multimodal_functions_1
        header_multimodal_2 = ["Agent"] + multimodal_functions_2

        rows_mean_unimodal = [header_unimodal]
        rows_iqm_unimodal = [header_unimodal]
        rows_auc_unimodal = [header_unimodal]
        rows_lowest_unimodal = [header_unimodal]
        
        rows_mean_multimodal_1 = [header_multimodal_1]
        rows_iqm_multimodal_1 = [header_multimodal_1]
        rows_auc_multimodal_1 = [header_multimodal_1]
        rows_lowest_multimodal_1 = [header_multimodal_1]
        
        rows_mean_multimodal_2 = [header_multimodal_2]
        rows_iqm_multimodal_2 = [header_multimodal_2]
        rows_auc_multimodal_2 = [header_multimodal_2]
        rows_lowest_multimodal_2 = [header_multimodal_2]

        mean_min_uni = [[0, 0, np.inf]]
        iqm_min_uni = [[0, 0, np.inf]]
        lowest_min_uni = [[0, 0, np.inf]]
        auc_min_uni = [[0, 0, np.inf]]
        
        mean_min_multi_1 = [[0, 0, np.inf]]
        iqm_min_multi_1 = [[0, 0, np.inf]]
        lowest_min_multi_1 = [[0, 0, np.inf]]
        auc_min_multi_1 = [[0, 0, np.inf]]
        
        mean_min_multi_2 = [[0, 0, np.inf]]
        iqm_min_multi_2 = [[0, 0, np.inf]]
        lowest_min_multi_2 = [[0, 0, np.inf]]
        auc_min_multi_2 = [[0, 0, np.inf]]

        metrics = {}
        args.agents.insert(0, teacher)
        for agent in args.agents:
            if agent in args.teacher:
                agent_path = base_path / agent / str(args.ids[0]) / "aggregated_run_data.csv"
            else:
                agent_path = base_path / teacher / str(args.ids[0]) / "results" / agent
                agent_paths = list(agent_path.rglob(f"*{args.hpo_budget}/eval_data.csv"))
                agent_path = agent_paths if agent_paths else None

            print(agent)
            metrics[agent] = compute_metrics_for_agent(agent, agent_path, cma_es_function, objective, args.num_runs)

        for agent in args.agents:
            agent_str = f"\\acrshort{{{agent}}}"

            row_mean_uni = [agent_str]
            row_iqm_uni = [agent_str]
            row_auc_uni = [agent_str]
            row_lowest_uni = [agent_str]

            row_mean_multi_1 = [agent_str]
            row_iqm_multi_1 = [agent_str]
            row_auc_multi_1 = [agent_str]
            row_lowest_multi_1 = [agent_str]

            row_mean_multi_2 = [agent_str]
            row_iqm_multi_2 = [agent_str]
            row_auc_multi_2 = [agent_str]
            row_lowest_multi_2 = [agent_str]

            for function_name in unimodal_functions:
                teacher_metrics = metrics[teacher][function_name]
                agent_metrics = metrics[agent][function_name]
                if args.mean:
                    mean_a, std_a = agent_metrics["mean"]
                    mean_str = f"{format_number(mean_a)} {pm} {format_number(std_a)}"
                    iqm_mean_a, iqm_std_a = agent_metrics["iqm"]
                    iqm_str = f"{format_number(iqm_mean_a)} {pm} {format_number(iqm_std_a)}"
                    if mean_a < teacher_metrics["mean"][0]:
                        mean_str = f"\\textbf{{{mean_str}}}"
                    if iqm_mean_a < teacher_metrics["iqm"][0]:
                        iqm_str = f"\\textbf{{{iqm_str}}}"
                    
                    row_mean_uni.append(mean_str)
                    row_iqm_uni.append(iqm_str)
                    if mean_a < mean_min_uni[0][2]:
                        mean_min_uni[0] = [len(rows_mean_unimodal), len(row_mean_uni) - 1, mean_a]
                    if iqm_mean_a < iqm_min_uni[0][2]:
                        iqm_min_uni[0] = [len(rows_iqm_unimodal), len(row_iqm_uni) - 1, iqm_mean_a]

                if args.lowest:
                    lowest_a = agent_metrics["lowest"]
                    lowest_str = f"{format_number(lowest_a)}"
                    if lowest_a < teacher_metrics["lowest"]:
                        lowest_str = f"\\textbf{{{lowest_str}}}"                    
                    row_lowest_uni.append(lowest_str)
                    if lowest_a < lowest_min_uni[0][2]:
                        lowest_min_uni[0] = [len(rows_lowest_unimodal), len(row_lowest_uni) - 1, lowest_a]

                if args.auc:
                    auc_mean_a, auc_std_a = agent_metrics["auc"]
                    auc_str = f"{format_number(auc_mean_a)} {pm} {format_number(auc_std_a)}"
                    if lowest_a < teacher_metrics["auc"][0]:
                        auc_str = f"\\textbf{{{auc_str}}}" 
                    
                    row_auc_uni.append(auc_str)
                    if auc_mean_a < auc_min_uni[0][2]:
                        auc_min_uni[0] = [len(rows_auc_unimodal), len(row_auc_uni) - 1, auc_mean_a]

            for function_name in multimodal_functions_1:
                agent_metrics = metrics[agent][function_name]
                if args.mean:
                    mean_a, std_a = agent_metrics["mean"]
                    mean_str = f"{format_number(mean_a)} {pm} {format_number(std_a)}"
                    iqm_mean_a, iqm_std_a = agent_metrics["iqm"]
                    iqm_str = f"{format_number(iqm_mean_a)} {pm} {format_number(iqm_std_a)}"
                    if mean_a < teacher_metrics["mean"][0]:
                        mean_str = f"\\textbf{{{mean_str}}}"
                    if iqm_mean_a < teacher_metrics["iqm"][0]:
                        iqm_str = f"\\textbf{{{iqm_str}}}"

                    row_mean_multi_1.append(mean_str)
                    row_iqm_multi_1.append(iqm_str)
                    if mean_a < mean_min_multi_1[0][2]:
                        mean_min_multi_1[0] = [len(rows_mean_multimodal_1), len(row_mean_multi_1) - 1, mean_a]
                    if iqm_mean_a < iqm_min_multi_1[0][2]:
                        iqm_min_multi_1[0] = [len(rows_iqm_multimodal_1), len(row_iqm_multi_1) - 1, iqm_mean_a]

                if args.lowest:
                    lowest_a = agent_metrics["lowest"]
                    lowest_str = f"{format_number(lowest_a)}"
                    if lowest_a < teacher_metrics["lowest"]:
                        lowest_str = f"\\textbf{{{lowest_str}}}" 
                    row_lowest_multi_1.append(lowest_str)
                    if lowest_a < lowest_min_multi_1[0][2]:
                        lowest_min_multi_1[0] = [len(rows_lowest_multimodal_1), len(row_lowest_multi_1) - 1, lowest_a]

                if args.auc:
                    auc_mean_a, auc_std_a = agent_metrics["auc"]
                    auc_str = f"{format_number(auc_mean_a)} {pm} {format_number(auc_std_a)}"
                    if lowest_a < teacher_metrics["auc"][0]:
                        auc_str = f"\\textbf{{{auc_str}}}" 
                    row_auc_multi_1.append(auc_str)
                    if auc_mean_a < auc_min_multi_1[0][2]:
                        auc_min_multi_1[0] = [len(rows_auc_multimodal_1), len(row_auc_multi_1) - 1, auc_mean_a]

            for function_name in multimodal_functions_2:
                agent_metrics = metrics[agent][function_name]
                if args.mean:
                    mean_a, std_a = agent_metrics["mean"]
                    mean_str = f"{format_number(mean_a)} {pm} {format_number(std_a)}"
                    iqm_mean_a, iqm_std_a = agent_metrics["iqm"]
                    iqm_str = f"{format_number(iqm_mean_a)} {pm} {format_number(iqm_std_a)}"
                    if mean_a < teacher_metrics["mean"][0]:
                        mean_str = f"\\textbf{{{mean_str}}}"
                    if iqm_mean_a < teacher_metrics["iqm"][0]:
                        iqm_str = f"\\textbf{{{iqm_str}}}"

                    row_mean_multi_2.append(mean_str)
                    row_iqm_multi_2.append(iqm_str)
                    if mean_a < mean_min_multi_2[0][2]:
                        mean_min_multi_2[0] = [len(rows_mean_multimodal_2), len(row_mean_multi_2) - 1, mean_a]
                    if iqm_mean_a < iqm_min_multi_2[0][2]:
                        iqm_min_multi_2[0] = [len(rows_iqm_multimodal_2), len(row_iqm_multi_2) - 1, iqm_mean_a]

                if args.lowest:
                    lowest_a = agent_metrics["lowest"]
                    lowest_str = f"{format_number(lowest_a)}"
                    if lowest_a < teacher_metrics["lowest"]:
                        lowest_str = f"\\textbf{{{lowest_str}}}" 
                    row_lowest_multi_2.append(lowest_str)
                    if lowest_a < lowest_min_multi_2[0][2]:
                        lowest_min_multi_2[0] = [len(rows_lowest_multimodal_2), len(row_lowest_multi_2) - 1, lowest_a]

                if args.auc:
                    auc_mean_a, auc_std_a = agent_metrics["auc"]
                    auc_str = f"{format_number(auc_mean_a)} {pm} {format_number(auc_std_a)}"
                    if lowest_a < teacher_metrics["auc"][0]:
                        auc_str = f"\\textbf{{{auc_str}}}" 
                    row_auc_multi_2.append(auc_str)
                    if auc_mean_a < auc_min_multi_2[0][2]:
                        auc_min_multi_2[0] = [len(rows_auc_multimodal_2), len(row_auc_multi_2) - 1, auc_mean_a]

            if args.mean:
                rows_mean_unimodal.append(row_mean_uni)
                rows_iqm_unimodal.append(row_iqm_uni)
                rows_mean_multimodal_1.append(row_mean_multi_1)
                rows_iqm_multimodal_1.append(row_iqm_multi_1)
                rows_mean_multimodal_2.append(row_mean_multi_2)
                rows_iqm_multimodal_2.append(row_iqm_multi_2)
            
            if args.lowest:
                rows_lowest_unimodal.append(row_lowest_uni)
                rows_lowest_multimodal_1.append(row_lowest_multi_1)
                rows_lowest_multimodal_2.append(row_lowest_multi_2)
            
            if args.auc:
                rows_auc_unimodal.append(row_auc_uni)
                rows_auc_multimodal_1.append(row_auc_multi_1)
                rows_auc_multimodal_2.append(row_auc_multi_2)

        if args.mean:
            table_result_path = generate_file_path(base_path / teacher, "mean", "unimodal", args.format)
            table_content = generate_table(rows_mean_unimodal, args.format, mean_min_uni)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "iqm", "unimodal", args.format)
            table_content = generate_table(rows_iqm_unimodal, args.format, iqm_min_uni)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "mean", "multimodal_1", args.format)
            table_content = generate_table(rows_mean_multimodal_1, args.format, mean_min_multi_1)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "iqm", "multimodal_1", args.format)
            table_content = generate_table(rows_iqm_multimodal_1, args.format, iqm_min_multi_1)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "mean", "multimodal_2", args.format)
            table_content = generate_table(rows_mean_multimodal_2, args.format, mean_min_multi_2)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "iqm", "multimodal_2", args.format)
            table_content = generate_table(rows_iqm_multimodal_2, args.format, iqm_min_multi_2)
            with table_result_path.open("w") as f:
                f.write(table_content)
        
        if args.lowest:
            table_result_path = generate_file_path(base_path / teacher, "lowest", "unimodal", args.format)
            table_content = generate_table(rows_lowest_unimodal, args.format, lowest_min_uni)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "lowest", "multimodal_1", args.format)
            table_content = generate_table(rows_lowest_multimodal_1, args.format, lowest_min_multi_1)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "lowest", "multimodal_2", args.format)
            table_content = generate_table(rows_lowest_multimodal_2, args.format, lowest_min_multi_2)
            with table_result_path.open("w") as f:
                f.write(table_content)

        if args.auc:
            table_result_path = generate_file_path(base_path / teacher, "auc", "unimodal", args.format)
            table_content = generate_table(rows_auc_unimodal, args.format, auc_min_uni)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "auc", "multimodal_1", args.format)
            table_content = generate_table(rows_auc_multimodal_1, args.format, auc_min_multi_1)
            with table_result_path.open("w") as f:
                f.write(table_content)
            
            table_result_path = generate_file_path(base_path / teacher, "auc", "multimodal_2", args.format)
            table_content = generate_table(rows_auc_multimodal_2, args.format, auc_min_multi_2)
            with table_result_path.open("w") as f:
                f.write(table_content)