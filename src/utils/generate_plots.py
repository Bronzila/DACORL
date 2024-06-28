from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dacbench.envs.env_utils.function_definitions import (
    Ackley,
    Rastrigin,
    Rosenbrock,
    Sphere,
)
from matplotlib import rc

rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})

TINY_SIZE = 14
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=TINY_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=TINY_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

teacher_name_mapping = {
    "exponential_decay": "Exponential Decay",
    "step_decay": "Step Decay",
    "sgdr": "SGDR",
    "constant": "Constant",
}

def get_problem_from_name(function_name) -> Any:
    if function_name == "Rosenbrock":
        problem = Rosenbrock()
    elif function_name == "Rastrigin":
        problem = Rastrigin()
    elif function_name == "Ackley":
        problem = Ackley()
    elif function_name == "Sphere":
        problem = Sphere()
    return problem

def load_data(paths: list):
    aggregated_df = pd.DataFrame()
    for path in paths:
        # Read run data
        df = pd.read_csv(path)

        df["action"] = df["action"].map(lambda x: 10**x)

        aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
    return aggregated_df


def plot_optimization_trace(
    dir_path,
    agent_path=None,
    show=False,
    num_runs=1,
) -> None:
    # Get paths
    if not agent_path:
        run_data_path = Path(dir_path, "aggregated_run_data.csv")
    else:
        run_data_path = Path(dir_path, agent_path, "eval_data.csv")
    run_info_path = Path(dir_path, "run_info.json")

    # Get run info from file
    with Path.open(run_info_path) as file:
        run_info = json.load(file)
        env_info = run_info["environment"]
    function_name = env_info["function"]
    lower_bound = env_info["low"]
    upper_bound = env_info["high"]

    # Define problem
    problem = get_problem_from_name(function_name)
    objective_function = problem.objective_function

    # Read run data
    df = pd.read_csv(run_data_path)

    # Create a meshgrid for plotting the Rastrigin function
    x_range = np.linspace(lower_bound, upper_bound, 100)
    y_range = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective_function([torch.Tensor(X), torch.Tensor(Y)]).numpy()

    # Use logarithmically spaced contour levels for varying detail
    if function_name == "Rosenbrock":
        contour_levels = np.logspace(-3, 3.6, 30)
    else:
        contour_levels = 10

    # Group data by runs
    grouped_df = df.groupby("run")

    for idx, data in list(grouped_df)[:num_runs]:
        plt.clf()
        # Plot the function
        contour_plot = plt.contourf(
            X,
            Y,
            Z,
            levels=contour_levels,
            cmap="viridis",
            zorder=5,
        )

        # Extract x and y values from the DataFrame
        x_values = data["x_cur"].apply(lambda coord: eval(coord)[0])
        y_values = data["x_cur"].apply(lambda coord: eval(coord)[1])

        # Plot the points from the DataFrame
        colors = np.arange(len(x_values))
        sns.scatterplot(
            x=x_values,
            y=y_values,
            c=colors,
            cmap="spring",
            label="Trace",
            zorder=10,
        )

        # Add minimum
        min_point = problem.x_min.tolist()
        sns.scatterplot(
            x=[min_point[0]],
            y=[min_point[1]],
            color="green",
            s=100,
            marker="*",
            label=f"Minimum Point {min_point}",
            zorder=10,
        )

        # Add labels and a legend
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{function_name} Function with Optimization Trace")
        plt.legend()

        # Add a colorbar to indicate function values
        colorbar = plt.colorbar(contour_plot)
        colorbar.set_label("Objective Function Value")

        # Show or save the plot
        if show:
            plt.show()
        else:
            save_path = Path(
                dir_path,
                "figures",
                "point_traj",
            )

        if not save_path.exists():
            save_path.mkdir(parents=True)

        plt.savefig(save_path / f"point_traj_{idx}.svg", bbox_inches="tight")


def plot_type(
    plot_type: str,
    dir_path: str,
    fidelity: int,
    seed: int | None,
    show: bool = False,
    teacher: bool = True,
) -> None:
    plt.clf()

    dir_path = Path(dir_path)

    run_data_path = []
    agent_names = []
    if type(seed) is int:
        for path in (dir_path / "results").rglob(
            f"{seed}/{fidelity}/eval_data.csv",
        ):
            run_data_path.append(path)
        filename = f"{plot_type}_{fidelity}_{seed}"
    elif seed is None:
        for agent in (Path(dir_path) / "results").glob("*"):
            agent_runs = []
            agent_names.append(agent.name)
            for path in agent.rglob("*/eval_data.csv"):
                agent_runs.append(path)
            run_data_path.append(agent_runs)
        filename = f"{plot_type}_{fidelity}"

    run_info_path = Path(dir_path, "run_info.json")

    if teacher:
        run_data_teacher_path = Path(dir_path, "aggregated_run_data.csv")
        run_data_teacher = pd.read_csv(run_data_teacher_path)
        completed_runs_ids = run_data_teacher[run_data_teacher["batch"] == 99][
            "run"
        ].unique()
        completed_runs = run_data_teacher[
            run_data_teacher["run"].isin(completed_runs_ids)
        ]
        single_teacher_run = completed_runs[
            completed_runs["run"] == completed_runs_ids[0]
        ]
        single_teacher_run["action"] = 10 ** single_teacher_run["action"]

        # Get run info from file
        with Path.open(run_info_path) as file:
            run_info = json.load(file)
            teacher_drawstyle = "default"
            if run_info["agent"]["type"] == "step_decay":
                teacher_drawstyle = "steps-post"

    if teacher:
        ax = sns.lineplot(
            data=single_teacher_run,
            x="batch",
            y=plot_type,
            drawstyle=teacher_drawstyle,
            label=dir_path.parents[1].name,
        )
    for agent_name, agent_paths in zip(agent_names, run_data_path):
        aggregated_df = load_data(agent_paths)

        print(f"Max {agent_name}: {aggregated_df['action'].max()}")
        print(f"Is inf: {any(np.isinf(aggregated_df['action']))}")
        print(f"Is nan: {any(np.isnan(aggregated_df['action']))}")

        sns.lineplot(
            data=aggregated_df,
            x="batch",
            y=plot_type,
            ax=ax,
            label=agent_name.upper(),
        )

    if plot_type == "action":
        ax.set_ylim(0, 1.0)
    ax.set_title(f"{plot_type} on {dir_path.name}")
    # Show or save the plot
    if show:
        plt.show()
    else:
        save_path = Path(
            dir_path.parents[2],  # PROJECT/ToySGD/
            "figures",
            dir_path.name,  # FUNCTION/
            dir_path.parents[1].name,  # TEACHER/
        )

        if not save_path.exists():
            save_path.mkdir(parents=True)
        print(f"Saving figure to {save_path / f'{filename}_aggregate.svg'}")
        plt.savefig(save_path / f"{filename}_aggregate.svg", bbox_inches="tight")


def plot_actions(
    dir_path: str,
    agent_type: str,
    fidelity: int,
    seed: int | None,
    show: bool = False,
    num_runs: int = 1,
    aggregate: bool = True,
    teacher: bool = True,
    reward: bool = False,
) -> None:
    plt.clf()
    run_data_path = []
    if seed:
        run_data_path.append(
            Path(dir_path)
            / "results"
            / agent_type
            / seed
            / fidelity
            / "eval_data.csv",
        )
        filename = f"action_{agent_type}_{seed}_{fidelity}"
    elif seed is None:
        filename = f"action_{agent_type}_aggregate_{fidelity}"
        for path in (Path(dir_path) / "results" / agent_type).rglob(
            "*/eval_data.csv",
        ):
            run_data_path.append(path)

    run_info_path = Path(dir_path, "run_info.json")

    # Get exemplatory teacher run for plotting
    if teacher:
        run_data_teacher_path = Path(dir_path, "aggregated_run_data.csv")
        run_data_teacher = pd.read_csv(run_data_teacher_path)
        completed_runs_ids = run_data_teacher[run_data_teacher["batch"] == 99][
            "run"
        ].unique()
        completed_runs = run_data_teacher[
            run_data_teacher["run"].isin(completed_runs_ids)
        ]
        single_teacher_run = completed_runs[
            completed_runs["run"] == completed_runs_ids[0]
        ]
        single_teacher_run["action"] = 10 ** single_teacher_run["action"]

        # Get run info from file
        with Path.open(run_info_path) as file:
            run_info = json.load(file)
            teacher_drawstyle = "default"
            if run_info["agent"]["type"] == "step_decay":
                teacher_drawstyle = "steps-post"

    drawstyle = "default"
    aggregated_df = load_data(run_data_path)

    if num_runs > 0:
        for data in list(aggregated_df.groupby("run")):
            plt.clf()
            ax = sns.lineplot(
                data=data,
                x="batch",
                y="action",
                drawstyle=drawstyle,
                label=agent_type.upper(),
            )
            if reward:
                ax2 = ax.twinx()
                sns.lineplot(
                    data=data,
                    x="batch",
                    y="reward",
                    drawstyle=drawstyle,
                    label="Reward",
                    ax=ax2,
                )
            if teacher:
                sns.lineplot(
                    data=single_teacher_run,
                    x="batch",
                    y="action",
                    drawstyle=teacher_drawstyle,
                    label="Teacher",
                )

    if aggregate:
        plt.clf()
        ax = sns.lineplot(
            data=aggregated_df,
            x="batch",
            y="action",
            drawstyle=drawstyle,
            label=agent_type.upper(),
            legend=False,
        )
        if reward:
            ax2 = ax.twinx()
            sns.lineplot(
                data=aggregated_df,
                x="batch",
                y="reward",
                color="g",
                label="Reward",
                ax=ax2,
                legend=False,
            )
        if teacher:
            sns.lineplot(
                data=single_teacher_run,
                x="batch",
                y="action",
                drawstyle=teacher_drawstyle,
                label="Teacher",
                ax=ax,
                legend=False,
            )

        ax.figure.legend()
        ax.set(xlabel="Step", ylabel="Learning Rate")
        # Show or save the plot
        if show:
            plt.show()
        else:
            dir_path = Path(dir_path)
            save_path = Path(
                dir_path.parents[2],  # PROJECT/ToySGD/
                "figures",
                dir_path.name,  # FUNCTION/
                dir_path.parents[1].name,  # TEACHER/
            )

            if not save_path.exists():
                save_path.mkdir(parents=True)
            print(f"Saving figure to {save_path / f'{filename}_aggregate.svg'}")
            plt.savefig(save_path / f"{filename}_aggregate.svg", bbox_inches="tight")

def plot_comparison(
    dir_paths: list,
    agent_labels: list,
    teacher: bool = False,
    show: bool = False,
) -> None:
    for dir_path, agent_label in zip(dir_paths, agent_labels):
        dir_path = Path(dir_path)
        teacher_name = dir_path.parents[1].name
        func_name = dir_path.name
        teacher_data = None
        agent_data = None
        if teacher:
            teacher_path = dir_path / "aggregated_run_data.csv"
            teacher_data = load_data([teacher_path])

        result_paths = dir_path.rglob("*/eval_data.csv")
        agent_data = load_data(result_paths)

        if teacher_data is not None:
            ax = sns.lineplot(teacher_data,
                              x="batch",
                              y="f_cur",
                              label=teacher_name_mapping[teacher_name])
        if agent_data is not None:
            ax = sns.lineplot(
                agent_data,
                x="batch",
                y="f_cur",
                label=agent_label,
            )
    if func_name in ["Rosenbrock", "Sphere"]:
        ax.set_yscale("log")
    else:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Step $i$")
    ax.set_ylabel("$f(\\theta_i)$")

    if show:
        plt.show()
    else:
        dir_path = Path(dir_paths[0])
        save_path = Path(
            dir_path.parents[2],  # PROJECT/ToySGD/
            "figures",
            "comparison",
            dir_path.name
        )
        if not save_path.exists():
            save_path.mkdir(parents=True)
        file_name = f"trajectory_comparison_agent_teacher_{teacher_name}.pdf" if len(dir_paths) == 1 else "trajectory_comparison_agents.pdf"
        print(f"Saving figure to {save_path / file_name}")
        plt.savefig(save_path / file_name, bbox_inches="tight")

def plot_comparison(
    dir_paths: list,
    agent_labels: list,
    teacher: bool = False,
    show: bool = False,
) -> None:
    for dir_path, agent_label in zip(dir_paths, agent_labels):
        dir_path = Path(dir_path)
        teacher_name = dir_path.parents[1].name
        func_name = dir_path.name
        teacher_data = None
        agent_data = None
        if teacher:
            teacher_path = dir_path / "aggregated_run_data.csv"
            teacher_data = load_data([teacher_path])

        result_paths = dir_path.rglob("*/eval_data.csv")
        agent_data = load_data(result_paths)

        if teacher_data is not None:
            ax = sns.lineplot(teacher_data,
                              x="batch",
                              y="f_cur",
                              label=teacher_name_mapping[teacher_name])
        if agent_data is not None:
            ax = sns.lineplot(
                agent_data,
                x="batch",
                y="f_cur",
                label=agent_label,
            )
    if func_name in ["Rosenbrock", "Sphere"]:
        ax.set_yscale("log")
    else:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Step $i$")
    ax.set_ylabel("$f(\\theta_i)$")

    if show:
        plt.show()
    else:
        dir_path = Path(dir_paths[0])
        save_path = Path(
            dir_path.parents[2],  # PROJECT/ToySGD/
            "figures",
            "comparison",
            dir_path.name
        )
        if not save_path.exists():
            save_path.mkdir(parents=True)
        file_name = f"trajectory_comparison_agent_teacher_{teacher_name}.pdf" if len(dir_paths) == 1 else "trajectory_comparison_agents.pdf"
        print(f"Saving figure to {save_path / file_name}")
        plt.savefig(save_path / file_name, bbox_inches="tight")

def plot_teacher_actions(
    dir_path: str,
    show: bool = False,
    reward: bool = False,
    single_plot: bool = False,
    function: str = "Ackley",
) -> None:
    plt.clf()
    dfs = []
    run_infos = []
    if single_plot:
        paths = Path(dir_path).rglob(f"*/{function}")

        for path in sorted(paths):
            run_info_path = Path(path, "run_info.json")
            # Get run info from file
            with Path.open(run_info_path) as file:
                run_info = json.load(file)
                run_info["draw_style"] = "default"
                if run_info["agent"]["type"] == "step_decay":
                    run_info["draw_style"] = "steps-post"

            if path.parents[0].name == "x_learned":
                run_data_path = []
                for path in (path / "results" / "td3_bc").rglob(
                    "*/eval_data.csv",
                ):
                    run_data_path.append(path)
                aggregated_df = pd.DataFrame()
                for seed_path in run_data_path:
                    # Read run data
                    df = pd.read_csv(seed_path)

                    df["action"] = df["action"].map(lambda x: 10**x)

                    aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
                run_info["draw_style"] = "default"
                dfs.append(aggregated_df)
                run_infos.append(run_info)
            else:
                data_path = Path(path, "aggregated_run_data.csv")
                # Read run data
                df = pd.read_csv(data_path)

                df["action"] = df["action"].map(lambda x: 10**x)

                dfs.append(df)
                run_infos.append(run_info)
    else:
        run_info_path = Path(dir_path, "run_info.json")
        # Get run info from file
        with Path.open(run_info_path) as file:
            run_info = json.load(file)
            run_info["draw_style"] = "default"
            if run_info["agent"]["type"] == "step_decay":
                run_info["draw_style"] = "steps-post"

        data_path = Path(dir_path, "aggregated_run_data.csv")
        # Read run data
        df = pd.read_csv(data_path)

        df["action"] = df["action"].map(lambda x: 10**x)

        dfs.append(df)
        run_infos.append(run_info)

    for run_info, df in zip(run_infos, dfs):
        print(run_info["agent"]["id"])
        ax = sns.lineplot(
            data=df,
            x="batch",
            y="action",
            drawstyle=run_info["draw_style"],
            label=run_info["agent"]["id"],
            legend=False,
        )
        if reward:
            ax2 = ax.twinx()
            sns.lineplot(
                data=df,
                x="batch",
                y="reward",
                color="g",
                label="Reward",
                ax=ax2,
                legend=False,
            )

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.legend(prop={"size": 12})
    plt.ticklabel_format(axis="y", style="sci")
    # Show or save the plot
    if show:
        plt.show()
    else:
        dir_path = Path(dir_path)
        if single_plot:
            save_path = Path(
                dir_path.parents[0],  # PROJECT/ToySGD/
                "figures",
                function,  # FUNCTION/
                dir_path.name,  # TEACHER/
            )
            if not save_path.exists():
                save_path.mkdir(parents=True)
            print(f"Saving figure to {save_path / 'action_teacher_single_plot.pdf'}")
            plt.savefig(save_path / "action_teacher_single_plot.pdf", bbox_inches="tight")
        else:
            save_path = Path(
                dir_path.parents[2],  # PROJECT/ToySGD/
                "figures",
                dir_path.name,  # FUNCTION/
                dir_path.parents[1].name,  # TEACHER/
            )
            teach_id = dir_path.parents[0].name
            if not save_path.exists():
                save_path.mkdir(parents=True)
            print(f"Saving figure to {save_path / f'action_teacher_{teach_id}_aggregate.svg'}")
            plt.savefig(save_path / f"action_teacher_{teach_id}_aggregate.svg", bbox_inches="tight")


##############################
# PLEASE IGNORE
##############################
# I only used this for my thesis - in case I need to recreate the plot again :D

def plot_methodology():
    rast_path = "data_single_64/ToySGD/exponential_decay/0/Rastrigin/aggregated_run_data.csv"
    data = load_data([rast_path])

    last_steps = data[data["batch"] == 100]
    best_run_id = last_steps.loc[last_steps["f_cur"].idxmax()]["run"]
    worst_run_id = last_steps.loc[last_steps["f_cur"].idxmin()]["run"]

    run_1 = data[data["run"] == best_run_id]
    run_2 = data[data["run"] == worst_run_id]

    ax = sns.lineplot(run_1,
                        x="batch",
                        y="f_cur",
                        label="Best starting point")
    ax = sns.lineplot(run_2,
                              x="batch",
                              y="f_cur",
                              label="Worst starting point")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Step $i$")
    ax.set_ylabel("$f(\\theta_i)$")
    plt.savefig(Path("best_vs_worst.pdf"), bbox_inches="tight")