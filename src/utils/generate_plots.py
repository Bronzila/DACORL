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

        plt.savefig(save_path / f"point_traj_{idx}.svg")


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
        aggregated_df = pd.DataFrame()
        # if agent_name == "edac" or agent_name == "":
        for seed_path in agent_paths:
            print(seed_path)
            # Read run data
            df = pd.read_csv(seed_path)

            df["action"] = df["action"].map(lambda x: 10**x)

            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)

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
        plt.savefig(save_path / f"{filename}_aggregate.svg")


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

    aggregated_df = pd.DataFrame()
    drawstyle = "default"
    for seed_path in run_data_path:
        # Read run data
        df = pd.read_csv(seed_path)

        df["action"] = df["action"].map(lambda x: 10**x)

        aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)

        if num_runs > 0:
            for data in list(df.groupby("run")):
                # Adjust action value from the DataFrame
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

                # # Show or save the plot
                # if show:
                #         dir_path,
                #         "figures",
                #         "action",

                #     if not save_path.exists():

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
            plt.savefig(save_path / f"{filename}_aggregate.svg")

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

        for path in paths:
            run_info_path = Path(path, "run_info.json")
            # Get run info from file
            with Path.open(run_info_path) as file:
                run_info = json.load(file)
                drawstyle = "default"
                if run_info["agent"]["type"] == "step_decay":
                    drawstyle = "steps-post"

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
            drawstyle = "default"
            if run_info["agent"]["type"] == "step_decay":
                drawstyle = "steps-post"

        data_path = Path(dir_path, "aggregated_run_data.csv")
        # Read run data
        df = pd.read_csv(data_path)

        df["action"] = df["action"].map(lambda x: 10**x)

        dfs.append(df)
        run_infos.append(run_info)

    for run_info, df in zip(run_infos, dfs):
        ax = sns.lineplot(
            data=df,
            x="batch",
            y="action",
            drawstyle=drawstyle,
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

    ax.figure.legend()
    plt.title(f"Actions of {run_infos[0]['agent']['type']} agent")
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
            print(f"Saving figure to {save_path / 'action_teacher_single_plot.svg'}")
            plt.savefig(save_path / "action_teacher_single_plot.svg")
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
            plt.savefig(save_path / f"action_teacher_{teach_id}_aggregate.svg")