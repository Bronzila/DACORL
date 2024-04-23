from __future__ import annotations

import json
from pathlib import Path

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


def get_problem_from_name(function_name):
    if function_name == "Rosenbrock":
        problem = Rosenbrock()
    elif function_name == "Rastrigin":
        problem = Rastrigin()
    elif function_name == "Ackley":
        problem = Ackley()
    elif function_name == "Sphere":
        problem = Sphere()
    return problem


def plot_optimization_trace(dir_path, agent_path=None, show=False, num_runs=1):
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


def plot_actions(
    dir_path,
    agent_path=None,
    show=False,
    num_runs=1,
    aggregate=True,
    teacher=True,
):
    plt.clf()
    # Get paths
    if not agent_path:
        run_data_path = Path(dir_path, "aggregated_run_data.csv")
        filename = "action"
    else:
        run_data_path = Path(dir_path, agent_path, "eval_data.csv")

        tmp = agent_path.split("/")
        if tmp[0] is None:
            agent_type = tmp[2]
            fidelity = tmp[3]
        else:
            agent_type = tmp[1]
            fidelity = tmp[2]
        filename = f"action_{agent_type}_{fidelity}"

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

    # Read run data
    df = pd.read_csv(run_data_path)

    drawstyle = "default"
    # Get run info from file
    with Path.open(run_info_path) as file:
        run_info = json.load(file)
        teacher_drawstyle = "default"
        if run_info["agent"]["type"] == "step_decay":
            teacher_drawstyle = "steps-post"

    label = "Agent"
    if agent_path is None:
        label = "Teacher"
        drawstyle = teacher_drawstyle

    # Group data by runs
    grouped_df = df.groupby("run")

    aggregated_data = pd.DataFrame()

    for idx, data in list(grouped_df):
        # Adjust action value from the DataFrame
        data["action"] = 10 ** data["action"]

        aggregated_data = aggregated_data.append(data)

        if idx < num_runs:
            plt.clf()
            sns.lineplot(
                data=data,
                x="batch",
                y="action",
                drawstyle=drawstyle,
                label=label,
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
        sns.lineplot(
            data=aggregated_data,
            x="batch",
            y="action",
            drawstyle=drawstyle,
            label=label,
        )
        if teacher:
            sns.lineplot(
                data=single_teacher_run,
                x="batch",
                y="action",
                drawstyle=teacher_drawstyle,
                label="Teacher",
            )

        # Show or save the plot
        if show:
            plt.show()
        else:
            save_path = Path(
                dir_path,
                "figures",
                "action",
            )

            if not save_path.exists():
                save_path.mkdir(parents=True)
            print(f"Saving figure to {save_path / f'{filename}_aggregate.svg'}")
            plt.savefig(save_path / f"{filename}_aggregate.svg")
