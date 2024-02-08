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
    if function_name == "Rastrigin":
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
        sns.scatterplot(
            x=x_values,
            y=y_values,
            color="red",
            hue=data["batch"],
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


def plot_multiple_optim_trace(dir_path, agent_path, show=False, num_runs=1):
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

    # Create a meshgrid for plotting the Rastrigin function
    x_range = np.linspace(lower_bound, upper_bound, 100)
    y_range = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective_function([torch.Tensor(X), torch.Tensor(Y)]).numpy()

    # Use logarithmically spaced contour levels for varying detail
    if function_name == "Rastrigin":
        contour_levels = np.logspace(-1, 2.5, 10)
    else:
        contour_levels = 10

    paths = [
        Path("data", "momentum0", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum10", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum20", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum30", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum40", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum50", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum60", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum70", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum80", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum85", "step_decay_1", "ToySGD", "Rastrigin"),
        Path("data", "momentum90", "step_decay_1", "ToySGD", "Rastrigin"),
    ]

    momentum = [
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.85,
        0.9,
    ]

    colors = sns.color_palette("Spectral", len(paths))

    for run_id in range(num_runs):
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

        # Add minimum
        min_point = problem.x_min.tolist()
        sns.scatterplot(
            x=[min_point[0]],
            y=[min_point[1]],
            color="green",
            s=100,
            marker="*",
            label=f"min at {min_point}",
            zorder=10,
        )
        for id, path in enumerate(paths):
            data_path = path / "aggregated_run_data.csv"
            # Read run data
            df = pd.read_csv(data_path)

            # Group data by runs
            grouped_df = df.groupby("run")

            # for idx, data in list(grouped_df)[:num_runs]:
            data = list(grouped_df)[run_id : run_id + 1][0][1]

            # Extract x and y values from the DataFrame
            x_values = data["x_cur"].apply(lambda coord: eval(coord)[0])
            y_values = data["x_cur"].apply(lambda coord: eval(coord)[1])

            # Plot the points from the DataFrame
            sns.scatterplot(
                x=x_values,
                y=y_values,
                label=f"{momentum[id]}",
                color=colors[len(momentum) - 1 - id],
                zorder=10,
                alpha=0.5,
            )

        # Add labels and a legend
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{function_name} Function with Optimization Trace")
        plt.legend(prop={"size": 5})

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

        plt.savefig(save_path / f"point_traj_{run_id}.png")


def plot_actions(dir_path, agent_path=None, show=False):
    plt.clf()
    # Get paths
    # Get paths
    if not agent_path:
        run_data_path = Path(dir_path, "aggregated_run_data.csv")
    else:
        run_data_path = Path(dir_path, agent_path, "eval_data.csv")
    run_info_path = Path(dir_path, "run_info.json")

    # Read run data
    df = pd.read_csv(run_data_path)

    # Get run info from file
    with Path.open(run_info_path) as file:
        run_info = json.load(file)
        drawstyle = "default"
        if run_info["agent"]["type"] == "step_decay":
            drawstyle = "steps-post"

    # Remove initial row
    df = df.drop(df[df.batch == 0].index)

    # Adjust action value from the DataFrame
    df["action"] = 10 ** df["action"]

    sns.lineplot(data=df, x="batch", y="action", drawstyle=drawstyle)

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

        plt.savefig(save_path / "action.svg")
