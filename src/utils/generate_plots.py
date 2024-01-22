from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dacbench.envs.env_utils.function_definitions import Rastrigin, Rosenbrock


def get_problem_from_name(function_name):
    if function_name == "Rosenbrock":
        problem = Rosenbrock()
    elif function_name == "Rastrigin":
        problem = Rastrigin()
    return problem


def plot_optimization_trace(dir_path, show=False):
    plt.clf()
    # Get paths
    run_data_path = os.path.join(dir_path, "aggregated_run_data.csv")
    run_info_path = os.path.join(dir_path, "run_info.json")

    # Get run info from file
    with open(run_info_path) as file:
        run_info = json.load(file)
    function_name = run_info["function"]
    lower_bound = run_info["lower_bound"]
    upper_bound = run_info["upper_bound"]

    # Define problem
    problem = get_problem_from_name(function_name)
    objective_function = problem.objective_function

    # Read run data
    df = pd.read_csv(run_data_path)

    # Extract x and y values from the DataFrame
    x_values = df["x_cur"].apply(lambda coord: eval(coord)[0])
    y_values = df["x_cur"].apply(lambda coord: eval(coord)[1])

    # Create a meshgrid for plotting the Rastrigin function
    x_range = np.linspace(lower_bound, upper_bound, 100)
    y_range = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective_function([torch.Tensor(X), torch.Tensor(Y)]).numpy()

    # Use logarithmically spaced contour levels for varying detail
    contour_levels = np.logspace(0, 5, 10)

    # Plot the function
    contour_plot = plt.contour(
        X,
        Y,
        Z,
        levels=contour_levels,
        cmap="viridis",
        zorder=5,
    )
    # Plot the points from the DataFrame
    sns.scatterplot(
        x=x_values,
        y=y_values,
        color="red",
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
        save_path = os.path.join(
            dir_path,
            "figures",
            "point_traj",
            function_name,
        )

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        plt.savefig(os.path.join(save_path, "point_traj.svg"))


def plot_actions(dir_path, show=False):
    plt.clf()
    # Get paths
    run_data_path = os.path.join(dir_path, "aggregated_run_data.csv")
    run_info_path = os.path.join(dir_path, "run_info.json")

    # Read run data
    df = pd.read_csv(run_data_path)

    # Get run info from file
    with open(run_info_path) as file:
        run_info = json.load(file)
        function_name = run_info["function"]
        drawstyle = "default"
        if run_info["agent_type"] == "step_decay":
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
        save_path = os.path.join(
            dir_path,
            "figures",
            "action",
            function_name,
        )

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        plt.savefig(os.path.join(save_path, "action.svg"))
