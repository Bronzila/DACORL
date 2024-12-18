from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dacbench.envs.env_utils.function_definitions import (
    Ackley,
    Rastrigin,
    Rosenbrock,
    Sphere,
)
from matplotlib import rc

font = {"family": "serif", "serif": ["Computer Modern Roman"]}

rc("text", usetex=True)
rc("font", **font)

TINY_SIZE = 14
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
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


def load_data(paths: list, num_runs: int = 1000):
    aggregated_df = pd.DataFrame()
    length = 0
    for idx, path in enumerate(paths):
        # Read run data
        df = pd.read_csv(path)
        length += len(df.index)

        df["run_idx"] += idx * num_runs

        df["action"] = df["action"].map(lambda x: 10**x)

        aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
    assert length == len(aggregated_df.index)
    return aggregated_df


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
    labels: list = [],
    title: str = "",
    heterogeneous: bool = False,
) -> None:
    plt.clf()
    run_data_path = []
    if seed:
        run_data_path.append(
            Path(dir_path)
            / "results"
            / agent_type
            / f"{seed}"
            / fidelity
            / "eval_data.csv",
        )
        filename = f"action_{agent_type}_{seed}_{fidelity}"
    elif seed is None:
        dir_path = Path(dir_path)
        teacher_name = dir_path.parents[1].name
        filename = f"action_{labels[1]}_{dir_path.name}_{teacher_name}_aggregate_{fidelity}"
        for path in (Path(dir_path) / "results" / agent_type).rglob(
            "*/eval_data.csv",
        ):
            run_data_path.append(path)

    run_info_path = Path(dir_path, "run_info.json")

    if teacher:
        run_data_teacher_path = Path(dir_path, "aggregated_run_data.csv")
        run_data_teacher = pd.read_csv(run_data_teacher_path)

        teacher_layers = np.sort(run_data_teacher["layer_idx"].unique())

        # Get run info from file
        with Path.open(run_info_path) as file:
            run_info = json.load(file)
            teacher_drawstyle = "default"
            if run_info["agent"]["type"] == "step_decay":
                teacher_drawstyle = "steps-post"

    drawstyle = "default"
    aggregated_df = load_data(
        run_data_path,
        run_info["environment"]["num_runs"],
    )
    agent_layers = np.sort(aggregated_df["layer_idx"].unique())

    if num_runs > 0:
        for data in list(aggregated_df.groupby("run_idx")):
            for layer_idx in agent_layers:
                ax = sns.lineplot(
                    data=data[data["layer_idx"] == layer_idx],
                    x="batch_idx",
                    y="action",
                    drawstyle=drawstyle,
                    label=agent_type.upper(),
                )
            if reward:
                ax2 = ax.twinx()
                sns.lineplot(
                    data=data,
                    x="batch_idx",
                    y="reward",
                    drawstyle=drawstyle,
                    label="Reward",
                    ax=ax2,
                )
            if teacher:
                for layer_idx in teacher_layers:
                    sns.lineplot(
                        data=run_data_teacher[
                            run_data_teacher["layer_idx"] == layer_idx
                        ],
                        x="batch_idx",
                        y="action",
                        drawstyle=teacher_drawstyle,
                        label="Teacher",
                    )

    if aggregate:
        plt.clf()
        if teacher:
            for layer_idx in teacher_layers:
                ax = sns.lineplot(
                    data=run_data_teacher[
                        run_data_teacher["layer_idx"] == layer_idx
                    ],
                    x="batch_idx",
                    y="action",
                    drawstyle=teacher_drawstyle,
                    label=labels[0],
                    errorbar=("pi", 80),
                )

        for layer_idx in agent_layers:
            ax = sns.lineplot(
                data=aggregated_df[aggregated_df["layer_idx"] == layer_idx],
                x="batch_idx",
                y="action",
                drawstyle=drawstyle,
                label=labels[1],
                errorbar=("pi", 80),
            )
        if reward:
            ax2 = ax.twinx()
            sns.lineplot(
                data=aggregated_df,
                x="batch_idx",
                y="reward",
                color="g",
                label="Reward",
                ax=ax2,
            )

        if title:
            plt.title(title, fontsize=BIGGER_SIZE)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set(xlabel="Step $i$", ylabel="Learning Rate $\\alpha_i$")
        # Show or save the plot
        if show:
            plt.show()
        else:
            dir_path = Path(dir_path)
            if not heterogeneous:
                save_path = Path(
                    dir_path.parents[2],  # PROJECT/ToySGD/
                    "figures",
                    dir_path.name,  # FUNCTION/
                    dir_path.parents[1].name,  # TEACHER/
                )
            else:
                save_path = Path(
                    dir_path.parents[1],  # PROJECT/ToySGD/
                    dir_path.parents[1],  # PROJECT/ToySGD/
                    "figures",
                    dir_path.name,  # FUNCTION/
                    dir_path.parents[0].name,  # TEACHER/
                )

            if not save_path.exists():
                save_path.mkdir(parents=True)
            print(f"Saving figure to {save_path / f'{filename}_aggregate.pdf'}")
            plt.savefig(
                save_path / f"{filename}_aggregate.pdf",
                bbox_inches="tight",
            )


def plot_comparison(
    dir_paths: list,
    agent_labels: list,
    teacher: bool = False,
    show: bool = False,
    title: str = "",
    teacher_path: str = "",
    teacher_label: str = "",
    heterogeneous: bool = False,
    metric: str = "f_cur",
) -> None:
    for path, agent_label in zip(dir_paths, agent_labels, strict=True):
        dir_path = Path(path)
        run_info_path = Path(dir_path, "run_info.json")
        with Path.open(run_info_path) as file:
            run_info = json.load(file)

        teacher_name = dir_path.parents[1].name
        if teacher_name not in teacher_name_mapping:  # heterogeneous case
            teacher_name = dir_path.parents[0].name
        func_name = dir_path.name
        teacher_data = None
        agent_data = None
        if teacher:
            if teacher_path == "":
                teacher_path = dir_path / "aggregated_run_data.csv"
            else:
                teacher_path = Path(teacher_path) / "aggregated_run_data.csv"
            teacher_data = load_data([teacher_path], run_info["num_runs"])

            if teacher_label == "":
                teacher_label = teacher_name_mapping[teacher_name]

        result_paths = dir_path.rglob("*/eval_data.csv")

        agent_data = load_data(result_paths, run_info["num_runs"])

        if metric == "f_cur":

            def fill_missing_values(group, metric):
                last_value = group[metric].iloc[-1]

                # Create full run
                all_steps = pd.DataFrame({"batch_idx": range(101)})

                # Merge group with full run
                filled_group = all_steps.merge(
                    group,
                    on="batch_idx",
                    how="left",
                )

                filled_group["run_idx"] = group["run_idx"].iloc[0]
                filled_group[metric] = filled_group[metric].fillna(last_value)

                return filled_group

            agent_required_df = agent_data[[metric, "run_idx", "batch_idx"]]
            teacher_required_df = teacher_data[[metric, "run_idx", "batch_idx"]]

            agent_df_filled = (
                agent_required_df.groupby("run_idx")
                .apply(fill_missing_values, metric)
                .reset_index(drop=True)
            )
            teacher_df_filled = (
                teacher_required_df.groupby("run_idx")
                .apply(fill_missing_values, metric)
                .reset_index(drop=True)
            )
        else:
            teacher_df_filled = teacher_data
            agent_df_filled = agent_data

        if teacher_data is not None:
            ax = sns.lineplot(
                teacher_df_filled,
                x="batch_idx",
                y=metric,
                label=teacher_label,
                errorbar=("ci", 99),
            )
        if agent_data is not None:
            ax = sns.lineplot(
                agent_df_filled,
                x="batch_idx",
                y=metric,
                label=agent_label,
                errorbar=("ci", 99),
            )
    if func_name in ["Rosenbrock", "Sphere"]:
        ax.set_yscale("log")
    else:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    x_label = "Step" if metric == "f_cur" else "Batch"
    y_label = "$f(\\theta_i)$" if metric == "f_cur" else "Validation Acc."
    ax.set_xlabel(f"{x_label} $i$")
    ax.set_ylabel(y_label)
    if title:
        plt.title(title, fontsize=BIGGER_SIZE)

    if show:
        plt.show()
    elif metric == "f_cur":
        dir_path = Path(dir_paths[0])
        save_path = Path(
            dir_path.parents[2],  # PROJECT/ToySGD/
            "figures",
            "comparison",
            dir_path.name,
        )
        if not save_path.exists():
            save_path.mkdir(parents=True)
        file_name = (
            f"comparison_{dir_path.name}_{teacher_name}.pdf"
            if len(dir_paths) == 1
            else "trajectory_comparison_agents.pdf"
        )
        print(f"Saving figure to {save_path / file_name}")
        plt.savefig(save_path / file_name, bbox_inches="tight")
    else:
        if not heterogeneous:
            dir_path = Path(dir_paths[0])
            save_path = Path(
                dir_path.parents[1],  # PROJECT/ToySGD/
                "figures",
                "comparison",
            )
            if not save_path.exists():
                save_path.mkdir(parents=True)
            file_name = (
                f"comparison_{dir_path.parents[0].name}.pdf"
                if len(dir_paths) == 1
                else "trajectory_comparison_agents.pdf"
            )
        else:
            dir_path = Path(dir_paths[0])
            save_path = Path(
                dir_path.parents[0],  # PROJECT/ToySGD/
                "figures",
                "comparison",
            )
            if not save_path.exists():
                save_path.mkdir(parents=True)
            file_name = (
                f"comparison_{dir_path.name}.pdf"
                if len(dir_paths) == 1
                else "trajectory_comparison_agents.pdf"
            )

        print(f"Saving figure to {save_path / file_name}")
        plt.savefig(save_path / file_name, bbox_inches="tight")


def plot_actions_sgd(
    dir_path: Path,
    agent_type: str,
    fidelity: int,
    show: bool = False,
    num_runs: int = 1,
    aggregate: bool = True,
    teacher: bool = True,
    reward: bool = False,
    labels: list = [],
    title: str = "",
    heterogeneous: bool = False,
) -> None:
    plt.clf()
    run_data_path = []

    teacher_name = (
        dir_path.parents[0].name if not heterogeneous else dir_path.name
    )
    teacher_id = dir_path.name if not heterogeneous else 0
    filename = f"action_{teacher_name}_{teacher_id}_aggregate"
    for path in (dir_path / "results" / agent_type).rglob(
        "*/eval_data.csv",
    ):
        if path.parent.name == str(fidelity):
            run_data_path.append(path)

    run_info_path = dir_path / "run_info.json"

    if teacher:
        run_data_teacher_path = dir_path / "aggregated_run_data.csv"
        run_data_teacher = pd.read_csv(run_data_teacher_path)
        run_data_teacher["action"] = 10 ** run_data_teacher["action"]

        # Get run info from file
        with Path.open(run_info_path) as file:
            run_info = json.load(file)
            teacher_drawstyle = "default"
            if run_info["agent"]["type"] == "step_decay" and not heterogeneous:
                teacher_drawstyle = "steps-post"

    drawstyle = "default"
    aggregated_df = load_data(run_data_path, run_info["num_runs"])
    agent_layers = np.sort(aggregated_df["layer_idx"].unique())

    if num_runs > 0:
        for data in list(aggregated_df.groupby("run_idx")):
            for layer_idx in agent_layers:
                ax = sns.lineplot(
                    data=data[data["layer_idx"] == layer_idx],
                    x="batch_idx",
                    y="action",
                    drawstyle=drawstyle,
                    label=agent_type.upper(),
                )
            if reward:
                ax2 = ax.twinx()
                sns.lineplot(
                    data=data,
                    x="batch_idx",
                    y="reward",
                    drawstyle=drawstyle,
                    label="Reward",
                    ax=ax2,
                )
            if teacher:
                sns.lineplot(
                    data=run_data_teacher[run_data_teacher["layer_idx"] == 0],
                    x="batch_idx",
                    y="action",
                    drawstyle=teacher_drawstyle,
                    label="Teacher",
                )

    if aggregate:
        plt.clf()
        if teacher:
            # Teacher does not differ for different layers
            ax = sns.lineplot(
                data=run_data_teacher[run_data_teacher["layer_idx"] == 0],
                x="batch_idx",
                y="action",
                drawstyle=teacher_drawstyle,
                label=labels[0],
                errorbar=("pi", 80),
            )
        for layer_idx in agent_layers:
            ax = sns.lineplot(
                data=aggregated_df[aggregated_df["layer_idx"] == layer_idx],
                x="batch_idx",
                y="action",
                drawstyle=drawstyle,
                label=labels[1] + f"-{layer_idx}",
                errorbar=("pi", 80),
            )
        if reward:
            ax2 = ax.twinx()
            sns.lineplot(
                data=aggregated_df,
                x="batch_idx",
                y="reward",
                color="g",
                label="Reward",
                ax=ax2,
            )

        if title:
            plt.title(title, fontsize=BIGGER_SIZE)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set(xlabel="Batch $i$", ylabel="Learning Rate $\\alpha_i$")
        # Show or save the plot
        if show:
            plt.show()
        else:
            if not heterogeneous:
                save_path = Path(
                    dir_path.parents[1],  # PROJECT/ToySGD/
                    "figures",
                    "action",
                )
            else:
                save_path = Path(
                    dir_path.parents[0],  # PROJECT/ToySGD/
                    "figures",
                    "action",
                )

            if not save_path.exists():
                save_path.mkdir(parents=True)
            print(f"Saving figure to {save_path / f'{filename}.pdf'}")
            plt.savefig(save_path / f"{filename}.pdf", bbox_inches="tight")


def plot_teacher_actions(
    dir_path: str | Path,
    show: bool = False,
    reward: bool = False,
    single_plot: bool = False,
    function: str = "Ackley",
    heterogeneous: bool = False,
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
                for eval_data_path in (path / "results" / "td3_bc").rglob(
                    "*/eval_data.csv",
                ):
                    run_data_path.append(eval_data_path)
                aggregated_df = pd.DataFrame()
                for seed_path in run_data_path:
                    # Read run data
                    df = pd.read_csv(seed_path)

                    df["action"] = df["action"].map(lambda x: 10**x)

                    aggregated_df = pd.concat(
                        [aggregated_df, df],
                        ignore_index=True,
                    )
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

    for run_info, df in zip(run_infos, dfs, strict=True):
        print(run_info["agent"]["id"])
        ax = sns.lineplot(
            data=df,
            x="batch_idx",
            y="action",
            drawstyle=run_info["draw_style"],
            label=run_info["agent"]["id"],
            legend=False,
        )
        if reward:
            ax2 = ax.twinx()
            sns.lineplot(
                data=df,
                x="batch_idx",
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
            print(
                f"Saving figure to {save_path / 'action_teacher_single_plot.pdf'}",
            )
            plt.savefig(
                save_path / "action_teacher_single_plot.pdf",
                bbox_inches="tight",
            )
        else:
            dir_path = Path(dir_path)
            if not heterogeneous:
                save_path = Path(
                    dir_path.parents[1],  # PROJECT/ToySGD/
                    "figures",
                    "action",
                )
            else:
                save_path = Path(
                    dir_path.parents[0],  # PROJECT/ToySGD/
                    "figures",
                    "action",
                )

            filename = "action_teacher_multi"
            if not save_path.exists():
                save_path.mkdir(parents=True)
            print(f"Saving figure to {save_path / f'{filename}.pdf'}")
            plt.savefig(save_path / f"{filename}.pdf", bbox_inches="tight")
