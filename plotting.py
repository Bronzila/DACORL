import argparse
import json
from pathlib import Path

from src.utils.generate_plots import (
    plot_actions,
    plot_actions_sgd,
    plot_comparison,
    plot_optimization_trace,
    plot_teacher_actions,
    plot_type,
)

map_teacher_label = {
    "exponential_decay": "Exponential Decay",
    "step_decay": "Step Decay",
    "sgdr": "SGDR",
    "constant": "Constant",
}

map_agent_label = {
    "bc": "BC",
    "td3_bc": "TD3+BC",
    "cql": "CQL",
    "awac": "AWAC",
    "edac": "EDAC",
    "sac_n": "SAC-N",
    "lb_sac": "LB-SAC",
    "iql": "IQL",
    "td3": "TD3",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots based on run data"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to the directory including the run information and run data",
    )
    parser.add_argument(
        "--agent",
        help="agent to plot the actions",
    )
    parser.add_argument(
        "--fidelity",
        default=30000,
        help="which fidelity the agent was trained on",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="specifies a seed to get the plots from. If None, it aggregates over all seeds",
    )
    parser.add_argument(
        "--optim_trace",
        help="Generate plots for optimization trace",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--action",
        help="Generate plots for teacher-agent action comparison",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--plot_type",
        help="Generate plots for teacher-agent comparison for a specified type",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--trajectory",
        help="Generate plots for optimization trajectory (function values)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--show",
        default=False,
        help="Either show or save the selected plots. Default is save.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--num_runs",
        help="Specifies how many individual runs should be plotted.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--aggregate",
        help="Defines whether action plot should aggregate all actions \
              for each batch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--teacher",
        help="Defines whether action plot should also include teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--action_teacher",
        help="Plot teachers actions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reward",
        help="Defines whether action plot should also include the reward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--single_plot",
        help="Defines whether teacher action plot should feature all ids in one plot",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--function",
        help="Which function to generate plots for. Used when single_plot is active.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--custom_paths",
        type=Path,
        help="Path to json file containing all base paths to plot for",
    )
    parser.add_argument(
        "--agent_labels",
        type=str,
        nargs="*",
        help="Data labels for the agents, have to be sorted according to the specified custom paths",
    )
    parser.add_argument(
        "--title", type=str, help="Title for the plot", default=""
    )
    parser.add_argument(
        "--teacher_dir",
        type=Path,
        help="Path to teacher/baseline. Used for comparison plot if the baseline differs from the teacher it has been trained on.",
        default="",
    )
    parser.add_argument(
        "--teacher_type",
        type=str,
        help="Type of teacher for comparison plot.",
        default="",
    )
    parser.add_argument(
        "--comparison",
        help="Plot f_cur comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--heterogeneous",
        help="Defines whether plots are for heterogeneous agents.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    agent_label = [map_agent_label[args.agent]]
    teacher_label = map_teacher_label[args.teacher_type]

    benchmark = args.data_dir.parents[1].name
    if benchmark == "SGD":
        if args.action:
            plot_actions_sgd(
                dir_path=args.data_dir,
                agent_type=args.agent,
                fidelity=args.fidelity,
                seed=args.seed,
                show=args.show,
                num_runs=args.num_runs,
                aggregate=args.aggregate,
                teacher=args.teacher,
                reward=args.reward,
                labels=[teacher_label, agent_label],
                title=args.title,
            )
        if args.comparison:
            plot_comparison(
                dir_paths=[args.data_dir],
                agent_type=args.agent,
                fidelity=args.fidelity,
                agent_labels=agent_label,
                teacher=args.teacher,
                show=args.show,
                title=args.title,
                teacher_path=args.teacher_dir,
                teacher_label=teacher_label,
                metric="valid_acc")
    elif benchmark == "ToySGD":
        if args.optim_trace:
            plot_optimization_trace(
                args.data_dir,
                args.agent_path,
                args.show,
                args.num_runs,
            )
        if args.action:
            plot_actions(
                args.data_dir,
                args.agent,
                args.fidelity,
                args.seed,
                args.show,
                args.num_runs,
                args.aggregate,
                args.teacher,
                args.reward,
                [teacher_label, agent_label[0]],
                args.title,
                args.heterogeneous,
            )
        if args.plot_type:
            plot_type(
                args.plot_type,
                args.data_dir,
                args.fidelity,
                args.seed,
                args.show,
                args.teacher,
            )
        if args.action_teacher:
            plot_teacher_actions(
                args.data_dir,
                args.show,
                args.reward,
                args.single_plot,
                args.function,
            )
        if args.comparison:
            if args.data_dir:
                plot_comparison(
                    dir_paths=[args.data_dir],
                    agent_type=args.agent,
                    fidelity=args.fidelity,
                    agent_labels=agent_label,
                    teacher=args.teacher,
                    show=args.show,
                    title=args.title,
                    teacher_path=args.teacher_dir,
                    teacher_label=teacher_label,
                )
    else:
        print(f"Unsupported type {benchmark}")

    