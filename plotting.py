import argparse
import json
from pathlib import Path

from src.utils.generate_plots import (
    plot_actions,
    plot_comparison,
    plot_optimization_trace,
    plot_teacher_actions,
    plot_type,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots based on run data"
    )
    parser.add_argument(
        "--data_dir",
        help="Path to the directory including the run information and run data",
    )
    parser.add_argument(
        "--agent",
        help="agent to plot the actions",
    )
    parser.add_argument(
        "--fidelity",
        default=15000,
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
        "--custom_paths", type=str, help="Path to json file containing all base paths to plot for",
    )
    parser.add_argument(
        "--agent_labels", type=str, nargs="*", help="Data labels for the agents, have to be sorted according to the specified custom paths",
    )
    parser.add_argument(
        "--comparison",
        help="Plot f_cur comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

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
            plot_comparison([args.data_dir],
                            args.agent_labels,
                            args.teacher,
                            args.show)
        elif args.custom_paths:
            with Path(args.custom_paths).open("r") as f:
                custom_paths = json.load(f)
            plot_comparison(custom_paths,
                            args.agent_labels,
                            args.teacher,
                            args.show)