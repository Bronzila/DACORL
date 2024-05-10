import argparse

from src.utils.generate_plots import (
    plot_actions,
    plot_optimization_trace,
    plot_type
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
        default=10000,
        help="which fidelity the agent was trained on",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="specifies a seed to get the plots from.",
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
        help="Path to the directory including the run information and run data",
        type=int,
        default=1,
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
        default=True,
    )
    parser.add_argument(
        "--reward",
        help="Defines whether action plot should also include the reward",
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
