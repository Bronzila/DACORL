import argparse

from src.utils.run_statistics import calculate_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data from a CSV file.")
    parser.add_argument(
        "--path", type=str, help="Path to csv",
    )
    parser.add_argument(
        "--lowest",
        help="Get top n fbest (default: True)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="f_cur",
        help="Name of the column to analyze (default: 'f_cur')",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of fbest values to retrieve (default: 10)",
    )

    parser.add_argument(
        "--mean",
        help="Get mean and std deviation of final solutions (default: True)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--results",
        help="Find results subfolders",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose",
        help="Verbose output",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    min_mean, min_std, lowest_vals_of_min_mean, min_path = calculate_statistics(args.mean, args.lowest, args.n, args.path, args.results, args.verbose)

    print("Overall lowest statistics:")
    print(f"Found for path {min_path}")
    print("Mean +- Std {mean:.3e} ± {std:.3e}".format(mean=min_mean, std=min_std))
    print("Lowest values:")
    print(lowest_vals_of_min_mean)
