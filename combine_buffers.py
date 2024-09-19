import argparse
import json
from pathlib import Path

from src.utils.run_statistics import combine_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine generated data"
    )
    parser.add_argument(
        "--root_dir",
        help="Path where different agents are laying",
    )
    parser.add_argument(
        "--custom_paths",
        help="Path to a json file containing all paths to combine",
    )
    parser.add_argument(
        "--function",
        help="function",
    )
    parser.add_argument(
        "--teacher",
        help="teacher",
    )
    parser.add_argument(
         "--combined_dir",
         default="data_combined/",
    )
    args = parser.parse_args()

    if args.root_dir:
        paths = get_homogeneous_agent_paths(args.root_dir, args.teacher, args.function)
    elif args.custom_paths:
        with Path(args.custom_paths).open("r") as f:
            paths = json.load(f)
    output_path = Path(args.root_dir, "ToySGD", args.teacher, "combined", args.function)
    output_path.mkdir(parents=True, exist_ok=True)
    combined_buffer, combined_run_info, combined_run_data = combine_runs(paths)
    buffer_path = Path(output_path, "rep_buffer")
    run_info_path = Path(output_path, "run_info.json")
    run_data_path = Path(output_path, "aggregated_run_data.csv")
    combined_buffer.save(buffer_path)
    with run_info_path.open(mode="w") as f:
            json.dump(combined_run_info, f, indent=4)
    combined_run_data.to_csv(run_data_path, index=False)