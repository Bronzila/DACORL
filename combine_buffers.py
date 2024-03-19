import argparse
import json
from pathlib import Path

from src.utils.general import combine_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots based on run data"
    )
    parser.add_argument(
        "--root_dir",
        help="Path where different agents are laying",
    )
    parser.add_argument(
        "--function",
        help="function",
    )
    args = parser.parse_args()

    combined_buffer, combined_run_info, combined_run_data = combine_runs(args.root_dir, args.function)

    buffer_path = Path(args.root_dir, "combined", "rep_buffer")
    run_info_path = Path(args.root_dir, "combined", "run_info.json")
    run_data_path = Path(args.root_dir, "combined", "aggregated_run_data.csv")

    combined_buffer.save(buffer_path)
    with run_info_path.open(mode="w") as f:
            json.dump(combined_run_info, f, indent=4)
    combined_run_data.to_csv(run_data_path, index=False)