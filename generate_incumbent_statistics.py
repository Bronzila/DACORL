import json
import argparse
from pathlib import Path
import pandas as pd

from tap import Tap


def find_config_files(root_path: Path, id: str, budget: int) -> list[str]:
    config_files = list(
        root_path.glob(f"ToySGD/*/{id}/*/results/*/*/{budget}/config.json")
    )
    return config_files


def parse_config_file(file_path: str) -> dict[str, str]:
    with open(file_path, "r") as file:
        config = json.load(file)

    # Extract the parts of the path
    parts = file_path.parts
    teacher = parts[-8]
    function = parts[-6]
    agent = parts[-4]
    seed = parts[-3]

    # Add path components to the config dictionary
    config["teacher"] = teacher
    config["function"] = function
    config["agent"] = agent
    config["seed"] = seed

    return config


def create_dataframe(config_files: list[str]) -> pd.DataFrame:
    data = []
    for file_path in config_files:
        config = parse_config_file(file_path)
        data.append(config)

    df = pd.DataFrame(data)
    return df


def main(folder_path: str, id: str, budget: int) -> pd.DataFrame:
    root_path = Path(folder_path)
    if not root_path.exists():
        print(f"Error: Path {root_path} does not exist.")
        return

    config_files = find_config_files(root_path, id, budget)
    if not config_files:
        print("No config.json files found.")
        return

    df = create_dataframe(config_files)
    # Save the DataFrame to a CSV file
    output_csv = root_path / "experiment_configurations.csv"
    df.to_csv(output_csv, index=False)
    print(f"DataFrame saved to {output_csv}")



if __name__ == "__main__":
    class IncumbentStatisticsParser(Tap):
        folder_path: Path # Path to the root folder containing the experiments.
        id: str = "0" # ID which teacher instantiation was used.
        budget: int = 30000 # On which budget the agent was trained on.

    args = IncumbentStatisticsParser().parse_args()

    main(args.folder_path, args.id, args.budget)
