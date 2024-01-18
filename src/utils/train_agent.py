import os
import json

from src.utils.replay_buffer import ReplayBuffer


def load_run_data(data_dir: str):
    data_info_path = os.path.join(data_dir, "run_info.json")
    
    with open(data_info_path, "rb") as f:
        data_info = json.load(f)
    results_dir = os.path.join(data_dir, "results", "agent_name")

    replay_buffer_path = os.path.join(data_dir, "rep_buffer")
    replay_buffer = ReplayBuffer.load(replay_buffer_path)


