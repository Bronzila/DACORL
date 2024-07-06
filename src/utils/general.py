from __future__ import annotations

import json
import os
import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from CORL.algorithms.offline import td3_bc
from torch import nn

from src.agents import (
    ConstantAgent,
    ExponentialDecayAgent,
    SGDRAgent,
    StepDecayAgent,
)
from src.utils.replay_buffer import ReplayBuffer


def init_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))

def init_xavier_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain("relu"))

def init_kaiming_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

def init_kaiming_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

def init_orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, nn.init.calculate_gain("relu"))

init_map = {
    "xavier_uniform": init_xavier_uniform,
    "xavier_normal": init_xavier_normal,
    "kaiming_uniform": init_kaiming_uniform,
    "kaiming_normal": init_kaiming_normal,
    "orthogonal": init_orthogonal,
}

# Time out related class and function
class OutOfTimeError(Exception):
    pass


def timeouthandler(signum: Any, frame: Any) -> None:
    raise OutOfTimeError


def set_timeout(timeout: int) -> None:
    if timeout > 0:
        # conversion from hours to seconds
        timeout = timeout * 60 * 60
        signal.signal(signal.SIGALRM, timeouthandler)
        signal.alarm(timeout)


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_agent(
    agent_type: str,
    agent_config: dict[str, Any],
    hyperparameters: dict[str, Any] | None = None,
    device: str = "cpu",
) -> Any:
    if agent_type == "step_decay":
        return StepDecayAgent(**agent_config["params"])
    if agent_type == "exponential_decay":
        return ExponentialDecayAgent(**agent_config["params"])
    if agent_type == "sgdr":
        return SGDRAgent(**agent_config["params"])
    if agent_type == "constant":
        return ConstantAgent(**agent_config["params"])
    if agent_type == "td3_bc":
        if hyperparameters is None:
            print("No hyperparameters specified, resorting to defaults.")
            hyperparameters = {}
        else:
            hyperparameters = dict(hyperparameters)

        config = td3_bc.TrainConfig

        state_dim = agent_config["state_dim"]
        action_dim = agent_config["action_dim"]
        max_action = agent_config["max_action"]
        min_action = agent_config["min_action"]

        alpha = hyperparameters.get("alpha", config.alpha)

        initialization = hyperparameters.get("initialization", None)
        if initialization:
            initialization = init_map[hyperparameters["initialization"]]
        actor = td3_bc.Actor(
            state_dim,
            action_dim,
            max_action,
            get_activation(hyperparameters.get("activation", "ReLU")),
            hidden_dim=hyperparameters.get("hidden_dim", 64),
            hidden_layers=hyperparameters.get("hidden_layers", 1),
            dropout_rate=hyperparameters.get("dropout_rate", 0.2),
            initialization=initialization,
        ).to(device)
        print(f"hidden_dim: {hyperparameters.get('hidden_dim', 64)}")
        actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=hyperparameters.get("lr_actor", 3e-4),
        )

        critic_1 = td3_bc.Critic(
            state_dim,
            action_dim,
        ).to(device)
        critic_1_optimizer = torch.optim.Adam(
            critic_1.parameters(),
            lr=hyperparameters.get("lr_critic", 3e-4),
        )

        critic_2 = td3_bc.Critic(
            state_dim,
            action_dim,
        ).to(device)
        critic_2_optimizer = torch.optim.Adam(
            critic_2.parameters(),
            lr=hyperparameters.get("lr_critic", 3e-4),
        )

        kwargs = {
            "max_action": max_action,
            "min_action": min_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic_1": critic_1,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2": critic_2,
            "critic_2_optimizer": critic_2_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "device": device,
            # TD3
            "policy_noise": config.policy_noise,
            "noise_clip": config.noise_clip,
            "policy_freq": config.policy_freq,
            # TD3 + BC
            "alpha": alpha,
        }
        return td3_bc.TD3_BC(**kwargs)

    raise NotImplementedError(
        f"No agent with type {agent_type} implemented.",
    )


def get_environment(env_config: dict) -> Any:
    from dacbench.benchmarks import SGDBenchmark, ToySGD2DBenchmark
    if env_config["type"] == "ToySGD":
        # setup benchmark
        bench = ToySGD2DBenchmark()
        bench.config.cutoff = env_config["num_batches"]
        bench.config.low = env_config["low"]
        bench.config.high = env_config["high"]
        bench.config.function = env_config["function"]
        bench.config.initial_learning_rate = env_config["initial_learning_rate"]
        bench.config.state_version = env_config["state_version"]
        bench.config.reward_version = env_config["reward_version"]
        bench.config.boundary_termination = env_config["boundary_termination"]
        bench.config.seed = env_config["seed"]
        return bench.get_environment()
    elif env_config["type"] == "SGD":
        bench = SGDBenchmark(config=env_config)
        return bench.get_benchmark()
    else:
        raise NotImplementedError(
            f"No environment of type {env_config['type']} found.",
        )


def get_activation(activation: str) -> nn.Module:
    if activation == "ReLU":
        return nn.ReLU
    if activation == "LeakyReLU":
        return nn.LeakyReLU
    if activation == "Tanh":
        return nn.Tanh
    return None


def save_agent(state_dicts: dict, results_dir: Path, iteration: int, seed: int=0) -> None:
    filename = results_dir / str(seed) / f"{iteration + 1}"
    if not filename.exists():
        filename.mkdir(parents=True)

    for key, s in state_dicts.items():
        torch.save(s, filename / f"agent_{key}")


def load_agent(agent_type: str, agent_config: dict, agent_path: Path) -> Any:
    hp_conf_path = agent_path / "config.json"
    with hp_conf_path.open("r") as f:
        hyperparameters = json.load(f)
    print(hyperparameters)
    agent = get_agent(agent_type, agent_config, hyperparameters)
    state_dict = agent.state_dict()
    new_state_dict = {}
    for key, _ in state_dict.items():
        s = torch.load(agent_path / f"agent_{key}")
        new_state_dict.update({key: s})

    agent.load_state_dict(new_state_dict)
    return agent

def get_homogeneous_agent_paths(root_dir: str, function: str):
    root_path = Path(root_dir)
    # Sort directories to ensure same sequence for reproducibility
    agent_dirs = sorted([entry.name for entry in root_path.iterdir() if entry.is_dir() and entry.name != "combined"])
    paths = []
    for dirname in agent_dirs:
        agent_path = Path(root_dir, dirname, function)
        paths.append(agent_path)
    return paths

def concat_runs(agent_paths):
    combined_buffer = None
    combined_run_info = None
    combined_run_data = []
    for idx, root_path in enumerate(agent_paths):
        replay_path = Path(root_path, "rep_buffer")
        run_info_path = Path(root_path, "run_info.json")
        run_data_path = Path(root_path, "aggregated_run_data.csv")

        with run_info_path.open(mode="rb") as f:
            run_info = json.load(f)
        temp_buffer = ReplayBuffer.load(replay_path)

        if combined_buffer == None and combined_run_info == None:
            combined_buffer = temp_buffer
            combined_run_info = {"environment": run_info["environment"],
                                 "starting_points": run_info["starting_points"],
                                 "seed": run_info["seed"],
                                 "num_runs": run_info["num_runs"],
                                 "num_batches": run_info["num_batches"],
                                 "agent": {"type": run_info["agent"]["type"]}}
        else:
            combined_buffer.merge(temp_buffer)
            # combined_run_info["starting_points"].extend(run_info["starting_points"])

        df = pd.read_csv(run_data_path)
        df["run"] += idx * run_info["num_runs"]
        combined_run_data.append(df)
    return combined_buffer, combined_run_info, pd.concat(combined_run_data, ignore_index=True)

def get_run_ids_by_agent_path(path_data_mapping, combination_strategy, total_size):
    if combination_strategy == "perf_sampling":
        performances = []
        paths = []
        n_runs = 0
        for path, data in path_data_mapping.items():
            final_run_values = data["run_data"].groupby("run").last()["f_cur"]
            mean_final_score = np.mean(final_run_values)
            path_data_mapping[path]["performance"] = mean_final_score
            performances.append(mean_final_score)
            paths.append(path)
            n_runs = data["run_info"]["num_runs"]
        # Get max score, invert scores and normalize them
        performances = np.array(performances)
        max_score = max(performances)
        inverted_scores = max_score - performances
        normalized_scores = inverted_scores / np.sum(inverted_scores)

        # Power factor > 1 emphasizes well-performing heuristics over worse performing
        power_factor = 1
        emphasized_scores = normalized_scores ** power_factor

        # Introduce baseline probability to also sample from worst policy
        # NOTE: This does not mean, that the worst teacher has a final prob. of being sampled of  1 / len_agents
        # It just adds a small baseline probability for samples
        baseline_prob = 1 / len(emphasized_scores)
        emph_base_probs = (1 - baseline_prob) * emphasized_scores + baseline_prob

        final_normalized_scores = emph_base_probs / np.sum(emph_base_probs)

        # Get sampling weights from scores
        sampling_weights = final_normalized_scores.tolist()

        # Always use same rng
        rng = np.random.default_rng(seed=0)
        zipped_paths_and_weights = zip(paths, sampling_weights)
        # Sort by performance in order to pass overflow samples
        sorted_paths_and_weights = sorted(zipped_paths_and_weights, key=lambda x: x[1], reverse=True)
        # only needed if runs overflow
        runs_to_distribute = 0
        remaining_teachers = len(sorted_paths_and_weights)
        for i, (path, weight) in enumerate(sorted_paths_and_weights):
            num_samples = int(round(total_size * weight + (1 / remaining_teachers) * runs_to_distribute))

            # If number of samples which should be taken from this teacher is higher than runs available
            # Add all runs to final buffer and distribute overflow runs to other teachers
            if num_samples > n_runs:
                run_ids = np.arange(n_runs)
                runs_to_distribute = num_samples - n_runs
                remaining_teachers = len(sorted_paths_and_weights) - i - 1 # - 1 since we need to subtract current teacher as well
            else:
                # Sample run_ids without replacement
                run_ids = rng.choice(np.arange(n_runs), size=num_samples, replace=False)
            
            path_data_mapping[path]["run_ids"] = run_ids
            # assert that there are no duplicated transitions
            assert len(run_ids) == len(set(run_ids))
        return path_data_mapping
    elif combination_strategy == "perf_per_run":
        # Get n_runs and initialize run_ids lists
        for path, data in path_data_mapping.items():
            path_data_mapping[path]["run_ids"] = []
            n_runs = data["run_info"]["num_runs"]
        for run_id in range(n_runs):
            f_values = []
            paths = []
            for path, data in path_data_mapping.items():
                final_run_values = data["run_data"].groupby("run").last()["f_cur"]
                paths.append(path)
                f_values.append(final_run_values[run_id])
            path_id = np.argmin(np.array(f_values))
            path = paths[path_id]
            path_data_mapping[path]["run_ids"].append(run_id)

        for path, data in path_data_mapping.items():
            print(path)
            print(len(data["run_ids"]) / n_runs)
        return path_data_mapping
    else:
        raise NotImplementedError()

def filter_buffer(data):
    device = torch.device("cpu")
    # Turn states etc. into pandas dataframe fo easier access
    states_np = data["buffer"]._states.cpu().numpy()
    actions_np = data["buffer"]._actions.cpu().numpy()
    rewards_np = data["buffer"]._rewards.cpu().numpy()
    next_states_np = data["buffer"]._next_states.cpu().numpy()
    dones_np = data["buffer"]._dones.cpu().numpy()

    transitions_df = pd.DataFrame({
        "state": list(states_np),
        "action": list(actions_np),
        "reward": rewards_np.flatten(),
        "next_state": list(next_states_np),
        "done": dones_np.flatten(),
    })

    # Filter out rows where states are all zeros --> empty rows in replay_buffer
    transitions_df = transitions_df[~(transitions_df["state"].apply(lambda x: np.all(np.array(x) == 0)))]

    # Assign run IDs based on the first state value being 1
    run_id = -1
    run_ids = []
    for state in transitions_df["state"]:
        if state[0] == 1:
            run_id += 1
        run_ids.append(run_id)

    transitions_df["run_id"] = run_ids

    # Filter the DataFrame to only keep specific run IDs
    filtered_df = transitions_df[transitions_df["run_id"].isin(data["run_ids"])]

    # Convert the filtered DataFrame back to numpy arrays
    filtered_states = np.array(filtered_df["state"].tolist())
    filtered_actions = np.array(filtered_df["action"].tolist())
    filtered_rewards = filtered_df["reward"].to_numpy().reshape(-1, 1)
    filtered_next_states = np.array(filtered_df["next_state"].tolist())
    filtered_dones = filtered_df["done"].to_numpy().reshape(-1, 1)

    data["buffer"]._size = len(filtered_df)
    data["buffer"]._pointer = len(filtered_df)

    data["buffer"]._states = torch.tensor(filtered_states, dtype=torch.float32, device=device)
    data["buffer"]._actions = torch.tensor(filtered_actions, dtype=torch.float32, device=device)
    data["buffer"]._rewards = torch.tensor(filtered_rewards, dtype=torch.float32, device=device)
    data["buffer"]._next_states = torch.tensor(filtered_next_states, dtype=torch.float32, device=device)
    data["buffer"]._dones = torch.tensor(filtered_dones, dtype=torch.float32, device=device)

def create_buffer_from_ids(path_data_mapping):
    # For each buffer
    # Group states etc. by run (use first state, optim budget as criterion when new run begins)
    # Remove all runs that are not in run_ids
    # Merge remaining buffer into combined one
    # Same for run data but there its a lot easier
    combined_buffer = None
    combined_run_data = None
    run_info = None
    for _path, data in path_data_mapping.items():
        # Filter buffer for run_ids
        filter_buffer(data)

        # Filter run data
        data["run_data"] = data["run_data"][data["run_data"]["run"].isin(data["run_ids"])]

        if combined_buffer is None:
            combined_buffer = data["buffer"]
            combined_run_data = data["run_data"]
            run_info = data["run_info"]
        else:
            combined_buffer.merge(data["buffer"])
            data["run_data"]["run"] += data["run_info"]["num_runs"]
            combined_run_data = pd.concat([combined_run_data, data["run_data"]], ignore_index=True)

    return combined_buffer, run_info, combined_run_data

def combine_runs(agent_paths, combination_strategy="concat", total_size=3000):
    if combination_strategy == "concat":
        return concat_runs(agent_paths)
    else:
        path_data_mapping = {}
        for root_path in agent_paths:
            replay_path = Path(root_path, "rep_buffer")
            run_info_path = Path(root_path, "run_info.json")
            run_data_path = Path(root_path, "aggregated_run_data.csv")

            with run_info_path.open(mode="rb") as f:
                run_info = json.load(f)

            buffer = ReplayBuffer.load(replay_path)

            df_run_data = pd.read_csv(run_data_path)
            path_data_mapping[root_path] = {
                "buffer": buffer,
                "run_data": df_run_data,
                "run_info": run_info,
            }
        path_data_mapping = get_run_ids_by_agent_path(path_data_mapping, combination_strategy, total_size)

        return create_buffer_from_ids(path_data_mapping)

def combine_run_data(data_paths, num_runs=100):
    combined_run_data = []
    for idx, root_path in enumerate(data_paths):
        run_data_path = Path(root_path)
        df = pd.read_csv(run_data_path)
        df["run"] += idx * num_runs
        combined_run_data.append(df)
    return pd.concat(combined_run_data, ignore_index=True)

def find_lowest_values(df, column_name, n=10):
    final_evaluations = df.groupby("run").last()

    # Sort the DataFrame by the specified column in ascending order
    sorted_df = final_evaluations.sort_values(by=column_name)

    # Get the lowest n values from the sorted DataFrame
    return sorted_df.head(n)

def calc_mean_and_std_dev(df):
    final_evaluations = df.groupby("run").last()

    fbests = final_evaluations["f_cur"]
    return fbests.mean(), fbests.std()

def calculate_single_seed_statistics(calc_mean=True, calc_lowest=True, n_lowest=1, path=None,
                                    results=True, verbose=False, interpolation=False):
    paths = []
    filename = "eval_data_interpolation.csv" if interpolation else "eval_data.csv"
    if results:
        for folder_path, _, _ in os.walk(path):
            paths.extend(Path(folder_path).glob(f"*/{filename}"))
    else:
        paths.append(path)
    # Load data
    min_mean = np.inf
    min_std = np.inf
    min_iqm = np.inf
    min_iqm_std = np.inf
    min_path = ""
    lowest_vals_of_min_mean = []
    for path in paths:
        incumbent_changed = False
        df = pd.read_csv(path)
        if verbose:
            print(f"Calculating for path {path}")
        if calc_mean:
            mean, std = calc_mean_and_std_dev(df)
            mean = float(f"{mean:.3e}")
            std = float(f"{std:.3e}")
            if mean < min_mean or mean == min_mean and std < min_std:
                incumbent_changed = True
                min_mean = mean
                min_std = std
                min_path = path
                min_iqm, min_iqm_std = compute_IQM(df)
            if verbose:
                print(f"Mean +- Std {mean:.3e} ± {std:.3e}")
        if calc_lowest:
            lowest_vals = find_lowest_values(df, "f_cur", n_lowest)
            if incumbent_changed:
                lowest_vals_of_min_mean = lowest_vals["f_cur"]
            if verbose:
                print("Lowest values:")
                print(lowest_vals["f_cur"])
    return min_mean, min_std, lowest_vals_of_min_mean, min_iqm, min_iqm_std, min_path

def calculate_multi_seed_statistics(calc_mean=True, calc_lowest=True, n_lowest=1, path=None,
                                    results=True, verbose=False, num_runs=100, interpolation=False):
    seed_dirs = set()
    filename = "eval_data_interpolation.csv" if interpolation else "eval_data.csv"
    if results:
        for eval_file in Path(path).rglob(filename):
            print(eval_file)
            # Extract the seed directory
            seed_dir = eval_file.parents[1]
            seed_dirs.add(seed_dir)
    else:
        seed_dirs.add(path)

    best_iterations_paths = []
    # This is only used if there are multiple checkpoints in the seed directory --> choose the best one
    for seed_dir in seed_dirs:
        min_mean, min_std, _, _, _, min_path = calculate_single_seed_statistics(calc_mean, calc_lowest,
                                                                                n_lowest, seed_dir, results, verbose,
                                                                                interpolation)
        if verbose:
            print(f"Minimum mean {min_mean} +- {min_std} for path {min_path}")
        best_iterations_paths.append(min_path)
    combined_data = combine_run_data(best_iterations_paths, num_runs=num_runs)
    if calc_mean:
            mean, std = calc_mean_and_std_dev(combined_data)
            mean = float(f"{mean:.3e}")
            std = float(f"{std:.3e}")
            iqm, iqm_std = compute_IQM(combined_data)
    if calc_lowest:
        lowest_vals = find_lowest_values(combined_data, "f_cur", n_lowest)["f_cur"]
    return mean, std, lowest_vals, iqm, iqm_std, 0 # path doesnt really matter here

def calculate_statistics(calc_mean=True, calc_lowest=True, n_lowest=1, path=None,
                         results=True, verbose=False, multi_seed=False, num_runs=100,
                         interpolation=False):
    if multi_seed:
        return calculate_multi_seed_statistics(calc_mean, calc_lowest,
                                               n_lowest, path, results, verbose, num_runs,
                                               interpolation)
    else:
        return calculate_single_seed_statistics(calc_mean, calc_lowest, n_lowest,
                                                path, results, verbose, interpolation)

def compute_IQM(df):
    final_evaluations = df.groupby("run").last()
    df_sorted = final_evaluations.sort_values(by="f_cur")

    # Calculate the number of rows representing 25% of the DataFrame
    num_rows = len(df_sorted)
    num_to_remove = int(0.25 * num_rows)

    # Remove the upper and lower 25% of the DataFrame
    df_trimmed = df_sorted[num_to_remove:-num_to_remove]
    fbests = df_trimmed["f_cur"]
    return fbests.mean(), fbests.std()

def compute_prob_outperformance(df_teacher, df_agent):
    final_evaluations_teacher = df_teacher.groupby("run").last()["f_cur"]
    final_evaluations_agent = df_agent.groupby("run").last()["f_cur"]

    assert len(final_evaluations_agent) == len(final_evaluations_teacher)

    p = 0
    for i in range(len(final_evaluations_agent)):
        if final_evaluations_agent[i] < final_evaluations_teacher[i]:
            p += 1

    return p / len(final_evaluations_agent)