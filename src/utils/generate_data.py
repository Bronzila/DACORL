import json
import os
import random
import signal

import numpy as np
import pandas as pd
import torch
from dacbench.benchmarks import ToySGD2DBenchmark

from src.agents import StepDecayAgent, ExponentialDecayAgent
from src.utils.replay_buffer import ReplayBuffer


# Time out related class and function
class OutOfTimeException(Exception):
    pass


def timeouthandler(signum, frame):
    raise OutOfTimeException

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_data(
    save_run_data,
    aggregated_run_data,
    save_rep_buffer,
    replay_buffer,
    results_dir,
    run_info,
    starting_points,
):
    run_info["starting_points"] = starting_points
    save_path = os.path.join(results_dir, f"rep_buffer")
    if save_rep_buffer:
        replay_buffer.save(save_path)
        with open(os.path.join(results_dir, f"run_info.json"), "w") as f:
            json.dump(run_info, f, indent=4)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if save_run_data:
        aggregated_run_data.to_csv(os.path.join(results_dir, "aggregated_run_data.csv"))

def get_environment(environment_type):
    if environment_type == "ToySGD":
        # setup benchmark
        bench = ToySGD2DBenchmark()
        return bench.get_environment()
    else:
        print(f"No environment of type {environment_type} found.")

def generate_dataset(agent_type, agent_config, environment_type, num_runs,
                     num_batches, seed, results_dir, save_run_data,
                     save_rep_buffer, timeout):

    if timeout > 0:
        # conversion from hours to seconds
        timeout = timeout * 60 * 60
        signal.signal(signal.SIGALRM, timeouthandler)
        signal.alarm(timeout)

    set_seeds(seed)

    if not (save_run_data or save_rep_buffer):
        input("You are not saving any results. Enter a key to continue anyway.")

    if results_dir == "":
        results_dir = os.path.join("data", agent_type, environment_type)
    else:
        results_dir = os.path.join(results_dir, agent_type, environment_type)

    env = get_environment(environment_type)
    state = env.reset()[0]
    state_dim = state.shape[0]
    buffer_size = num_runs * num_batches
    replay_buffer = ReplayBuffer(state_dim=state_dim,
                                 action_dim=1,
                                 buffer_size=buffer_size)

    agent = None
    if agent_type == "step_decay":
        agent = StepDecayAgent(**agent_config)
    elif agent_type == "exponential_decay":
        agent = ExponentialDecayAgent(**agent_config)
    else:
        print(f"No agent with type {agent_type} implemented.")

    aggregated_run_data = None
    run_info = {
        "agent_type": agent_type,
        "environment": environment_type,
        "seed": seed,
        "num_runs": num_runs,
        "num_batches": num_batches,
        "function": env.instance["function"],
        "lower_bound": env.lower_bound,
        "upper_bound": env.upper_bound,
    }

    try:
        for run in range(num_runs):
            if save_run_data:
                actions = []
                rewards = []
                f_curs = []
                x_curs = []
                states = []
                batch_indeces = []
                run_indeces = []
                starting_points = []
            state, meta_info = env.reset()
            starting_points.append(meta_info["start"])
            agent.reset()
            if save_run_data:
                actions.append(np.NaN)
                rewards.append(np.NaN)
                x_curs.append(env.x_cur.tolist())
                f_curs.append(env.objective_function(env.x_cur).numpy())
                states.append(state.numpy())
                batch_indeces.append(0)
                run_indeces.append(run)

            for batch in range(1, num_batches):
                print(f"Starting batch {batch}/{num_batches} of run {run}. \
                    Total {batch + run * num_batches}/{num_runs * num_batches}")

                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add_transition(state, action, next_state, reward, truncated)
                state = next_state
                if save_run_data:
                    actions.append(action)
                    rewards.append(reward.numpy())
                    x_curs.append(env.x_cur.tolist())
                    f_curs.append(env.objective_function(env.x_cur).numpy())
                    states.append(state.numpy())
                    batch_indeces.append(batch)
                    run_indeces.append(run)

            if save_run_data:
                run_data = pd.DataFrame({
                    "action": actions,
                    "reward": rewards,
                    "f_cur": f_curs,
                    "x_cur": x_curs,
                    "state": states,
                    "batch": batch_indeces,
                    "run": run_indeces,
                })
                if aggregated_run_data is None:
                    aggregated_run_data = run_data
                else:
                    aggregated_run_data = aggregated_run_data.append(
                        run_data, ignore_index=True
                    )
    except OutOfTimeException:
        save_data(
            save_run_data,
            aggregated_run_data,
            save_rep_buffer,
            replay_buffer,
            results_dir,
            run_info,
            starting_points,
        )
        print("Saved checkpoint, because run was about to end")

    save_data(
        save_run_data,
        aggregated_run_data,
        save_rep_buffer,
        replay_buffer,
        results_dir,
        run_info,
        starting_points,
    )

    if save_rep_buffer or save_run_data:
        msg = "Saved "
        msg += "rep_buffer " if save_rep_buffer else ""
        msg += "run_data " if save_run_data else ""
        print(f"{msg}to {results_dir}")