import json
import os
import signal

import pandas as pd
from dacbench.benchmarks import ToySGDBenchmark

from src.agents.step_decay import StepDecayAgent
from src.utils.replay_buffer import ReplayBuffer
from src.utils.general import OutOfTimeException, timeouthandler, set_seeds


def save_data(
    save_run_data,
    aggregated_run_data,
    save_rep_buffer,
    replay_buffer,
    results_dir,
    run_info,
):
    save_path = os.path.join(results_dir, f"rep_buffer")
    if save_rep_buffer:
        replay_buffer.save(save_path)
        with open(os.path.join(results_dir, f"run_info.json"), "w") as f:
            json.dump(run_info, f, indent=4)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if save_run_data:
        aggregated_run_data.to_csv(
            os.path.join(results_dir, "aggregated_run_data.csv")
        )


def get_environment(environment_type):
    if environment_type == "ToySGD":
        # setup benchmark
        bench = ToySGDBenchmark()
        return bench.get_environment()
    else:
        print(f"No environment of type {environment_type} found.")


def generate_dataset(
    agent_type,
    agent_config,
    environment_type,
    num_runs,
    num_batches,
    seed,
    results_dir,
    save_run_data,
    save_rep_buffer,
    timeout,
):
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
    replay_buffer = ReplayBuffer(
        state_dim=state_dim, action_dim=1, buffer_size=buffer_size
    )

    agent = None
    if agent_type == "step_decay":
        agent = StepDecayAgent(**agent_config)
    else:
        print(f"No agent with type {agent_type} implemented.")

    aggregated_run_data = None
    run_info = {
        "agent_type": agent_type,
        "environment": environment_type,
        "seed": seed,
        "num_runs": num_runs,
        "num_batches": num_batches,
    }

    try:
        for run in range(num_runs):
            if save_run_data:
                actions = []
                rewards = []
                f_currs = []
                states = []
                batch_indeces = []
                run_indeces = []
            # TODO properly reset to different starting point
            state = env.reset()[0]
            agent.reset()
            if save_run_data:
                actions.append(-1)
                rewards.append(-1337)
                f_currs.append(env.objective_function(env.x_cur))
                states.append(state)
                batch_indeces.append(0)
                run_indeces.append(run)

            for batch in range(1, num_batches):
                print(
                    f"Starting batch {batch}/{num_batches} of run {run}. \
                    Total {batch + run * num_batches}/{num_runs * num_batches}"
                )

                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                replay_buffer.add_transition(
                    state, action, next_state, reward, truncated
                )
                state = next_state
                if save_run_data:
                    actions.append(action)
                    rewards.append(reward)
                    f_currs.append(env.objective_function(env.x_cur))
                    states.append(state)
                    batch_indeces.append(batch)
                    run_indeces.append(run)

            if save_run_data:
                run_data = pd.DataFrame(
                    {
                        "action": actions,
                        "reward": rewards,
                        "f_curr": f_currs,
                        "state": states,
                        "batch": batch_indeces,
                        "run": run_indeces,
                    }
                )
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
        )
        print("Saved checkpoint, because run was about to end")

    save_data(
        save_run_data,
        aggregated_run_data,
        save_rep_buffer,
        replay_buffer,
        results_dir,
        run_info,
    )

    if save_rep_buffer or save_run_data:
        msg = "Saved "
        msg += "rep_buffer " if save_rep_buffer else ""
        msg += "run_data " if save_run_data else ""
        print(f"{msg}to {results_dir}")
