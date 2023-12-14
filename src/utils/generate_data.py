import os
import torch
import numpy as np
import random
from utils.replay_buffer import ReplayBuffer
from agents.step_decay import StepDecayAgent
import signal

# Time out related class and function
class OutOfTimeException(Exception):
    pass


def timeouthandler(signum, frame):
    raise OutOfTimeException

def get_environment(environment_type):
    if environment_type == "ToySGD":
        # setup benchmark
    else:
        print(f"No environment of type {environment_type} found.")

def generate_dataset(agent_type, agent_config, environment_type, function_name,
                     num_runs, num_batches, seed, results_dir, reward_structure,
                     state_version, save_run_data, save_rep_buffer, timeout):

    if timeout > 0:
        # conversion from hours to seconds
        timeout = timeout * 60 * 60
        signal.signal(signal.SIGALRM, timeouthandler)
        signal.alarm(timeout)

    if not (save_run_data or save_rep_buffer):
        input("You are not saving any results. Enter a key to continue anyway.")

    if results_dir == "":
        results_dir = os.path.join("data", agent_type, function_name)
    else:
        results_dir = os.path.join(results_dir, agent_type, function_name)

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
    else:
        print(f"No agent with type {agent_type} implemented.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    population_shape = (pop_size, dimensions)
    init_pops = [
        (env.upper_bound - env.lower_bound) * random_number_generator.rand(*population_shape) + env.lower_bound
        for _ in range(num_starting_points)
    ]

    aggregated_run_data = None

    for i, init_pop in enumerate(init_pops):

        if save_run_data:
            run_data = pd.DataFrame({
                'gen': [],
                'action': [],
                'reward': [],
                'fbest': [],
                'pop': [],
                'states': [],
                'es_run': []
            })
        for es_run in range(num_es_runs):
            print(f"Starting {es_run + i * num_es_runs}/{len(init_pops)*num_es_runs}")
            state = env.reset(init_pop)
            agent.reset()
            if save_run_data:
                run_data = run_data.append(
                    {
                        'gen': 0,
                        'fbest': env.fbest,
                        'pop': env.pop.tolist(),
                        'states': state.tolist(),
                        'es_run': es_run + i * num_es_runs
                    },
                    ignore_index=True)
            for gen in range(1, num_gens):
                if gen == 1:
                    action = [0.5]
                    if agent_type == 'random_agent':
                        agent = RandomAgent(action_dim=1, seed=es_run+i*num_es_runs)
                    elif "constant_agent" in agent_type:
                        action = agent.select_action()
                    elif agent_type == "step_agent":
                        action = agent.select_action()
                else:
                    if agent_type == 'simple_cauchy_agent':
                        action = (agent.calc_new_f_mu_simple_mean(
                            env.mut, env.successes), )
                    elif agent_type == 'lehmer_mean_agent':
                        action = (agent.calc_new_f_mu(env.mut, env.successes),)
                    elif agent_type == 'mde_cauchy_agent':
                        action = (agent.calc_new_f_mu(env.mut, env.successes),)
                    else:
                        action = agent.select_action(state)
                next_state, reward, done, cauchy_dict = env.step(action)
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                if save_run_data:
                    run_data = run_data.append(
                        {
                            'gen': gen,
                            'action': action[0],
                            'reward': reward,
                            'fbest': env.fbest,
                            'pop': env.pop.tolist(),
                            'states': state.tolist(),
                            'es_run': es_run + i * num_es_runs
                        },
                        ignore_index=True)
    
        if save_run_data:
            if aggregated_run_data is None:
                aggregated_run_data = run_data
            else:
                aggregated_run_data = aggregated_run_data.append(
                run_data, ignore_index=True)
    now = datetime.now().strftime("%d-%m-%y_%H-%M")
    # save_path = os.path.join(results_dir, f"rep_buffer_{now}")
    save_path = os.path.join(results_dir, f"rep_buffer")
    if save_rep_buffer:
        replay_buffer.save(save_path)
    run_info = {
        'agent_type': agent_type,
        'function_name': function_name,
        'seed': seed,
        'reward_structure': reward_structure,
        'state_version': state_version,
        'dimension': dimensions,
        'num_gens': num_gens,
        'num_es_runs': num_es_runs,
        'pop_size': pop_size,
        'init_pops': [pop.tolist() for pop in init_pops]
    }
    if save_rep_buffer:
        with open(os.path.join(results_dir, f"run_info.json"), 'w') as f:
            json.dump(run_info, f, indent=4)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if save_run_data:
        aggregated_run_data.to_csv(os.path.join(results_dir, "aggregated_run_data.csv"))

    if save_rep_buffer or save_run_data:
        msg = "Saved "
        msg += "rep_buffer " if save_rep_buffer else ""
        msg += "run_data " if save_run_data else ""
        print(f"{msg}to {results_dir}")