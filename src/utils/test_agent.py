from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.general import set_seeds


def run_batches(actor, env, n_batches, run_id):
    actions = []
    rewards = []
    x_curs = []
    f_curs = []
    states = []
    runs = []
    batches = []

    state = env.get_state()
    actions.append(math.log10(env.learning_rate))
    rewards.append(np.NaN)
    x_curs.append(env.x_cur.tolist())
    f_curs.append(env.objective_function(env.x_cur).numpy())
    states.append(state.numpy())
    runs.append(run_id)
    batches.append(0)
    for batch_id in tqdm(range(1, n_batches)):
        action = actor.act(state)
        next_state, reward, done, _, _ = env.step(action.item())
        state = next_state

        actions.append(action.item())
        rewards.append(reward.numpy())
        x_curs.append(env.x_cur.tolist())
        f_curs.append(env.objective_function(env.x_cur).numpy())
        states.append(state.numpy())
        runs.append(run_id)
        batches.append(batch_id)

        if done:
            break

    return actions, rewards, x_curs, f_curs, states, runs, batches

def test_agent(
    actor: Any,
    env: Any,
    n_runs: int,
    n_batches: int,
    seed: int,
    starting_points: np.ndarray=None,
) -> pd.DataFrame:
    env.seed(seed)
    set_seeds(seed)
    actor.eval()

    actions = []
    rewards = []
    x_curs = []
    f_curs = []
    states = []
    runs = []
    batches = []

    if starting_points is not None:
        for run_id, starting_point in tqdm(enumerate(starting_points[:n_runs])):
            env.reset(seed=None, options={
                "starting_point": torch.tensor(starting_point),
                },
            )
            r_a, r_r, r_x, r_f, r_s, r_runs, r_b = run_batches(actor, env,
                                                               n_batches, run_id)
            actions.extend(r_a)
            rewards.extend(r_r)
            x_curs.extend(r_x)
            f_curs.extend(r_f)
            states.extend(r_s)
            runs.extend(r_runs)
            batches.extend(r_b)
    else:
        for run_id in tqdm(range(n_runs)):
            env.reset()
            r_a, r_r, r_x, r_f, r_s, r_runs, r_b = run_batches(actor, env,
                                                            n_batches, run_id)
            actions.extend(r_a)
            rewards.extend(r_r)
            x_curs.extend(r_x)
            f_curs.extend(r_f)
            states.extend(r_s)
            runs.extend(r_runs)
            batches.extend(r_b)

    actor.train()
    return pd.DataFrame(
        {
            "action": actions,
            "reward": rewards,
            "x_cur": x_curs,
            "f_cur": f_curs,
            "states": states,
            "run": runs,
            "batch": batches,
        },
    )
