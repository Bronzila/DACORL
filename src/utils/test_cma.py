from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.utils.general import set_seeds


def run_batches(actor, env, n_batches, run_id):
    actions = []
    rewards = []
    f_curs = []
    states = []
    runs = []
    batches = []

    state = env.get_state()
    actions.append(env.es.parameters.sigma)
    rewards.append(np.NaN)
    lambdas.append(env.es.parameters.lambda_)
    f_curs.append(env.es.parameters.population.f)
    target_value.append(env.target)
    states.append(state.numpy())
    runs.append(run_id)
    batches.append(0)
    for batch_id in range(1, n_batches + 1):
        action = actor.act(state)
        next_state, reward, done, _, _ = env.step(action.item())
        state = next_state

        actions.append(action.item())
        rewards.append(reward.numpy())
        lambdas.append(env.es.parameters.lambda_)
        f_curs.append(env.es.parameters.population.f)
        target_value.append(env.target)
        states.append(state.numpy())
        runs.append(run_id)
        batches.append(batch_id)

        if done:
            break

        print(f"{action.item()}, {reward.numpy()}, {env.es.parameters.population.f}")
    return actions, rewards, lambdas, f_curs, target_value, states, runs, batches


def test_agent(
    actor: Any,
    env: Any,
    n_runs: int,
    n_batches: int,
    seed: int,
    starting_points: np.ndarray | None = None,
) -> pd.DataFrame:
    env.seed(seed)
    set_seeds(seed)
    actor.eval()

    actions = []
    rewards = []
    lambdas = []
    f_curs = []
    target_value = []
    states = []
    runs = []
    batches = []

    if starting_points is not None:
        for run_id, starting_point in enumerate(starting_points[:n_runs]):
            env.reset(
                seed=None,
                options={
                    "starting_point": torch.tensor(starting_point),
                },
            )
            r_a, r_r, r_l, r_f, r_t, r_s, r_runs, r_b = run_batches(
                actor,
                env,
                n_batches,
                run_id,
            )
            actions.extend(r_a)
            rewards.extend(r_r)
            lambdas.extend(r_l)
            f_curs.extend(r_f)
            target_value.extend(r_t)
            states.extend(r_s)
            runs.extend(r_runs)
            batches.extend(r_b)
    else:
        for run_id in range(n_runs):
            env.reset()
            r_a, r_r, r_l, r_f, r_t, r_s, r_runs, r_b = run_batches(
                actor,
                env,
                n_batches,
                run_id,
            )
            actions.extend(r_a)
            rewards.extend(r_r)
            lambdas.extend(r_l)
            f_curs.extend(r_f)
            target_value.extend(r_t)
            states.extend(r_s)
            runs.extend(r_runs)
            batches.extend(r_b)

    actor.train()
    return pd.DataFrame(
        {
            "action": actions,
            "reward": rewards,
            "lambdas": lambdas,
            "f_cur": f_curs,
            "target_value": target_value,
            "states": states,
            "run": runs,
            "batch": batches,
        },
    )
