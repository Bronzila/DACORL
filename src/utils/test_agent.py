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
    for batch_id in range(1, n_batches + 1):
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
        for run_id, starting_point in enumerate(starting_points[:n_runs]):
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
        for run_id in range(n_runs):
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

def test_agent_SGD(
    actor: Any,
    env: Any,
    n_runs: int,
    n_batches: int,
    seed: int,
) -> pd.DataFrame:
    env.seed(seed)
    set_seeds(seed)
    actor.eval()

    actions = []
    rewards = []
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    test_loss = []
    test_acc = []
    states = []
    runs = []
    batches = []

    for run in range(n_runs):
        state, meta_info = env.reset()

        # Save start of optimization
        rewards.append(np.NaN)
        states.append(state.numpy())
        batches.append(0)
        runs.append(run)
        actions.append(math.log10(env.learning_rate))
        train_loss.append(env.train_loss)
        valid_loss.append(env.validation_loss)
        train_acc.append(env.train_accuracy)
        valid_acc.append(env.validation_accuracy)
        test_loss.append(env.test_loss)
        test_acc.append(env.test_accuracy)

        for batch in range(1, n_batches + 1): # As we start with batch 1 and not 0, add 1

            action = actor.act(state)
            next_state, reward, done, truncated, info = env.step(action)

            actions.append(action)
            rewards.append(reward.numpy())
            states.append(state.numpy())
            batches.append(batch)
            runs.append(run)
            train_loss.append(env.train_loss)
            valid_loss.append(env.validation_loss)
            train_acc.append(env.train_accuracy)
            valid_acc.append(env.validation_accuracy)
            test_loss.append(env.test_loss)
            test_acc.append(env.test_accuracy)

            state = next_state
            if done:
                break


    actor.train()
    return pd.DataFrame(
        {
            "action": actions,
            "reward": rewards,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "states": states,
            "run": runs,
            "batch": batches,
        },
    )