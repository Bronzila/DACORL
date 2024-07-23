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
    states = []
    runs = []
    batches = []
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    test_loss = []
    test_acc = []

    state = env.get_state()
    actions.append(math.log10(env.learning_rate))
    rewards.append(np.NaN)
    train_loss.append(env.train_loss)
    valid_loss.append(env.validation_loss)
    train_acc.append(env.train_accuracy)
    valid_acc.append(env.validation_accuracy)
    test_loss.append(env.test_loss)
    test_acc.append(env.test_accuracy)
    runs.append(run_id)
    batches.append(0)
    for batch_id in range(1, n_batches + 1):
        action = actor.act(state)
        next_state, reward, done, _, _ = env.step(action.item())
        state = next_state

        actions.append(action.item())
        rewards.append(reward.numpy())
        states.append(state.numpy())
        runs.append(run_id)
        batches.append(batch_id)
        train_loss.append(env.train_loss)
        valid_loss.append(
            env.validation_loss,
        )
        train_acc.append(env.train_accuracy)
        valid_acc.append(env.validation_accuracy)
        test_loss.append(env.test_loss)
        test_acc.append(env.test_accuracy)

        if done:
            break

    return {
        "actions": actions,
        "rewards": rewards,
        "states": states,
        "runs": runs,
        "batches": batches,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


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

    logs = ({},)

    if starting_points is not None:
        for run_id, starting_point in enumerate(starting_points[:n_runs]):
            env.reset(
                seed=None,
                options={
                    "starting_point": torch.tensor(starting_point),
                },
            )
            run_logs = run_batches(
                actor,
                env,
                n_batches,
                run_id,
            )
            for k, v in run_logs.items():
                if k not in logs:
                    logs[k] = []

                logs[k].extend(v)

    else:
        for run_id in range(n_runs):
            env.reset()
            run_logs = run_batches(
                actor,
                env,
                n_batches,
                run_id,
            )
            for k, v in run_logs.items():
                if k not in logs:
                    logs[k] = []

                logs[k].extend(v)

    actor.train()
    return pd.DataFrame(logs)
