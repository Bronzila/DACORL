from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def test_agent(
    actor: Any,
    env: Any,
    n_runs: int,
    n_batches: int,
    seed: int,
) -> pd.DataFrame:
    env.seed(seed)
    actor.eval()

    actions = []
    rewards = []
    x_curs = []
    f_curs = []
    states = []
    runs = []
    batches = []

    for run_id in tqdm(range(n_runs)):
        state, _ = env.reset()
        actions.append(np.NaN)
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
