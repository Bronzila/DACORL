from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def test_agent(
    actor: Any,
    env: Any,
    n_episodes: int,
    seed: int,
) -> pd.DataFrame:
    env.seed(seed)
    state, _ = env.reset()
    actor.eval()

    actions = []
    rewards = []
    x_curs = []
    f_curs = []
    states = []

    actions.append(np.NaN)
    rewards.append(np.NaN)
    x_curs.append(env.x_cur.tolist())
    f_curs.append(env.objective_function(env.x_cur).numpy())
    states.append(state.numpy())

    for _ in tqdm(range(n_episodes)):
        action = actor.act(state)
        next_state, reward, done, _, _ = env.step(action.item())
        state = next_state

        actions.append(action)
        rewards.append(reward.numpy())
        x_curs.append(env.x_cur.tolist())
        f_curs.append(env.objective_function(env.x_cur).numpy())
        states.append(state.numpy())

    actor.train()
    return pd.DataFrame(
        {
            "action": actions,
            "reward": rewards,
            "x_cur": x_curs,
            "f_cur": f_curs,
            "states": states,
        },
    )
