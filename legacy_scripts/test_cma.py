from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from src.utils.general import set_seeds

if TYPE_CHECKING:
    from DACBench.dacbench import AbstractMADACEnv
    from torch.nn import Module


def run_batches(
    actor: Module,
    env: AbstractMADACEnv,
    n_batches: int,
    run_id: int,
) -> dict:
    actions = []
    rewards = []
    f_curs = []
    population = []
    lambdas = []
    target_value = []
    states = []
    function_id = []
    runs = []
    batches = []

    state = env.get_state()
    actions.append(env.es.parameters.sigma)
    rewards.append(np.NaN)
    lambdas.append(env.es.parameters.lambda_)
    f_curs.append(env.es.parameters.fopt)
    population.append(env.es.parameters.population.f)
    target_value.append(env.target)
    states.append(state.numpy())
    runs.append(run_id)
    batches.append(0)
    function_id.append(env.fid)
    for batch_id in range(1, n_batches + 1):
        action = actor.act(state) + 1e-10
        next_state, reward, done, _, _ = env.step(action.item())
        state = next_state

        actions.append(action.item())
        rewards.append(reward.numpy())
        lambdas.append(env.es.parameters.lambda_)
        f_curs.append(env.es.parameters.fopt)
        population.append(env.es.parameters.population.f)
        target_value.append(env.target)
        states.append(state.numpy())
        runs.append(run_id)
        batches.append(batch_id)
        function_id.append(env.fid)

        if done:
            break

    print(all(a == 0 for a in actions))
    return {
        "action": actions,
        "reward": rewards,
        "lambda": lambdas,
        "f_cur": f_curs,
        "population": population,
        "target_value": target_value,
        "states": states,
        "run": runs,
        "batch": batches,
        "function_id": function_id,
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

    logs: dict = {}

    if starting_points is not None and len(starting_points) > 0:
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
        env.use_test_set()
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
