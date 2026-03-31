"""CMA ES Environment."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
import torch
from IOHexperimenter import IOH_function
from modcma import ModularCMAES, Parameters

from dacbench import AbstractMADACEnv


def _norm(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(x)))  # type: ignore


class CMAESEnv(AbstractMADACEnv):
    """The CMA ES environment controlles the step size on BBOB functions."""

    def __init__(self, config: dict) -> None:
        """Initialize the environment."""
        super().__init__(config)

        self.es: ModularCMAES
        self.budget = config.budget  # type: ignore
        self.total_budget = self.budget

        self._hist_len = 10

        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state: Callable[[CMAESEnv], torch.Tensor] = config.get(
            "state_method", self.get_default_state
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        if options is None:
            options = {}
        super().reset_(seed)
        self.dim, self.fid, self.iid, self.init_sigma, self.init_pop = self.instance
        self.objective = IOH_function(
            self.fid, self.dim, self.iid, target_precision=1e-8
        )
        self.target = self.objective.get_target()

        parameters = Parameters.from_config_array(
            self.dim, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(int)
        )
        if "starting_point" in options:
            parameters.sigma = options["starting_point"]["sigma"][0]
            parameters.m = options["starting_point"][1]
        else:
            parameters.sigma = self.init_sigma
            parameters.m = np.array(self.init_pop).reshape(self.dim, 1)
        parameters.budget = self.budget
        self.es = ModularCMAES(self.objective, parameters=parameters)

        self.es.step()
        self.c_step = 0

        self._chi_N = self.dim**0.5 * (
            1 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2)
        )

        self._xopts = deque(np.zeros((self._hist_len, self.dim)))
        self._fopts = deque(np.ones(self._hist_len) * self.es.parameters.fopt)
        self._f = deque(np.zeros(self._hist_len, self.es.parameters.lambda_))
        self._f[0] = self.es.parameters.population.f
        self._sigma_hist = deque(np.zeros(self._hist_len))
        self._sigma_hist[0] = self.init_sigma

        self.norm_delta_f = lambda p, q: (p - q) / (abs(p - q) + abs(q) + 1e-5)
        delta_bounds = self.es.parameters.ub - self.es.parameters.lb
        self.norm_delta_x = lambda p, q: (p - q) / delta_bounds

        self._delta_f_opt = deque(torch.zeros(self._hist_len))
        self._delta_f = deque(torch.zeros(2, self.es.parameters.lambda_))

        return self.get_state(self), {"start": [self.init_sigma, self.init_pop]}

    def step(
        self, action: float
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        """Make one step of the environment."""
        truncated = super().step_()

        self.es.parameters.sigma = action

        self._sigma_hist.pop()
        self._sigma_hist.appendleft(torch.tensor(action))

        terminated = not self.es.step()

        self._cur_ps = _norm(self.es.parameters.ps) / self._chi_N - 1

        self._f.pop()
        self._f.appendleft(self.es.parameters.population.f)

        self._xopts.pop()
        self._xopts.appendleft(self.es.parameters.xopt)

        self._fopts.pop()
        self._fopts.appendleft(self.es.parameters.fopt)

        return self.get_state(self), self.get_reward(self), terminated, truncated, {}

    def close(self) -> bool:
        """Closes the environment."""
        return True

    def get_default_reward(self, *_: tuple) -> torch.Tensor:
        """The default reward function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            float: The calculated reward
        """
        return torch.tensor(-self.es.parameters.fopt)

    def get_default_state(self, *_: tuple) -> torch.Tensor:
        """Default state function.

        Args:
            _ (_type_): Empty parameter, which can be used when overriding

        Returns:
            dict: The current state
        """
        # Inter-generational delta f (normalized diff between max of the last generation)
        self._delta_f_opt.pop()
        self._delta_f_opt.appendleft(self.norm_delta_f(self._fopts[0], self._fopts[1]))

        # normalized diff between function values of 2 consecutive generations
        self._delta_f.pop()
        self._delta_f.appendleft(
            (self.norm_delta_f(self._f[0], self._f[1])).astype(np.float32)
        )
        # # Intra-generational delta f (normalized diff between max and min fitness of current pop)
        # best_f = np.min(self.es.parameters.population.f)
        # worst_f = np.max(self.es.parameters.population.f)
        # intra_delta_f.append(self.norm_delta_f(best_f, worst_f))

        # # Inter-generational delta X (normalized diff between the best genotypes in two consecutive generations)
        # inter_delta_x.append(self.norm_delta_x(self._xopts[i], self._xopts[i+1]))

        # # Intra-generational delta X (normalized diff between the best and worst genotypes in pop)
        # best_x = self.es.parameters.population.x[np.argmax(self.es.parameters.population.f)]
        # worst_x = self.es.parameters.population.x[np.argmin(self.es.parameters.population.f)]
        # intra_delta_x.append(self.norm_delta_x(best_x, worst_x))

        return torch.concat(
            (
                torch.tensor(
                    [
                        (self.es.parameters.budget - self.es.parameters.used_budget)
                        / self.es.parameters.budget
                    ]
                ),
                torch.tensor([self.es.parameters.sigma]),
                torch.tensor([self.es.parameters.population.f.mean()]),
                torch.tensor([self.es.parameters.population.f.std()]),
                torch.tensor(self._delta_f_opt),
                torch.tensor(self.es.parameters.ps.reshape(-1)),
                # torch.tensor(self._delta_f).flatten(),
                torch.tensor(self._sigma_hist),
                torch.tensor([self.fid]),
                torch.tensor([self.iid]),
            )
        )

    def render(self, mode: str = "human") -> None:
        """Render progress."""
        raise NotImplementedError("CMA-ES does not support rendering at this point")
