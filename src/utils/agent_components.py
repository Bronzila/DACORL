from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributions import Normal

if TYPE_CHECKING:
    import numpy as np


class ConfigurableCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: nn.Module,
    ):
        super().__init__()

        # build model based on hyperparameters
        self.net = nn.Sequential(
            *[
                nn.Linear(state_dim + action_dim, hidden_dim),
                activation(),
                *[
                    module
                    for _ in range(hidden_layers)
                    for module in (
                        nn.Linear(hidden_dim, hidden_dim),
                        activation(),
                    )
                ],
                nn.Linear(hidden_dim, 1),
            ],
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class VectorizedConfigurableCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: nn.Module,
        num_critics: int,
        is_edac_init: bool,
    ):
        super().__init__()

        # build model based on hyperparameters
        self.net = nn.Sequential(
            *[
                VectorizedLinear(
                    state_dim + action_dim,
                    hidden_dim,
                    num_critics,
                ),
                activation(),
                *[
                    module
                    for _ in range(hidden_layers)
                    for module in (
                        VectorizedLinear(hidden_dim, hidden_dim, num_critics),
                        activation(),
                    )
                ],
                VectorizedLinear(hidden_dim, 1, num_critics),
            ],
        )

        if is_edac_init:
            # init as in the EDAC paper
            for layer in self.net[::2]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics,
            dim=0,
        )
        return self.net(state_action).squeeze(-1)


class ConfigurableActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float,
        hidden_layers: int,
        activation: nn.Module,
    ):
        super().__init__()

        # build model based on hyperparameters
        self.net = nn.Sequential(
            *[
                nn.Linear(state_dim, hidden_dim),
                activation(),
                *[
                    module
                    for _ in range(hidden_layers)
                    for module in (
                        nn.Linear(hidden_dim, hidden_dim),
                        activation(),
                    )
                ],
                nn.Linear(hidden_dim, action_dim),
                nn.ReLU(),
            ],
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return -self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(
            state.reshape(1, -1),
            device=device,
            dtype=torch.float32,
        )
        return self(state).cpu().data.numpy().flatten()


class ProbabilisticActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float,
        hidden_layers: int,
        activation: nn.Module,
        is_edac_init: bool = False,
    ):
        super().__init__()
        # build model based on hyperparameters
        self.net = nn.Sequential(
            *[
                nn.Linear(state_dim + action_dim, hidden_dim),
                activation(),
                *[
                    module
                    for _ in range(hidden_layers)
                    for module in (
                        nn.Linear(hidden_dim, hidden_dim),
                        activation(),
                    )
                ],
                nn.Linear(hidden_dim, action_dim),
                nn.ReLU(),
            ],
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        if is_edac_init:
            # init as in the EDAC paper
            for layer in self.trunk[::2]:
                torch.nn.init.constant_(layer.bias, 0.1)

            torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        action = mu if deterministic else policy_dist.rsample()

        log_prob = None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(
                axis=-1,
            )

        return -action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        return self(state, deterministic=deterministic)[0].cpu().numpy()


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features),
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=torch.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / torch.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias
