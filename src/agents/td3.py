from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, List

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    import numpy as np

TensorBatch = List[torch.Tensor]

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(
        target.parameters(),
        source.parameters(),
    ):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data,
        )


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: int,
        min_action: int,
        dropout_rate: float,
        hidden_dim: int,
        tanh_scaling: bool = False,
        action_positive: bool = False,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, action_dim),
        )

        self._max_action = max_action
        self._min_action = min_action
        self._tanh_scaling = tanh_scaling
        self._pos_act = action_positive

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action: torch.Tensor = self.net(state)
        if self._tanh_scaling:
            tanh_action = torch.tanh(action)
            # instead of [-1,1] -> [self.min_action, self.max_action]
            action = (tanh_action - 1) * (
                (self._max_action - self._min_action) / 2
            ) + self._max_action
        else:
            # Have to do it this way to avoid in-place operation
            if self._pos_act:
                relu_action = torch.nn.functional.relu(action)
            else:
                relu_action = - torch.nn.functional.relu(action)
            relu_action.clamp_(self._min_action, self._max_action)
            action = relu_action

        return action

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(
            state.reshape(1, -1),
            device=device,
            dtype=torch.float32,
        )
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3:
    def __init__(
        self,
        max_action: float,
        min_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.device = device

    def select_action(self, state: torch.Tensor) -> np.ndarray:
        state = state.reshape(1, -1)
        out: torch.Tensor = self.actor(state)
        return out.cpu().data.numpy().flatten()

    def train(self, batch: TensorBatch) -> dict[str, float]:
        log_dict = {}
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, not_done = batch

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip,
                self.noise_clip,
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.min_action,
                self.max_action,
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_q1, target_q2).cpu()
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_Q) + F.mse_loss(
            current_q2,
            target_Q,
        )

        log_dict["critic_loss"] = critic_loss.item()

        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            log_dict["actor_loss"] = actor_loss.item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(
            state_dict["critic_1_optimizer"],
        )
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(
            state_dict["critic_2_optimizer"],
        )
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]
