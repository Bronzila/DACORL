from typing import Dict, Optional, Tuple, Union
import math
import numpy as np
import torch
import pandas as pd
from dacbench.envs.env_utils.function_definitions import Rosenbrock, Rastrigin, Ackley, Sphere

from dacbench import AbstractEnv

class ToySGD2DEnv(AbstractEnv):
    """
    Optimize toy functions with SGD + Momentum.

    Action: [log_learning_rate, log_momentum] (log base 10)
    State: Dict with entries remaining_budget, gradient, learning_rate, momentum
    Reward: negative log regret of current and true function value

    An instance can look as follows:
    ID                                                  0
    function                                   Rosenbrock
    low                                                -5
    high                                                5

    """

    def __init__(self, config):
        """Init env."""
        super(ToySGD2DEnv, self).__init__(config)
        self.dimensions = 2
        self.velocity = torch.zeros(self.dimensions)
        self.gradient = torch.zeros(self.dimensions)
        self.history = []
        self.problem = None
        self.objective_function = None
        self.x_min = None
        self.f_min = None
        self.x_cur = None
        self.f_cur = None
        self.momentum = config.get("initial_momentum", 0.9)
        self.initial_momentum = config.get("initial_momentum", 0.9)
        self.learning_rate = config["initial_learning_rate"]
        self.initial_learning_rate = config["initial_learning_rate"]
        self.lower_bound = config["low"]
        self.upper_bound = config["high"]
        self.function = config["function"]
        self.state_version = config["state_version"]
        self.reward_version = config["reward_version"]
        self.boundary_termination = config["boundary_termination"]
        self.lr_history = torch.ones(5) * math.log10(self.initial_learning_rate)
        self.seed(config["seed"])
    
    def seed(self, seed, seed_action_space=False):
        super(ToySGD2DEnv, self).seed(seed, seed_action_space)

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


    def build_objective_function(self):
        """Make base function."""
        if self.function == "Rosenbrock":
            self.problem = Rosenbrock()
        elif self.function == "Rastrigin":
            self.problem = Rastrigin()
        elif self.function == "Ackley":
            self.problem = Ackley()
        elif self.function == "Sphere":
            self.problem = Sphere()
        else:
            raise NotImplementedError(
                "Function not found."
            )
        self.x_min = self.problem.x_min
        self.f_min = self.problem.f_min
        self.objective_function = self.problem.objective_function

    def clip_gradient(self):
        self.gradient = torch.clip(self.gradient, -100, 100)

    def get_state(self):
        remaining_budget = self.n_steps - self.c_step
        log_learning_rate = math.log10(self.learning_rate) if self.learning_rate != 0 else math.log10(1e-10)
        if self.state_version == "basic":
            state = [-self.c_step, log_learning_rate, self.gradient[0], self.gradient[1]]
        elif self.state_version == "extended":
            lr_hist_deltas = self.lr_history - log_learning_rate
            # Flag indicating whether agent is inside optimization bounds or outside
            is_outside_bounds = ((self.x_cur <= self.lower_bound) | (self.x_cur >= self.upper_bound)).any()

            # state = torch.cat(
            #     (torch.tensor([remaining_budget]),
            #     torch.tensor([log_learning_rate]),
            #     lr_hist_deltas[1:], # First value is the difference to current learning rate --> always 0
            #     torch.tensor([self.gradient[0], self.gradient[1], is_outside_bounds.to(torch.int)]))
            # )

            # EXPERIMENTAL
            norm_grad = self.gradient / 100
            norm_budget = remaining_budget / self.n_steps
            state = torch.cat(
                (torch.tensor([norm_budget]),
                torch.tensor([log_learning_rate]),
                lr_hist_deltas[1:], # First value is the difference to current learning rate --> always 0
                torch.tensor([norm_grad[0], norm_grad[1], is_outside_bounds.to(torch.int)]))
            )
        elif self.state_version == "lr_hist":
            lr_hist_deltas = self.lr_history - log_learning_rate
            norm_budget = remaining_budget / self.n_steps
            state = torch.cat(
                (torch.tensor([norm_budget]),
                torch.tensor([log_learning_rate]),
                lr_hist_deltas[1:], # First value is the difference to current learning rate --> always 0
                torch.tensor([self.gradient[0], self.gradient[1]]))
            )
        elif self.state_version == "basic_norm":
            norm_grad = self.gradient / 100
            norm_budget = remaining_budget / self.n_steps
            state = torch.cat(
                (torch.tensor([norm_budget]),
                torch.tensor([log_learning_rate]),
                torch.tensor([norm_grad[0], norm_grad[1]]))
            )
        elif self.state_version == "extended_velocity":
            lr_hist_deltas = self.lr_history - log_learning_rate
            norm_budget = remaining_budget / self.n_steps
            norm_grad = self.gradient / 100
            norm_x_cur = 2 * (self.x_cur - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1
            state = torch.cat(
                (torch.tensor([norm_budget]),
                torch.tensor([log_learning_rate]),
                lr_hist_deltas[1:], # First value is the difference to current learning rate --> always 0
                torch.tensor([norm_grad[0], norm_grad[1], self.velocity[0], self.velocity[1], norm_x_cur[0], norm_x_cur[1]]))
            )
        elif self.state_version == "extended_momentum":
            lr_hist_deltas = self.lr_history - log_learning_rate
            norm_budget = remaining_budget / self.n_steps
            norm_grad = self.gradient / 100
            momentum_term = self.momentum * self.velocity
            norm_x_cur = 2 * (self.x_cur - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1
            state = torch.cat(
                (torch.tensor([norm_budget]),
                torch.tensor([log_learning_rate]),
                lr_hist_deltas[1:], # First value is the difference to current learning rate --> always 0
                torch.tensor([norm_grad[0], norm_grad[1], momentum_term[0], momentum_term[1], norm_x_cur[0], norm_x_cur[1]]))
            )
        elif self.state_version == "extended_all":
            lr_hist_deltas = self.lr_history - log_learning_rate
            norm_budget = remaining_budget / self.n_steps
            norm_grad = self.gradient / 100
            momentum_term = self.momentum * self.velocity
            norm_x_cur = 2 * (self.x_cur - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1
            state = torch.cat(
                (torch.tensor([norm_budget]),
                torch.tensor([log_learning_rate]),
                lr_hist_deltas[1:], # First value is the difference to current learning rate --> always 0
                torch.tensor([norm_grad[0], norm_grad[1], momentum_term[0], momentum_term[1], self.velocity[0], self.velocity[1], norm_x_cur[0], norm_x_cur[1]]))
            )

        return torch.tensor(state)

    def step(
        self, action: float
    ) -> Tuple[Dict[str, float], float, bool, Dict]:
        """
        Take one step with SGD.

        Parameters
        ----------
        action: float
            log_learning_rate

        Returns
        -------
        Tuple[torch.tensor, float, bool, Dict]

            - state : torch.tensor
            - reward : float
            - terminated : bool
            - truncated : bool
            - info : Dict

        """
        truncated = super(ToySGD2DEnv, self).step_()
        info = {}

        # parse action
        log_learning_rate = action
        self.learning_rate = 10**log_learning_rate

        # Update action history
        self.lr_history[1:] = self.lr_history[:-1].clone()
        self.lr_history[0] = log_learning_rate

        # Calculate function value when not changing the learning rate
        if self.reward_version == "difference":
            prev_lr = 10 ** self.lr_history[1]
            velocity_prev_lr = self.momentum * self.velocity + prev_lr * self.gradient
            x_prev_lr = self.x_cur - velocity_prev_lr
            f_prev_lr = self.objective_function(x_prev_lr)

        # SGD + Momentum update
        self.velocity = (
            self.momentum * self.velocity + self.learning_rate * self.gradient
        )
        self.x_cur -= self.velocity

        # Clip position to optimization bounds if out of bounds
        is_optimizee_out_of_bounds = False
        if torch.any(self.x_cur > self.upper_bound) or torch.any(self.x_cur < self.lower_bound):
            self.x_cur = torch.clip(self.x_cur, self.lower_bound, self.upper_bound)
            is_optimizee_out_of_bounds = True
        
        # Reward
        # current function value
        # Here we need to initialize a new tensor, because we used x_cur in an in-place operation above
        # Moreover this is important to "reset" the gradients tracked by x_cur_tensor
        x_cur_tensor = torch.tensor(self.x_cur, requires_grad=True)
        self.f_cur = self.objective_function(x_cur_tensor)
        # Gradient
        self.f_cur.backward()
        self.gradient = x_cur_tensor.grad
        self.clip_gradient()
        log_regret = torch.log10(torch.abs(self.f_min - self.f_cur))

        if self.reward_version == "regret":
            reward = -log_regret
        elif self.reward_version == "difference":
            log_regret_prev_lr = torch.log10(torch.abs(self.f_min - f_prev_lr))
            # Difference reward specified as R = R(a_cur) - R(a_no-op)
            # Since our reward is negative log regret R = -log(regret_cur) - (-log(regret_no-op))
            reward = log_regret_prev_lr - log_regret
            
        # Clip reward to disallow inf values due to log
        reward = torch.minimum(reward, torch.tensor(20))

        # State
        state = self.get_state()

        self.history.append(self.x_cur)

        # If optimizee is out of bounds, penalize and terminate run
        if self.boundary_termination and is_optimizee_out_of_bounds:
            reward = -5
            return state, torch.tensor(reward), True, truncated, info

        return state, reward.detach(), False, truncated, info

    def reset(self, seed=None, options={}):
        """
        Reset environment.

        Parameters
        ----------
        seed : int
            seed
        options : dict
            options dict (not used)

        Returns
        -------
        torch.Tensor
            Environment state
        dict
            Meta-info

        """
        super(ToySGD2DEnv, self).reset_(seed)

        if seed is not None:
            self.seed(seed)
        self.velocity = torch.zeros(self.dimensions)
        self.gradient = torch.zeros(self.dimensions)
        self.history = []
        self.objective_function = None
        self.x_min = None
        self.f_min = None
        self.problem = None
        self.momentum = self.initial_momentum
        self.learning_rate = self.initial_learning_rate
        # self.n_steps = 0
        self.build_objective_function()
        if "starting_point" in options:
            self.x_cur = options["starting_point"]
        else:
            self.x_cur = torch.FloatTensor(2).uniform_(self.lower_bound, self.upper_bound, generator=self.rng)
        
        # calculate f_cur and gradient
        x_cur_tensor = torch.tensor(self.x_cur, requires_grad=True)
        self.f_cur = self.objective_function(x_cur_tensor)
        self.f_cur.backward()
        self.gradient = x_cur_tensor.grad
        self.clip_gradient()
        
        self.lr_history = torch.ones(5) * math.log10(self.initial_learning_rate)        
        return self.get_state(), {"start": self.x_cur.tolist()}

    def render(self, **kwargs):
        """Render progress."""
        import matplotlib.pyplot as plt

        history = np.array(self.history).flatten()
        X = np.linspace(1.05 * np.amin(history), 1.05 * np.amax(history), 100)
        Y = self.objective_function(X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X, Y, label="True")
        ax.plot(
            history,
            self.objective_function(history),
            marker="x",
            color="black",
            label="Observed",
        )
        ax.plot(
            self.x_cur,
            self.objective_function(self.x_cur),
            marker="x",
            color="red",
            label="Current Optimum",
        )
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")        
        plt.show()

    def close(self):
        """Close env."""
        pass
