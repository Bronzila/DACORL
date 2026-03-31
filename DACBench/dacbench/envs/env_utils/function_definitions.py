from __future__ import annotations

import numpy as np
import torch


class Rosenbrock:
    def __init__(self) -> None:
        self.x_min = torch.tensor([1, 1])
        self.f_min = 0

    def objective_function(self, input: torch.Tensor) -> torch.Tensor:
        x = input[0]
        y = input[1]
        return torch.pow((1 - x), 2) + 100 * torch.pow((y - x**2), 2)


class Rastrigin:
    def __init__(self) -> None:
        self.x_min = torch.tensor([0, 0])
        self.f_min = 0

    def objective_function(self, input: torch.Tensor) -> torch.Tensor:
        x = input[0]
        y = input[1]
        return (
            20
            + torch.pow(x, 2)
            - 10 * torch.cos(2 * np.pi * x)
            + torch.pow(y, 2)
            - 10 * torch.cos(2 * np.pi * y)
        )


class Ackley:
    def __init__(self) -> None:
        self.x_min = torch.tensor([0, 0])
        self.f_min = 0

    def objective_function(self, input: torch.Tensor) -> torch.Tensor:
        x = input[0]
        y = input[1]
        return (
            -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2)))
            - torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y)))
            + torch.exp(torch.tensor(1))
            + 20
        )


class Sphere:
    def __init__(self) -> None:
        self.x_min = torch.tensor([0, 0])
        self.f_min = 0

    def objective_function(self, input: torch.Tensor) -> torch.Tensor:
        x = input[0]
        y = input[1]
        return x**2 + y**2  # type: ignore
