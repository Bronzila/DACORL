"""SGD environment."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn
from xautodl.models.cell_infers import TinyNetwork

from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils import sgd_utils
from dacbench.envs.env_utils.sgd_utils import random_torchvision_loader

ParameterizedLayerType = nn.Linear | nn.Conv2d


def set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _optimizer_actions(
    optimizer: torch.optim.Optimizer, indices: list[int], actions: list[float]
) -> None:
    for idx, lr in zip(indices, actions, strict=True):
        optimizer.param_groups[idx]["lr"] = lr


def _get_layer_encoding(layer_type: str) -> int:
    if layer_type == "Linear":
        return 0
    if layer_type == "Conv2d":
        return 1
    raise ValueError("Unkown layer type.")


def test(
    model,
    loss_function,
    loader,
    batch_size,
    batch_percentage: float = 1.0,
    device="cpu",
):
    """Evaluate given `model` on `loss_function`.

    Percentage defines how much percentage of the data shall be used.
    If nothing given the whole data is used.

    Returns:
        test_losses: Batch validation loss per data point
    """
    nmb_sets = batch_percentage * (len(loader.dataset) / batch_size)
    model.eval()
    test_losses = []
    test_accuracies = []
    i = 0

    with torch.no_grad():
        for data, target in loader:
            d_data, d_target = data.to(device), target.to(device)
            output = model(d_data)
            if isinstance(model, TinyNetwork):
                _, output = output
                output = nn.functional.log_softmax(output, dim=1)
            _, preds = output.max(dim=1)
            test_losses.append(loss_function(output, d_target))
            test_accuracies.append(torch.sum(preds == d_target) / len(d_target))
            i += 1
            if i >= nmb_sets:
                break
    return (
        torch.cat(test_losses).cpu().numpy(),
        torch.tensor(test_accuracies).cpu().numpy(),
    )


class LayerwiseSGDEnv(AbstractMADACEnv):
    """The SGD DAC Environment implements the problem of dynamically configuring
    the learning rate hyperparameter of a neural network optimizer
    (more specifically, torch.optim.AdamW) for a supervised learning task.
    While training, the model is evaluated after every epoch.

    Actions correspond to learning rate values in [0,+inf[
    For observation space check `observation_space` method docstring.
    For instance space check the `SGDInstance` class docstring
    Reward:
        negative loss of model on test_loader of the instance       if done
        crash_penalty of the instance                               if crashed
        0                                                           otherwise
    """

    metadata = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(self, config):
        """Init env."""
        super().__init__(config)
        torch.use_deterministic_algorithms(True)  # For reproducibility
        self.epoch_mode = config.get("epoch_mode", True)
        self.device = config.get("device")

        self.optimizer_type = torch.optim.SGD
        self.optimizer_params = config.get("optimizer_params")
        self.batch_size = config.get("training_batch_size")
        self.model = config.get("model")
        self.crash_penalty = config.get("crash_penalty")
        self.loss_function = config["loss_function"](**config["loss_function_kwargs"])
        self.dataset_name = config.get("dataset_name")
        self.use_generator = config.get("model_from_dataset")
        self.torchub_model = config.get("torch_hub_model", (False, None, False))

        # Use default reward function, if no specific function is given
        self.get_reward = config.get("reward_function", self.get_default_reward)

        # Use default state function, if no specific function is given
        self.get_states = config.get("state_method", self.get_default_states)

        self.initial_learning_rate = config.get("initial_learning_rate")
        self.state_version = config.get("state_version")
        self.initial_seed = config.get("seed")
        self.seed(self.initial_seed)

        self.instance_set_path = config.get("instance_set_path")
        self.dataset_path = config.get("dataset_path")
        self.fraction_of_dataset = config.get("fraction_of_dataset")
        self.train_validation_ratio = config.get("train_validation_ratio")
        self.instance_mode = config.get("instance_mode")
        self.instance_set = config.get("instance_set")
        self.inst_id = 0

        self.predictions = deque(torch.zeros(2))

    def step(self, actions: list[float]):
        """Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step).
        """
        truncated = super().step_()
        info = {}

        log_learning_rates = actions
        self.learning_rates = [
            10**log_learning_rate for log_learning_rate in log_learning_rates
        ]

        # Update action history
        self._update_lr_histories(log_learning_rates)

        _optimizer_actions(
            self.optimizer, self.adaptable_layer_indices, self.learning_rates
        )

        if self.epoch_mode:
            self.train_loss, self.average_loss = self.run_epoch(
                self.model,
                self.loss_function,
                self.train_loader,
                self.optimizer,
                self.device,
            )
        else:
            train_args = [
                self.model,
                self.loss_function,
                self.train_loader,
                self.device,
            ]
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train_loss, self.train_accuracy = self.forward_backward(*train_args)

        crashed = (
            not np.isfinite(self.train_loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )
        self.train_loss = self.train_loss.item()

        if crashed:
            self._done = True
            return (
                self.get_states(),
                torch.tensor(self.crash_penalty),
                False,
                True,
                info,
            )

        self._done = truncated

        if (
            self.c_step % len(self.train_loader) == 0 or self._done
        ):  # Calculate validation loss at the end of an epoch
            batch_percentage = 1.0
        else:
            batch_percentage = 0.1

        val_args = [
            self.model,
            self.loss_function,
            self.validation_loader,
            self.batch_size,
            batch_percentage,
            self.device,
        ]
        validation_loss, validation_accuracy = test(*val_args)

        self.validation_loss = validation_loss.mean()
        self.validation_accuracy = validation_accuracy.mean()
        if (
            self.min_validation_loss is None
            or self.validation_loss <= self.min_validation_loss
        ):
            self.min_validation_loss = self.validation_loss

        # Calculate test loss after every epoch + when done
        if self._done or self.c_step % len(self.train_loader) == 0:
            val_args = [
                self.model,
                self.loss_function,
                self.test_loader,
                self.batch_size,
                1.0,
                self.device,
            ]
            test_losses, self.test_accuracies = test(*val_args)
            self.test_loss = test_losses.mean()
            self.test_accuracy = self.test_accuracies.mean()

        reward = self.get_reward()

        return self.get_states(), reward, truncated, truncated, info

    def reset(self, seed=None, options=None):
        """Initialize the neural network, data loaders, etc. for given/random next task.
        Also perform a single forward/backward pass,
        not yet updating the neural network parameters.
        """
        super().reset_(seed=seed, scheme="round_robin")
        if options is None:
            options = {}

        # Set global seed for data loaders
        if self.instance_mode == "random_seed":
            run_seed = self.rng.integers(0, 1000000000)
        elif self.instance_mode == "instance_set":
            run_seed = self.instance[1]
        else:
            run_seed = self.initial_seed
        set_global_seeds(run_seed)

        print(f"run_seed: {run_seed}", flush=True)

        # Get loaders for instance
        self.datasets, loaders = random_torchvision_loader(
            run_seed,
            self.dataset_path,
            (
                self.dataset_name
                if self.instance_mode != "instance_set"
                else self.instance[0]
            ),
            self.batch_size,
            self.fraction_of_dataset,
            self.train_validation_ratio,
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

        self.epoch_length = len(self.train_loader)

        if self.instance_mode == "random_instances":
            (
                self.model,
                self.optimizer_params,
                self.batch_size,
                self.crash_penalty,
            ) = sgd_utils.random_instance(self.rng, self.datasets)
        elif self.instance_mode == "instance_set":
            self.model = sgd_utils.create_model(
                self.instance[2], len(self.datasets[0].classes)
            )
        elif self.instance_mode == "random_seed":
            self.model = sgd_utils.create_model(
                self.config.get("layer_specification"), len(self.datasets[0].classes)
            )
        else:
            raise NotImplementedError(
                f"No implementation for instance version: {self.instance_mode}"
            )

        self.optimizer_type = torch.optim.SGD
        self.info = {}
        self._done = False

        self.model.to(self.device)

        # create param groups then feed them into optimizer
        (
            param_groups,
            self.layer_types,
            self.adaptable_layer_indices,
        ) = self._create_param_groups()
        self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            param_groups,
            **self.optimizer_params,
        )

        self.n_adaptable_layers = len(self.adaptable_layer_indices)
        self.learning_rates = [self.initial_learning_rate] * self.n_adaptable_layers
        self.lr_histories = [
            deque(torch.ones(5) * math.log10(self.initial_learning_rate))
        ] * self.n_adaptable_layers
        # Evaluate model initially
        train_args = [
            self.model,
            self.loss_function,
            self.train_loader,
            self.batch_size,
            1.0,
            self.device,
        ]
        losses, train_accuracy = test(*train_args)
        self.train_loss = losses.mean()
        self.train_accuracy = train_accuracy.mean()

        test_args = [
            self.model,
            self.loss_function,
            self.test_loader,
            self.batch_size,
            1.0,
            self.device,
        ]
        test_losses, test_accuracies = test(*test_args)
        self.test_loss = test_losses.mean()
        self.test_accuracy = test_accuracies.mean()

        val_args = [
            self.model,
            self.loss_function,
            self.validation_loader,
            self.batch_size,
            1.0,
            self.device,
        ]
        validation_loss, validation_accuracy = test(*val_args)

        self.validation_loss = validation_loss.mean()
        self.validation_accuracy = validation_accuracy.mean()

        self.min_validation_loss = None

        if self.epoch_mode:
            self.average_loss = 0

        return self.get_states(), {}

    def _create_param_groups(self) -> tuple[list[Any], list[Any]]:
        param_groups = []
        layer_types = []
        adaptable_layer_indices = []

        def has_trainable_parameters(layer: torch.nn.Module) -> bool:
            return any(p.requires_grad for p in layer.parameters())

        def extract_adaptable_and_trainable_layers(module):
            for child in module.children():
                if has_trainable_parameters(child) and list(child.children()) == []:
                    if isinstance(
                        child, torch.nn.Linear | torch.nn.Conv2d
                    ):  # TODO: Layertype anpassen
                        param_groups.append(
                            {
                                "params": child.parameters(),
                                "lr": self.initial_learning_rate,
                            }
                        )
                        layer_types.append(f"{type(child).__name__}")
                        adaptable_layer_indices.append(len(param_groups) - 1)
                    else:
                        param_groups.append(
                            {
                                "params": child.parameters(),
                                "lr": self.initial_learning_rate,
                            }
                        )
                else:
                    # Recursively apply to child modules
                    extract_adaptable_and_trainable_layers(child)

        extract_adaptable_and_trainable_layers(self.model)
        return param_groups, layer_types, adaptable_layer_indices

    def _add_to_param_groups(
        self, layer: nn.Module, param_groups: list[dict], layer_types: list[str]
    ) -> list[dict] | list[str]:
        param_groups.append(
            {
                "params": layer.parameters(),
                "lr": self.initial_learning_rate,
            }
        )
        layer_types.append(type(layer).__name__)
        return param_groups, layer_types

    def get_default_reward(self) -> torch.tensor:
        """The default reward function.

        Returns:
            float: The calculated reward
        """
        return torch.tensor(-self.validation_loss)

    def get_default_states(self) -> list[torch.Tensor]:
        """Default state function.

        Returns:
            list[dict]: The current states of all layers
        """
        states = []
        # Global observations
        remaining_budget = torch.tensor(
            [(self.n_steps - self.c_step) / self.n_steps], device=self.device
        )
        is_train_loss_finite = int(np.isfinite(self.train_loss))
        loss_ratio = np.log(self.validation_loss / self.train_loss)

        global_observations_tensor = torch.cat(
            [
                remaining_budget,
                torch.tensor([is_train_loss_finite], device=self.device),
                torch.tensor(
                    [math.log10(self.initial_learning_rate)], device=self.device
                ),
                torch.tensor([self.train_loss], device=self.device),
                torch.tensor([self.validation_loss], device=self.device),
                torch.tensor([loss_ratio], device=self.device),
                torch.tensor([self.train_accuracy.item()], device=self.device),
                torch.tensor([self.validation_accuracy], device=self.device),
            ]
        )

        # Layerspecific observations
        for layer_idx, learning_rate, layer_type, lr_history in zip(
            self.adaptable_layer_indices,
            self.learning_rates,
            self.layer_types,
            self.lr_histories,
            strict=True,
        ):
            param_group = self.optimizer.param_groups[layer_idx]
            local_observations = []
            # Layer encoding
            layer_enc = _get_layer_encoding(layer_type)
            local_observations.append(torch.tensor([layer_enc], device=self.device))

            log_learning_rate = (
                np.log10(learning_rate) if learning_rate != 0 else np.log10(1e-10)
            )
            local_observations.append(
                torch.tensor([log_learning_rate], device=self.device)
            )
            lr_hist_deltas = log_learning_rate - lr_history
            local_observations.append(torch.tensor(lr_hist_deltas, device=self.device))

            depth_enc = layer_idx / len(self.optimizer.param_groups)
            local_observations.append(torch.tensor([depth_enc], device=self.device))

            # Weight, gradient and momentum statistics
            weights_all = []
            grads_all = []
            velocities_all = []

            for param in param_group["params"]:
                # Weights
                weights = param.data.view(-1)
                weights_all.append(weights)

                # Gradients
                if param.grad is not None:
                    grads_all.append(param.grad.view(-1))
                else:
                    assert self.c_step < 2
                    grads_all.append(torch.zeros_like(weights))

                # Velocities
                state = self.optimizer.state[param]
                if "momentum_buffer" in state:
                    velocities_all.append(state["momentum_buffer"].view(-1))
                else:
                    assert self.c_step < 2
                    velocities_all.append(torch.zeros_like(weights))

            # Concatenate all
            weights_tensor = torch.cat(weights_all)
            weights_mean = weights_tensor.mean().unsqueeze(0).to(self.device)
            weights_var = weights_tensor.var().unsqueeze(0).to(self.device)
            weights_norm = weights_tensor.norm(p=2).unsqueeze(0).to(self.device)

            grads_tensor = torch.cat(grads_all)
            grads_mean = grads_tensor.mean().unsqueeze(0).to(self.device)
            grads_var = grads_tensor.var().unsqueeze(0).to(self.device)
            grads_norm = grads_tensor.norm(p=2).unsqueeze(0).to(self.device)

            velocities_tensor = torch.cat(velocities_all)
            velocities_mean = velocities_tensor.mean().unsqueeze(0).to(self.device)
            velocities_var = velocities_tensor.var().unsqueeze(0).to(self.device)
            velocities_norm = velocities_tensor.norm(p=2).unsqueeze(0).to(self.device)

            local_observations.extend(
                [
                    weights_mean,
                    weights_var,
                    weights_norm,
                    grads_mean,
                    grads_var,
                    grads_norm,
                    velocities_mean,
                    velocities_var,
                    velocities_norm,
                ]
            )

            local_observations_tensor = torch.cat(local_observations)
            layer_state = torch.cat(
                [local_observations_tensor, global_observations_tensor]
            )
            states.append(layer_state)

        return states

    def render(self, mode="human"):
        """Render progress."""
        if mode == "human":
            epoch = 1 + self.c_step // len(self.train_loader)
            epoch_cutoff = self.n_steps // len(self.train_loader)
            batch = 1 + self.c_step % len(self.train_loader)
            print(
                f"prev_lr {self.optimizer.param_groups[0]['lr'] if self.n_steps > 0 else None}, "  # noqa: E501
                f"epoch {epoch}/{epoch_cutoff}, "
                f"batch {batch}/{len(self.train_loader)}, "
                f"batch_loss {self.train_loss}, "
                f"val_loss {self.validation_loss}"
            )
        else:
            raise NotImplementedError

    def forward_backward(self, model, loss_function, loader, device="cpu"):
        """Do a forward and a backward pass for given `model` for `loss_function`.

        Returns:
            loss: Mini batch training loss per data point
        """
        model.train()
        (data, target) = next(iter(loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        if isinstance(model, TinyNetwork):
            _, output = output
            output = nn.functional.log_softmax(output, dim=1)
        loss = loss_function(output, target)
        loss.mean().backward()

        # for comparing current and last predictions
        self.predictions.pop()
        self.predictions.appendleft(output)

        accuracy = torch.sum(output.argmax(dim=1) == target) / len(target)
        return loss.mean().detach().cpu().numpy(), torch.tensor(accuracy).cpu().numpy()

    def run_epoch(self, model, loss_function, loader, optimizer, device="cpu"):
        """Run a single epoch of training for given `model` with `loss_function`."""
        last_loss = None
        running_loss = 0
        predictions = []
        model.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.mean().backward()
            optimizer.step()
            predictions.append(output)
            last_loss = loss
            running_loss += last_loss.mean().item()
        self.predictions.pop()
        self.predictions.appendleft(predictions.mean())
        return last_loss.mean().detach().cpu(), running_loss.cpu() / len(loader)

    def _update_lr_histories(self, log_learning_rates: list[float]) -> None:
        for lr_history, log_lr in zip(
            self.lr_histories, log_learning_rates, strict=True
        ):
            lr_history.pop()
            lr_history.appendleft(log_lr)

    def seed(self, seed, seed_action_space=False):
        super().seed(seed, seed_action_space)

        self.rng = np.random.default_rng(seed)
