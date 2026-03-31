import math
import numpy as np
import torch

from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils.utils import random_torchvision_loader


def optimizer_action(optimizer: torch.optim.Optimizer, action: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = action
    return optimizer


def test(
    model,
    loss_function,
    loader,
    batch_size,
    batch_percentage: float = 1.0,
    device="cpu",
):
    """Evaluate given `model` on `loss_function`.

    Percentage defines how much percentage of the data shall be used. If nothing given the whole data is used.

    Returns:
        test_losses: Batch validation loss per data point
    """
    nmb_sets = batch_percentage * (len(loader.dataset) / batch_size)
    model.eval()
    test_losses = []
    i = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_losses.append(loss_function(output, target))
            i += 1
            if i >= nmb_sets:
                break
    test_losses = torch.cat(test_losses)
    return test_losses


def forward_backward(model, loss_function, loader, device="cpu"):
    """Do a forward and a backward pass for given `model` for `loss_function`.

    Returns:
        loss: Mini batch training loss per data point
    """
    model.train()
    (data, target) = next(iter(loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_function(output, target)
    loss.mean().backward()
    return loss


class SGDEnv(AbstractMADACEnv):
    """
    The SGD DAC Environment implements the problem of dynamically configuring the learning rate hyperparameter of a
    neural network optimizer (more specifically, torch.optim.AdamW) for a supervised learning task. While training,
    the model is evaluated after every epoch.

    Actions correspond to learning rate values in [0,+inf[
    For observation space check `observation_space` method docstring.
    For instance space check the `SGDInstance` class docstring
    Reward:
        negative loss of model on test_loader of the instance       if done
        crash_penalty of the instance                               if crashed
        0                                                           otherwise
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config):
        """Init env."""
        super(SGDEnv, self).__init__(config)
        self.device = config.get("device")

        # self.learning_rate = None
        self.optimizer_type = torch.optim.AdamW
        self.optimizer_params = config.get("optimizer_params")
        self.batch_size = config.get("training_batch_size")
        self.model = config.get("model")
        self.crash_penalty = config.get("crash_penalty")
        self.loss_function = config.loss_function(**config.loss_function_kwargs)

        self.learning_rate = config.get("initial_learning_rate")
        self.initial_learning_rate = config.get("initial_learning_rate")
        self.state_version = config.get("state_version")

        # Get loaders for instance
        self.datasets, loaders = random_torchvision_loader(
            config.get("seed"),
            config.get("instance_set_path"),
            config.get("dataset"),  # If set to None, random data set is chosen; else specific set can be set: e.g. "MNIST"
            self.batch_size,
            config.get("fraction_of_dataset"),
            config.get("train_validation_ratio"),
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

        self.test_args = {
            "model": self.model,
            "loss_function": self.loss_function,
            "loader": self.test_loader,
            "batch_size": self.batch_size,
            "batch_percentage": 1.0,
            "device": self.device,
        }
        
        self.val_args = {
            "model": self.model,
            "loss_function": self.loss_function,
            "loader": self.validation_loader,
            "batch_size": self.batch_size,
            "device": self.device,
        }

        self.train_args = {
            "model": self.model,
            "loss_function": self.loss_function,
            "loader": self.train_loader,
            "device": self.device,
        }

    def step(self, action: float):
        """
        Update the parameters of the neural network using the given learning rate lr,
        in the direction specified by AdamW, and if not done (crashed/cutoff reached),
        performs another forward/backward pass (update only in the next step)."""
        truncated = super(SGDEnv, self).step_()
        info = {}

        log_learning_rate = action
        self.learning_rate = 10 ** log_learning_rate
        

        self.optimizer = optimizer_action(self.optimizer, self.learning_rate)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.loss = forward_backward(**self.train_args)

        crashed = (
            not torch.isfinite(self.loss).any()
            or not torch.isfinite(
                torch.nn.utils.parameters_to_vector(self.model.parameters())
            ).any()
        )

        if crashed:
            return (
                self._get_state(crashed),
                torch.tensor(-self.crash_penalty),
                False,
                True,
                info,
            )

        self._done = truncated

        if (
            self.n_steps % len(self.train_loader) == 0 or self._done
        ):  # Calculate validation loss at the end of an epoch
            batch_percentage = 1.0
        else:
            batch_percentage = 0.1

        validation_loss = test(**self.val_args, batch_percentage=batch_percentage)

        self.validation_loss = validation_loss.mean()
        if (
            self.min_validation_loss is None
            or self.validation_loss <= self.min_validation_loss
        ):
            self.min_validation_loss = self.validation_loss

        # if self._done:
        test_losses = test(**self.test_args)
        self.test_loss = test_losses.mean()
        reward = -test_losses.sum().item() / len(self.test_loader.dataset)
        # else:
        #     reward = 0.0
        return self._get_state(), torch.tensor(reward), False, truncated, info

    def reset(self, seed=None, options={}):
        """Initialize the neural network, data loaders, etc. for given/random next task. Also perform a single
        forward/backward pass, not yet updating the neural network parameters."""
        super(SGDEnv, self).reset_(seed)

        self.learning_rate = self.initial_learning_rate 
        self.optimizer_type = torch.optim.AdamW
        self.info = {}

        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            **self.optimizer_params, params=self.model.parameters()
        )

        self.loss = test(**self.train_args, batch_size=self.batch_size).mean()
        self.test_loss = test(**self.test_args).mean()
        self.validation_loss = test(**self.val_args, batch_percentage=0.1).mean()
        self.min_validation_loss = self.validation_loss

        self.n_steps = 0
        self._done = False

        return self._get_state(), {}

    def render(self, mode="human"):
        if mode == "human":
            epoch = 1 + self.n_steps // len(self.train_loader)
            epoch_cutoff = self.cutoff // len(self.train_loader)
            batch = 1 + self.n_steps % len(self.train_loader)
            print(
                f"prev_lr {self.optimizer.param_groups[0]['lr'] if self.n_steps > 0 else None}, "
                f"epoch {epoch}/{epoch_cutoff}, "
                f"batch {batch}/{len(self.train_loader)}, "
                f"batch_loss {self.loss.mean()}, "
                f"val_loss {self.validation_loss}"
            )
        else:
            raise NotImplementedError

    def _get_state(self, crashed: bool = False):
        if self.state_version == "basic":
            state = torch.tensor([self.n_steps, math.log10(self.learning_rate), self.loss, self.validation_loss, self._done])
            if crashed:
                # set both losses to -crash_penalty
                state[2:3] = -self.crash_penalty
        else:
            raise NotImplementedError(f"SGD Benchmark does not support state version {self.state_version}")

        return state