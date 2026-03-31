"""SGD environment."""

from __future__ import annotations

import math
import os
import pickle
import random
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from dacbench.envs.env_utils.sgd_utils import random_torchvision_loader
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

from dacbench import AbstractMADACEnv
from dacbench.envs.env_utils.nanoGPT import GPT, GPTConfig


def set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _optimizer_actions(optimizer: torch.optim.Optimizer, actions: list[float]) -> None:
    for g, lr in zip(optimizer.param_groups, actions, strict=True):
        g["lr"] = lr


def _get_layer_encoding(layer_type: str) -> int:
    if layer_type == "Linear":
        return 0
    if layer_type == "Conv2d":
        return 1
    raise ValueError("Unkown layer type.")


class LayerwiseNanoGPTEnv(AbstractMADACEnv):
    """The nanoGPT DAC Environment implements the problem of dynamically configuring the
    learning rate hyperparameter of an optimizer (more specifically, torch.optim.SGD)
    for training the nanoGPT model.
    During training, the model is evaluated after every epoch.

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
        # Note: torch.use_deterministic_algorithms(True) requires CUBLAS_WORKSPACE_CONFIG env var
        # For HPO, we rely on manual seed setting instead for reproducibility
        self.epoch_mode = config.get("epoch_mode", True)
        self.instance_set_path = config.get("instance_set_path")
        self.dataset_path = config.get("dataset_path")
        self.dataset_name = config.get("dataset_name")

        # nanoGPT
        # model
        self.n_layer = config.get("n_layer")
        self.n_head = config.get("n_head")
        self.n_embd = config.get("n_embd")
        self.block_size = config.get("block_size")
        self.bias = config.get("bias")
        self.dropout = config.get("dropout")
        self.compile_model = config.get("compile", False)  # Requires PyTorch >= 2.0

        meta_path = Path(self.dataset_path, "meta.pkl")
        self.meta_vocab_size = None
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta["vocab_size"]

        # training
        # train_iters between adapting the lr
        self.iters_per_epoch = config.get("iters_per_epoch")
        self.eval_interval = config.get("eval_interval")
        self.eval_iters = config.get("eval_iters", 10)  # Number of batches to evaluate for loss estimation
        # data
        self.batch_size = config.get("training_batch_size")  # Moved here to be available earlier
        self.gradient_accumulation_steps = config.get("grad_acc_steps")
        # DDP
        backend = config.get("backend")
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = (
                self.ddp_rank == 0
            )  # this process will do logging, checkpointing etc.
            self.seed_offset = self.ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.gradient_accumulation_steps % self.ddp_world_size == 0
            self.gradient_accumulation_steps //= self.ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        self.tokens_per_iter = (
            self.gradient_accumulation_steps
            * self.ddp_world_size
            * self.batch_size
            * self.block_size
        )
        print(f"tokens per iteration will be: {self.tokens_per_iter:,}")

        self.device = config.get("device")
        self.pdtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type=self.device, dtype=self.pdtype)
        )
        self.grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
        self.log_interval = 1  # clip gradients at this value, or disable if == 0.0

        self.optimizer_type = torch.optim.SGD
        self.optimizer_params = config.get("optimizer_params")
        # self.batch_size already set above
        self.crash_penalty = config.get("crash_penalty")

        # Use default reward function, if no specific function is given
        self.get_reward = config.get("reward_function", self.get_default_reward)

        # Use default state function, if no specific function is given
        self.get_states = config.get("state_method", self.get_default_states)

        self.initial_learning_rate = config.get("initial_learning_rate")
        self.initial_seed = config.get("seed")
        self.seed(self.initial_seed)

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

        # Handle both single float and list of floats
        if isinstance(actions, (int, float)):
            # Single learning rate for all layers
            log_learning_rates = [actions] * self.n_layers
        else:
            log_learning_rates = actions
            
        self.learning_rates = [
            10**log_learning_rate for log_learning_rate in log_learning_rates
        ]

        # Update action history
        self._update_lr_histories(log_learning_rates)

        _optimizer_actions(self.optimizer, self.learning_rates)

        if self.epoch_mode:
            self.train_loss, self.average_loss = self.run_epoch()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train_loss, self.train_accuracy = self.forward_backward(self.train_loader)

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

        batch_percentage = (
            1.0 if self.c_step % self.eval_interval == 0 or self._done else 0.1
        )
        validation_loss, validation_accuracy = self.evaluate_performance(
            self.validation_loader, batch_percentage
        )
        self.validation_loss = validation_loss.mean()
        self.validation_accuracy = validation_accuracy.mean()

        if (
            self.min_validation_loss is None
            or self.validation_loss <= self.min_validation_loss
        ):
            self.min_validation_loss = self.validation_loss

        # Calculate test loss after every epoch + when done
        if self._done or self.c_step % self.eval_interval == 0:
            test_losses, test_accuracies = self.evaluate_performance(
                self.test_loader, 1.0
            )
            self.test_loss = test_losses.mean()
            self.test_accuracy = test_accuracies.mean()

        reward = self.get_reward()

        return self.get_states(), reward, truncated, truncated, info

    def reset(self, seed=None, options=None):
        """Initialize the neural network, data loaders, etc. for given/random next task.
        Also perform a single forward/backward pass,
        not yet updating the neural network parameters.
        """
        super().reset_(seed=seed, scheme="round_robin")
        if self.ddp:
            destroy_process_group()
        if options is None:
            options = {}

        # Set global seed for data loaders
        run_seed = self.rng.integers(0, 1000000000)
        set_global_seeds(run_seed)
        print(f"run_seed: {run_seed}", flush=True)

        # Use custom nanoGPT data loader (memory-efficient for large datasets)
        from dacbench.envs.env_utils.nanoGPT.nanoGPT_utils import nanoGPT_data_loader
        
        self.datasets, loaders = nanoGPT_data_loader(
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
            block_size=self.block_size,
            device=self.device,
        )
        self.train_loader, self.validation_loader, self.test_loader = loaders

        self.epoch_length = len(self.train_loader)

        model_args = {
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "bias": self.bias,
            "vocab_size": None,
            "dropout": self.dropout,
        }  # start with model_args from command line
        # init a new model from scratch
        # determine the vocab size we'll use for from-scratch training
        # Default vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)
        model_args["vocab_size"] = (
            self.meta_vocab_size if self.meta_vocab_size is not None else 50304
        )
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)

        # TODO: unsure if this is the case for us, since we don't use checkpoints
        # crop down the model block size if desired, using model surgery
        if self.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.block_size)
            model_args[
                "block_size"
            ] = self.block_size  # so that the checkpoint will have the right value
        self.model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.pdtype == torch.float16)
        )

        if self.compile_model:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.optimizer_type = torch.optim.SGD
        self.info = {}
        self._done = False

        # create param groups then feed them into optimizer
        # Get weight_decay with a default if not provided
        weight_decay = self.optimizer_params.get("weight_decay", 0.1) if self.optimizer_params else 0.1
        param_groups, self.layer_types = self._create_param_groups(weight_decay)
        self.optimizer: torch.optim.Optimizer = self.optimizer_type(
            param_groups,
            **self.optimizer_params,
        )

        self.n_layers = len(self.optimizer.param_groups)
        self.learning_rates = [self.initial_learning_rate] * self.n_layers
        self.lr_histories = [
            deque(torch.ones(5) * math.log10(self.initial_learning_rate))
        ] * self.n_layers

        self.local_iter_num = 0
        self.raw_model = (
            self.model.module if self.ddp else self.model
        )  # unwrap DDP container if needed
        self.running_mfu = -1.0

        losses = self.estimate_loss()
        self.train_loss = losses["train"]
        self.validation_loss = losses["val"]  # Use consistent naming
        self.test_loss = losses["test"]
        self.train_accuracy = torch.tensor(0.0)  # Initialize train accuracy
        self.test_accuracy = 0.0  # Initialize test accuracy
        self.validation_accuracy = 0.0  # Initialize validation accuracy
        self.min_validation_loss = None  # Will be set on first validation

        self.X, self.Y = self.get_batch("train")  # fetch the very first batch
        self.t0 = time.time()

        if self.epoch_mode:
            self.average_loss = 0

        return self.get_states(), {}

    def _create_param_groups(self, weight_decay) -> tuple[list[Any], list[Any]]:
        """Create parameter groups for the optimizer.
        
        For GPT models, we iterate through all Linear layers (in attention and MLP blocks).
        Weight decay is applied to parameters with dim >= 2 (weights but not biases).
        """
        param_groups = []
        layer_types = []
        
        # Iterate through all modules to find Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Create a param group for this Linear layer
                # Apply weight decay only to weight tensors (dim=2), not biases (dim=1)
                param_groups.append(
                    {
                        "params": list(module.parameters()),
                        "lr": self.initial_learning_rate,
                        "weight_decay": weight_decay,  # Will be applied per-parameter by PyTorch
                    }
                )
                layer_types.append("Linear")  # Just "Linear" for encoding compatibility
        
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
        for layer_idx, param_group in enumerate(self.optimizer.param_groups):
            local_observations = []
            # Layer encoding
            layer_type = self.layer_types[layer_idx]
            layer_enc = _get_layer_encoding(layer_type)
            local_observations.append(torch.tensor([layer_enc], device=self.device))

            log_learning_rate = (
                np.log10(self.learning_rates[layer_idx])
                if self.learning_rates[layer_idx] != 0
                else np.log10(1e-10)
            )
            local_observations.append(
                torch.tensor([log_learning_rate], device=self.device)
            )
            lr_hist_deltas = log_learning_rate - self.lr_histories[layer_idx]
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

    def forward_backward(
        self, loader: torch.utils.data.DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Do a forward and a backward pass for given `model` for `loss_function`.

        Returns:
            loss: Mini batch training loss per data point
        """
        self.model.train()
        (data, target) = next(iter(loader))
        data, target = data.to(self.device), target.to(self.device)
        logits, loss = self.model(data, target)  # Pass target to compute loss

        # backward pass, with gradient scaling if training in fp16
        self.scaler.scale(loss).backward()
        # clip the gradient
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        # step the optimizer and scaler if training in fp16
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Calculate accuracy: logits shape is (batch, block_size, vocab_size)
        # target shape is (batch, block_size)
        accuracy = torch.sum(logits.argmax(dim=-1) == target) / target.numel()
        return loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy()

    def get_batch(self, split):
        """Get a batch from the specified data split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            tuple: (X, Y) tensors for input and target
        """
        if split == 'train':
            loader = self.train_loader
        elif split == 'val':
            loader = self.validation_loader
        else:
            loader = self.test_loader
        
        X, Y = next(iter(loader))
        X, Y = X.to(self.device), Y.to(self.device)
        return X, Y

    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss on train, val, and test sets.
        
        Returns:
            dict: Dictionary with 'train', 'val', and 'test' loss values
        """
        out = {}
        self.model.eval()
        for split in ['train', 'val', 'test']:
            losses = []
            loader = self.train_loader if split == 'train' else (
                self.validation_loader if split == 'val' else self.test_loader
            )
            # Evaluate on a few batches to get an estimate
            for _ in range(min(self.eval_iters, len(loader))):
                X, Y = next(iter(loader))
                X, Y = X.to(self.device), Y.to(self.device)
                _, loss = self.model(X, Y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses) if losses else float('inf')
        self.model.train()
        return out

    @torch.no_grad()
    def evaluate_performance(
        self,
        loader: torch.utils.data.DataLoader,
        batch_percentage: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate current performance on a given data loader.

        Percentage defines how much percentage of the data shall be used.
        If nothing given the whole data is used.

        Returns:
            losses: Batch validation loss per data point
        """
        nmb_sets = batch_percentage * (len(loader.dataset) / self.batch_size)
        self.model.eval()
        test_losses = []
        test_accuracies = []
        i = 0

        with torch.no_grad():
            for _ in range(self.eval_iters):
                data, target = next(iter(loader))
                d_data, d_target = data.to(self.device), target.to(self.device)
                logits, loss = self.model(d_data, d_target)  # Pass target to compute loss
                test_losses.append(loss.item())
                # Calculate accuracy: logits shape is (batch, block_size, vocab_size)
                accuracy = torch.sum(logits.argmax(dim=-1) == d_target) / d_target.numel()
                test_accuracies.append(accuracy.item())
                i += 1
                if i >= nmb_sets:
                    break
        
        # test_losses and test_accuracies are lists of scalars
        return (
            np.array(test_losses),
            np.array(test_accuracies),
        )

    def run_epoch(self):
        """Run a single epoch of training for given `model` with `loss_function`."""
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(self.gradient_accumulation_steps):
            if self.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.model.require_backward_grad_sync = (
                    micro_step == self.gradient_accumulation_steps - 1
                )
            with self.ctx:
                logits, loss = self.model(self.X, self.Y)
                loss = (
                    loss / self.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
                accuracy = torch.sum(logits.argmax(dim=1) == self.Y) / len(self.Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            self.X, self.Y = next(iter(self.train_loader))
            # backward pass, with gradient scaling if training in fp16
            self.scaler.scale(loss).backward()
        # clip the gradient
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        # step the optimizer and scaler if training in fp16
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        self.optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - self.t0
        self.t0 = t1
        if self.c_step % self.log_interval == 0 and self.master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * self.gradient_accumulation_steps
            if self.local_iter_num >= 5:  # let the training loop settle a bit
                mfu = self.raw_model.estimate_mfu(
                    self.batch_size * self.gradient_accumulation_steps, dt
                )
                self.running_mfu = (
                    mfu
                    if self.running_mfu == -1.0
                    else 0.9 * self.running_mfu + 0.1 * mfu
                )
            print(
                f"iter {self.c_step}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {self.running_mfu*100:.2f}%"
            )
        self.local_iter_num += 1
        return loss.item(), accuracy.item()

    def _update_lr_histories(self, log_learning_rates: list[float]) -> None:
        for lr_history, log_lr in zip(
            self.lr_histories, log_learning_rates, strict=True
        ):
            lr_history.pop()
            lr_history.appendleft(log_lr)

    def seed(self, seed, seed_action_space=False):
        super().seed(seed, seed_action_space)

        self.rng = np.random.default_rng(seed)
