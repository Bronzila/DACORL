"""Benchmark for SGD."""

from __future__ import annotations

import ConfigSpace as CS  # noqa: N817
import numpy as np
import torch
from gymnasium import spaces

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import LayerwiseNanoGPTEnv

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
LR = CS.Float(name="learning_rate", bounds=(0.0, 0.05))
DEFAULT_CFG_SPACE.add_hyperparameter(LR)

# TODO Update info
INFO = {
    "identifier": "LR",
    "name": "Layerwise Learning Rate Adaption for Neural Networks",
    "reward": "Negative Log Differential Validation Loss",
    "state_description": [
        "Step",
        "Current Learning Rate",
        "Loss",
        "Validation Loss",
        "Crashed",
        # "Predictive Change Variance (Discounted Average)",
        # "Predictive Change Variance (Uncertainty)",
        # "Loss Variance (Discounted Average)",
        # "Loss Variance (Uncertainty)",
        # "Training Loss",
        # "Alignment",
    ],
    "action_description": ["Learning Rate"],
}


LAYERWISE_NANOGPT_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "step": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "learning_rate": spaces.Box(low=0, high=1, shape=(1,)),
                "loss": spaces.Box(0, np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "crashed": spaces.Discrete(1),
            }
        ],
        "reward_range": [-(10**9), (10**9)],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "optimizer_params": {
            "momentum": 0.9,
            "lr": 0.002,
        },
        "fraction_of_dataset": 1.0,  # How much of the train/val set should be used
        "train_validation_ratio": 0.8,  # Percentage of train to validation
        "dataset_name": "OpenWebText",  # Currently only OpenWebText is supported;
        # "reward_function":,    # Can be set, to replace the default function
        # "state_method":,       # Can be set, to replace the default function
        "seed": 0,
        "crash_penalty": -10000.0,
        "initial_learning_rate": 0.002,
        "dataset_path": "./datasets/openwebtext/",
        "benchmark_info": INFO,
        "epoch_mode": False,
        "instance_mode": "random_seed",
        # nanoGPT
        # We used the file "https://github.com/karpathy/nanoGPT/blob/master/train.py"
        # for reference and thereupon created the environment as well.
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.0,
        "bias": False,
        "training_batch_size": 12,
        "block_size": 1024,
        "iters_per_epoch": 10,  # training iterations before adapting the lr
        "eval_interval": 400,  # when to evaluate test and validation on 100%
        "grad_acc_steps": 5 * 8,  # Gradient Accumulation Steps
        "backend": "nccl",  # DDP backend; 'nccl', 'gloo', etc.
    }
)


class LayerwiseNanoGPTBenchmark(AbstractBenchmark):
    """Benchmark with default configuration & relevant functions for SGD."""

    def __init__(self, config_path=None, config=None):
        """Initialize layerwise nanoGPT Benchmark.

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path, config)
        if not self.config:
            self.config = objdict(LAYERWISE_NANOGPT_DEFAULTS.copy())

        # Probably won't need this. Keeping it if necessary.
        for key in LAYERWISE_NANOGPT_DEFAULTS:
            if (
                key not in self.config
                or key == "instance_mode"
                and self.config[key] == ""
            ):
                self.config[key] = LAYERWISE_NANOGPT_DEFAULTS[key]

    def get_environment(self):
        """Return LayerwiseNanoGPTEnv env with current configuration.

        Returns:
        -------
        LayerwiseNanoGPTEnv (AbstractMADACEnv)
        """
        # We most likely will not need instance sets, since the task of nanoGPT
        # will be run in "instance_mode: random_seed". If this should be expanded
        # someday, please look at LayerwiseSGDEnv for reference

        env = LayerwiseNanoGPTEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        print(f"Running on device: {self.config['device']}")

        return env
