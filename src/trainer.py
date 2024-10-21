from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import wandb

from src.utils.calculate_sgd_statistic import calc_mean_and_std_dev
from src.utils.general import get_agent, get_environment, save_agent, set_seeds
from src.utils.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    import pandas as pd

    from src.evaluator import Evaluator

cma_es_function = {
    12: "BentCigar",
    11: "Discus",
    2: "Ellipsoid",
    23: "Katsuura",
    15: "Rastrigin",
    8: "Rosenbrock",
    17: "Schaffers",
    20: "Schwefel",
    1: "Sphere",
    16: "Weierstrass",
}


class Trainer:
    def __init__(
        self,
        data_dir: Path,
        agent_config: dict,
        agent_type: str,
        evaluator: Evaluator,
        seed: int,
        device: str = "cpu",
        wandb_group: str = "",
    ) -> None:
        self.data_dir = data_dir
        self.agent_config = agent_config
        self.batch_size = self.agent_config.get("batch_size", 256)
        self.agent_type = agent_type
        self.evaluator = evaluator
        self.seed = seed
        self.device = device
        self.wandb_group = wandb_group
        self.use_wandb = wandb_group != ""
        self.inc_value = np.inf
        self.rng = np.random.default_rng(self.seed)
        set_seeds(self.seed)

        self.results_dir = data_dir / "results" / agent_type / str(self.seed)

        with (data_dir / "run_info.json").open(mode="rb") as f:
            self.run_info = json.load(f)
        self.env_type = self.run_info["environment"]["type"]

        if self.agent_type != "td3":
            self.replay_buffer = ReplayBuffer.load(Path(data_dir, "rep_buffer"))
            self.replay_buffer.seed(seed)
            state, _, _, _, _ = self.replay_buffer.sample(1)
            state_dim = state.shape[1]
            self._setup_agent(state_dim)

        if self.use_wandb:
            fct = self.run_info["environment"]["function"]
            teacher = self.run_info["agent"]["type"]
            state_version = self.run_info["environment"]["state_version"]
            wandb.init(  # type: ignore
                project="DAC4DL",
                entity="study_project",
                group=wandb_group,
                config=agent_config,
                name=f"{agent_type}-{teacher}-{fct}-{state_version}",
                tags=["agent_test", f"{agent_type}", f"{teacher}", f"{fct}"],
            )

    def _setup_agent(self, state_dim: int) -> None:
        """Sets up agent according to given agent config and state dim.

        Args:
            state_dim (int): State dimensionality
        """
        self.agent_config.update(
            {
                "state_dim": state_dim,
                "action_dim": 1,
                "max_action": 0 if self.env_type != "CMAES" else 10,
                "min_action": -10 if self.env_type != "CMAES" else 0,
            },
        )

        self.agent = get_agent(
            self.agent_type,
            self.agent_config,
            self.device,
            cmaes=self.env_type == "CMAES",
        )

    def _eval_agent(self) -> pd.DataFrame:
        """Evaluates the trained agent.

        Returns:
            pd.DataFrame: Aggregated evaluation data
        """
        with torch.random.fork_rng():
            return self.evaluator.evaluate(self.agent.actor)

    def _update_inc_and_log_performance(
        self,
        eval_data: pd.DataFrame,
        t: int,
    ) -> None:
        """Helper function for incumbent and performance logging.

        Args:
            eval_data (pd.DataFrame): Evaluation data at t
            t (int): Current time step
        """
        if self.run_info["environment"]["type"] == "ToySGD":
            fbest_mean, _ = calc_mean_and_std_dev(eval_data, "f_cur")
            print(f"Mean at iteration {t+1}: {fbest_mean}")
            self.inc_value = (
                fbest_mean
                if self.inc_value is None
                else np.min([self.inc_value, fbest_mean])
            )
        elif self.run_info["environment"]["type"] == "SGD":
            # Statistics for train set
            train_loss_mean, train_loss_std = calc_mean_and_std_dev(
                eval_data,
                "train_loss",
            )
            print(
                f"Mean train loss at iteration {t+1}: {train_loss_mean} +- {train_loss_std}",
            )
            train_acc_mean, train_acc_std = calc_mean_and_std_dev(
                eval_data,
                "train_acc",
            )
            print(
                f"Mean train acc at iteration {t+1}: {train_acc_mean} +- {train_acc_std}",
            )

            # Statistics for validation set
            valid_loss_mean, valid_loss_std = calc_mean_and_std_dev(
                eval_data,
                "valid_loss",
            )
            print(
                f"Mean valid loss at iteration {t+1}: {valid_loss_mean} +- {valid_loss_std}",
            )
            valid_acc_mean, valid_acc_std = calc_mean_and_std_dev(
                eval_data,
                "valid_acc",
            )
            print(
                f"Mean valid acc at iteration {t+1}: {valid_acc_mean} +- {valid_acc_std}",
            )

            # Statistics for test set
            test_loss_mean, test_loss_std = calc_mean_and_std_dev(
                eval_data,
                "test_loss",
            )
            print(
                f"Mean test loss at iteration {t+1}: {test_loss_mean} +- {test_loss_std}",
            )
            test_acc_mean, test_acc_std = calc_mean_and_std_dev(
                eval_data,
                "test_acc",
            )
            print(
                f"Mean test acc at iteration {t+1}: {test_acc_mean} +- {test_acc_std}",
            )

            self.inc_value = (
                test_acc_mean
                if self.inc_value is None
                else np.max([self.inc_value, test_acc_mean])
            )
        elif self.env_type == "CMAES":
            final_evaluations = eval_data.groupby("run").last()
            function_group = final_evaluations.groupby("function_id")
            for function in function_group:
                fbests = function[1]["f_cur"].mean()
                target_value = function[1]["target_value"].mean()
                fid = int(function[1]["function_id"].mean())
                print(
                    f"Mean at iteration {t+1}: {fbests} - {target_value} at function {cma_es_function[fid]}",
                )
            fbest_mean = final_evaluations["f_cur"].mean()

    def _setup_environment(self) -> tuple[Any, np.ndarray]:
        """Sets up environment according to environment config.

        Returns:
            tuple[Any, np.ndarray]: Environment and initial state
        """
        env = get_environment(self.run_info["environment"].copy())
        initial_state = env.reset()[0]

        batches_per_epoch = 1
        if self.env_type == "SGD":
            if env.epoch_mode is False:
                # if SGD env, translates num_batches to num_epochs
                batches_per_epoch = len(env.train_loader)
                print(f"One epoch consists of {batches_per_epoch} batches.")
            else:
                print("Currently running in epoch mode.")

        return env, initial_state

    def _train_offline(
        self,
        num_train_iter: int,
        val_freq: int,
    ) -> tuple[dict, float]:
        """Trains agent using offline RL.

        Args:
            num_train_iter (int): Number of training iterations
            val_freq (int): Validation frequency

        Returns:
            tuple[dict, float]: Logs and incumbent performance
        """
        logs: dict = {}

        print("Starting training with the following configuration...")
        print(f"HP config: {self.agent_config}")
        for t in range(int(num_train_iter)):
            batch = self.replay_buffer.sample(self.batch_size)
            log_dict = self.agent.train(batch)
            for k, v in log_dict.items():
                if k not in logs:
                    logs[k] = [v]
                else:
                    logs[k].append(v)

            if self.use_wandb:
                wandb.log(log_dict, self.agent.total_it)  # type: ignore

            if val_freq != 0 and (t + 1) % val_freq == 0:
                eval_data = self._eval_agent()

                # Save agent early to enable continuation of pipeline
                save_agent(self.agent.state_dict(), self.results_dir, t)
                eval_data.to_csv(
                    self.results_dir / f"{t + 1}" / "eval_data.csv",
                )
                with (self.results_dir / f"{t + 1}" / "config.json").open(
                    "w",
                ) as f:
                    json.dump(dict(self.agent_config), f, indent=2)

                # Calculate mean performance for this checkpoint
                self._update_inc_and_log_performance(eval_data, t)

                if np.isnan(self.inc_value):
                    print(dict(self.agent_config))

        save_agent(self.agent.state_dict(), self.results_dir, t)

        if self.use_wandb:
            wandb.finish()  # type: ignore

        return logs, self.inc_value

    def _train_online(
        self,
        num_train_iter: int,
        val_freq: int,
        start_timesteps: int = 2560,
    ) -> tuple[dict, float]:
        """Trains agent using online RL.

        Args:
            num_train_iter (int): Number of training iterations
            val_freq (int): Validation frequency
            start_timesteps (int, optional): Defines how many random actions to use in
                order to fill ReplayBuffer. Defaults to 2560.

        Returns:
            tuple[dict, float]: Logs and incumbent value
        """
        env, state = self._setup_environment()
        # This is currently a quite ugly work-around for online-training
        state_dim = state.shape[0]
        self._setup_agent(state_dim)
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=1,
            buffer_size=num_train_iter,
            seed=self.seed,
        )

        min_action = self.agent_config["min_action"]
        max_action = self.agent_config["max_action"]
        action_dim = self.agent_config["action_dim"]

        logs: dict = {}

        print("Starting training with the following configuration...")
        print(f"HP config: {self.agent_config}")
        for t in range(int(num_train_iter)):
            # Select action randomly or according to policy
            if t < start_timesteps:
                action = self.rng.uniform(min_action, max_action, 1).astype(
                    np.float32,
                )
            else:
                print("First Time")
                action = (
                    self.agent.select_action(state.type(torch.float32))
                    + np.random.normal(  # noqa: NPY002
                        0,
                        max_action * 0.1,
                        size=action_dim,
                    ).astype(np.float32)
                ).clip(min_action, max_action)

            if self.env_type == "CMAES":
                action += 1e-10
            else:
                action = torch.from_numpy(action)

            # Perform action
            next_state, reward, done, _, _ = env.step(action.item())

            # Store data in replay buffer
            self.replay_buffer.add_transition(
                state,
                action,
                next_state,
                reward,
                done,
            )

            state = next_state

            # Train agent after collecting sufficient data
            if t >= start_timesteps:
                batch = self.replay_buffer.sample(
                    self.agent_config["batch_size"],
                )
                log_dict = self.agent.train(batch)
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = [v]
                    else:
                        logs[k].append(v)

                if self.use_wandb:
                    wandb.log(log_dict, self.agent.total_it)  # type: ignore

            # if we run out of bounds or reached max optimization iters
            if done:
                # Reset environment
                (state, _), done = env.reset(), False

            # Evaluate episode
            if val_freq != 0 and (t + 1) % val_freq == 0:
                eval_data = self._eval_agent()
                # Save agent early to enable continuation of pipeline
                save_agent(self.agent.state_dict(), self.results_dir, t)
                eval_data.to_csv(
                    self.results_dir / f"{t + 1}" / "eval_data.csv",
                )
                with (self.results_dir / f"{t + 1}" / "config.json").open(
                    "w",
                ) as f:
                    json.dump(dict(self.agent_config), f, indent=2)

                # Calculate mean performance for this checkpoint
                self._update_inc_and_log_performance(eval_data, t)

        save_agent(self.agent.state_dict(), self.results_dir, t)

        if self.use_wandb:
            wandb.finish()  # type: ignore

        return logs, self.inc_value

    def train(
        self,
        num_train_iter: int,
        val_freq: int,
        start_timesteps: int = 2560,
    ) -> tuple[dict, float]:
        """Train agent online/offline depending on trainer configuration.

        Args:
            num_train_iter (int): Nummber of training iterations
            val_freq (int): Validation frequency
            start_timesteps (int, optional): Only used for online training:
                Defines how many random actions to use in order to fill ReplayBuffer. Defaults to 2560.

        Returns:
            tuple[dict, float]: Logs and incumbent value
        """
        if self.agent_type == "td3":
            return self._train_online(
                num_train_iter=num_train_iter,
                val_freq=val_freq,
                start_timesteps=start_timesteps,
            )

        return self._train_offline(
            num_train_iter=num_train_iter,
            val_freq=val_freq,
        )
