import argparse
import time
from typing import Literal

from src.utils.general import get_config_space
from src.utils.train_agent import train_agent as train_offline
from src.utils.train_agent_online import train_agent as train_online
from train_hpo import Optimizee
from pathlib import Path
from tap import Tap

if __name__ == "__main__":
    class TrainParser(Tap):
        data_dir: Path # Path to the ReplayBuffer
        agent_type: str = "td3_bc"
        agent_config: dict = {}
        num_train_iter: int = 30000
        num_eval_runs: int = 1000
        batch_size: int = 256
        val_freq: int = 10000
        seed: int = 0
        wandb_group: str = None
        timeout: int = 0 # Timeout in sec. 0 -> no timeout
        debug: bool = False # Run for max. 5 iterations and don't log in wanbd.7
        wandb: bool = False
        eval_protocol: Literal["train", "interpolation"] = "train"
        tanh_scaling: bool = False
        eval_seed: int = 123
        cs_type: str = "reduced_no_arch_dropout"

    args = TrainParser().parse_args()
    start = time.time()

    cs = Optimizee(
        args.data_dir,
        agent_type=args.agent_type,
        debug=args.debug,
        budget=None,
        eval_protocol=args.eval_protocol,
        eval_seed=args.eval_seed,
        tanh_scaling=args.tanh_scaling,
    )
    hyperparameters = get_config_space(args.cs_type).get_default_configuration()

    if args.agent_type == "td3":
        train_agent = train_online
    else:
        train_agent = train_offline

    # data = Path(args.data_dir) / "results" / args.agent_type / str(args.seed) / str(30000) / "eval_data.csv"
    # if args.agent_type in ["lb_sac", "sac_n", "edac"] and data.exists():
    #         print(f"Found at folder at {data}")
    # else:
    train_agent(
        data_dir=args.data_dir,
        agent_type=args.agent_type,
        agent_config=args.agent_config,
        num_train_iter=args.num_train_iter,
        num_eval_runs=args.num_eval_runs,
        batch_size=args.batch_size,
        val_freq=args.val_freq,
        seed=args.seed,
        wandb_group=args.wandb_group,
        timeout=args.timeout,
        debug=args.debug,
        use_wandb=args.wandb,
        hyperparameters=hyperparameters,
        eval_protocol=args.eval_protocol,
        eval_seed=args.eval_seed,
        tanh_scaling=args.tanh_scaling
    )

    end = time.time()
    print(f"Took: {end-start}s to generate")
