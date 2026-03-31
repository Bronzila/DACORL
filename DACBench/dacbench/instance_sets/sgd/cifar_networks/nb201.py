from __future__ import annotations

import json
from pathlib import Path

from nas_201_api import NASBench201API as API


def main(result_dir: Path, min_acc: float, sort_by: str, is_descending: bool):
    # Load the benchmark dataset
    benchmark_file = "./NAS-Bench-201-v1_1-096897.pth"
    assert Path(
        benchmark_file
    ).exists(), "Download benchmark file here: https://github.com/D-X-Y/NAS-Bench-201"
    api = API(benchmark_file, verbose=False)

    selected_architectures = []

    # Iterate over all architectures in NAS-Bench-201
    for i in range(len(api)):
        cifar10_metrics = api.get_more_info(
            i, "cifar10", hp="200"
        )  # Get accuracy, loss, etc.

        cost_info = api.get_cost_info(
            i, dataset="cifar10"
        )  # Fetch additional cost info like FLOPs and params
        flops = cost_info["flops"]
        params = cost_info["params"]

        if cifar10_metrics["test-accuracy"] >= min_acc:
            selected_architectures.append(
                {
                    "index": i,
                    "accuracy": cifar10_metrics["test-accuracy"],
                    "flops": flops,
                    "params": params,
                }
            )

    # Sort selected architectures
    selected_architectures = sorted(
        selected_architectures, key=lambda x: x[sort_by], reverse=is_descending
    )

    for i, arch in enumerate(selected_architectures[:20]):
        if not result_dir.exists():
            result_dir.mkdir()
        with open(result_dir / f"{i}.txt", "w") as f:
            config = api.get_net_config(arch["index"], "cifar10")
            arch.update({"config": config})
            json.dump(arch, f)
    # network = get_cell_based_tiny_net(config)
    # print(network(torch.randn(64, 3, 32, 32)))


if __name__ == "__main__":
    # small: min_acc = 80.0, sorted flops ascending
    # medium: min_acc = 92.0, sorted flops ascending
    # large: min_acc = 94.0, sorted acc descending
    min_acc = 94.0
    sort_by = "accuracy"
    is_descending = True
    folder_name = Path("large")
    main(folder_name, min_acc, sort_by, is_descending)
