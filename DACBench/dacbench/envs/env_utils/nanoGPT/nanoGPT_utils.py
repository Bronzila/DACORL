from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Dataset


class OpenWebTextDataset(Dataset):
    """Define custom class for the OpenWebText dataset.
    Due to the large dataset size,
    we need to think of how to store this dataset locally and load it.
    """

    def __init__(
        self,
        dataset_path: Path,
        train: bool,
        block_size: int = 1024,
        device: torch.device = "cpu",
        **kwargs,
    ):
        """Dataset class for the OpenWebText dataset.

        Args:
            dataset_path (Path): Directory containing of the binary files.
            train (bool): If we look at the test or train/valid set.
            block_size (int): Size used to format the binary file.
            device (device): Device used to train the data. Relevant for DDP.
        """
        # print(kwargs)  # Commented out - causes error with ToTensor transform
        # Ensure dataset_path is a Path object
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        assert (
            dataset_path
        ).exists(), (
            "The dataset was not found. Please run env_utils/nanoGPT/prepare_dataset.py"
        )
        mode = "train" if train else "test"
        self.block_size = block_size
        self.device = device

        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        self.data = np.memmap(dataset_path / f"{mode}.bin", dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Handle both single index and batch of indices
        if isinstance(index, int):
            # Single index case (used by DataLoader with Subset)
            x = torch.from_numpy((self.data[index : index + self.block_size]).astype(np.int64))
            y = torch.from_numpy((self.data[index + 1 : index + 1 + self.block_size]).astype(np.int64))
        else:
            # Batch of indices case (original behavior)
            x = torch.stack(
                [
                    torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64))
                    for i in index
                ]
            )
            y = torch.stack(
                [
                    torch.from_numpy(
                        (self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                    )
                    for i in index
                ]
            )
        if "cuda" in self.device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


def nanoGPT_data_loader(
    seed: int,
    dataset_path: str | Path,
    dataset_name: str,
    batch_size: int,
    fraction_of_dataset: float,
    train_validation_ratio: float,
    block_size: int = 1024,
    device: str = "cuda",
):
    """Create data loaders for nanoGPT datasets (OpenWebText, Shakespeare) without loading entire dataset into memory.
    
    Args:
        seed: Random seed
        dataset_path: Path to dataset directory
        dataset_name: Name of dataset (e.g., "OpenWebText" or "Shakespeare")
        batch_size: Batch size for data loaders
        fraction_of_dataset: Fraction of dataset to use (0-1)
        train_validation_ratio: Ratio of train to validation split
        block_size: Context length for GPT
        device: Device to load data to
        
    Returns:
        tuple: (datasets_dict, (train_loader, val_loader, test_loader))
    """
    from torch.utils.data import DataLoader, Subset
    import random
    
    # Ensure dataset_path is a Path object
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    
    # Append dataset name to path (e.g., datasets/shakespeare)
    full_dataset_path = dataset_path / dataset_name.lower()
    
    # Create datasets - these use memory mapping, so they don't load everything into RAM
    train_val_dataset = OpenWebTextDataset(
        full_dataset_path, train=True, block_size=block_size, device=device
    )
    test_dataset = OpenWebTextDataset(
        full_dataset_path, train=False, block_size=block_size, device=device
    )
    
    # Calculate subset sizes
    total_size = len(train_val_dataset)
    
    # Adjust max size based on dataset
    # OpenWebText is HUGE (~9 billion tokens), Shakespeare is small (~300K tokens)
    if dataset_name == "Shakespeare":
        # Shakespeare is small enough to use entirely
        max_reasonable_size = total_size
    else:
        # For OpenWebText, cap at 10 million samples to avoid memory issues
        max_reasonable_size = 10_000_000
    
    use_size = min(int(total_size * fraction_of_dataset), max_reasonable_size)
    
    # Generate random subset indices using numpy (more memory efficient)
    random.seed(seed)
    np.random.seed(seed)
    # Sample indices evenly across the dataset
    if use_size < total_size:
        # Sample evenly across the dataset rather than from the beginning
        subset_indices = np.linspace(0, total_size - 1, use_size, dtype=np.int64).tolist()
        # Shuffle the sampled indices
        random.shuffle(subset_indices)
    else:
        subset_indices = list(range(use_size))
    
    # Split into train and validation
    train_size = int(len(subset_indices) * train_validation_ratio)
    train_size = train_size - train_size % batch_size  # Ensure divisible by batch size
    
    train_indices = subset_indices[:train_size]
    val_indices = subset_indices[train_size:]
    
    # Create subsets (still using memory mapping)
    train_subset = Subset(train_val_dataset, train_indices)
    val_subset = Subset(train_val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    
    # Return in format expected by environment
    datasets = {
        "train": train_subset,
        "validation": val_subset,
        "test": test_dataset,
    }
    
    return datasets, (train_loader, val_loader, test_loader)