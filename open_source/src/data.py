from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class ArrayDataset(Dataset):
    """Generic dataset for [N, C, T] arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "Mismatched samples between X and y"
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def load_single_branch(data_root: Path, transpose: bool) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(data_root / "X.npy", mmap_mode="r")
    y = np.load(data_root / "y.npy")
    if transpose:
        X = np.asarray(X).transpose(0, 2, 1)
    return X, y


def load_dual_branch(data_root: Path, transpose: bool):
    X_emg = np.load(data_root / "X_emg.npy", mmap_mode="r")
    X_imu = np.load(data_root / "X_imu.npy", mmap_mode="r")
    y_actions = np.load(data_root / "y_actions.npy")
    y_gyro = None
    gyro_path = data_root / "y_gyro_bins.npy"
    if gyro_path.exists():
        y_gyro = np.load(gyro_path)
    if transpose:
        X_emg = np.asarray(X_emg).transpose(0, 2, 1)
        X_imu = np.asarray(X_imu).transpose(0, 2, 1)
    return X_emg, X_imu, y_actions, y_gyro


def split_dataset(dataset: Dataset, val_ratio: float, seed: int):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
