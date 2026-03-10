"""
PyTorch Dataset i DataLoader dla PTB-XL.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb


class PTBXLDataset(Dataset):
    """
    Dataset PTB-XL z obsługą wieloetykietowej klasyfikacji 8 klas.

    Sygnał: (12, 5000) — 12 odprowadzeń, 10s @ 500Hz
    Etykieta: (8,) — wektor binarny
    """

    def __init__(self, df, data_dir, mean=None, std=None, augment=False):
        """
        Args:
            df: DataFrame z kolumnami 'filename', 'labels', 'strat_fold'
            data_dir: ścieżka bazowa do datasetu PTB-XL
            mean: per-lead mean for normalization, shape (12,)
            std: per-lead std for normalization, shape (12,)
            augment: whether to apply data augmentation
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = os.path.join(self.data_dir, row["filename"])

        # Load signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal.astype(np.float32)  # (5000, 12)

        # Handle missing values
        signal = np.nan_to_num(signal, nan=0.0)

        # Normalize
        if self.mean is not None and self.std is not None:
            signal = (signal - self.mean) / self.std

        # Augmentation
        if self.augment:
            signal = self._augment(signal)

        # Transpose to channels-first: (12, 5000)
        signal = signal.T

        label = row["labels"]

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def _augment(self, signal):
        """Simple augmentations for training."""
        # Random temporal shift (up to 50 samples)
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 51)
            signal = np.roll(signal, shift, axis=0)

        # Add Gaussian noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, signal.shape).astype(np.float32)
            signal = signal + noise

        return signal


def get_dataloaders(df, data_dir, mean, std, batch_size=64, num_workers=0):
    """
    Create train/val/test DataLoaders using PTB-XL strat_fold.

    Train: folds 1-8, Val: fold 9, Test: fold 10
    """
    train_df = df[df["strat_fold"].isin(range(1, 9))]
    val_df = df[df["strat_fold"] == 9]
    test_df = df[df["strat_fold"] == 10]

    train_ds = PTBXLDataset(train_df, data_dir, mean, std, augment=True)
    val_ds = PTBXLDataset(val_df, data_dir, mean, std, augment=False)
    test_ds = PTBXLDataset(test_df, data_dir, mean, std, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Dane: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader
