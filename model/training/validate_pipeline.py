from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from model.models import Inception1DNet
from model.training.dataset import ECGWFDBDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "training"


def main() -> None:
    meta_path = DATA_ROOT / "train" / "train_metadata.csv"
    meta = pd.read_csv(meta_path)
    label_columns = [c for c in meta.columns if c.startswith("class_")]
    file_columns: dict[str, str | None] = {
        "base": "local_record_base" if "local_record_base" in meta.columns else None,
        "dat": next((c for c in meta.columns if c.endswith("_dat_file")), None),
        "hea": next((c for c in meta.columns if c.endswith("_hea_file")), None),
    }

    ds = ECGWFDBDataset(
        split_dir=DATA_ROOT / "train",
        metadata_filename="train_metadata.csv",
        label_columns=label_columns,
        file_columns=file_columns,
        target_length=5000,
    )

    print("Dataset size after filtering:", len(ds))

    x, y = ds[0]
    print("Sample x shape:", tuple(x.shape), "dtype:", x.dtype)
    print("Sample y shape:", tuple(y.shape), "dtype:", y.dtype)

    model = Inception1DNet(num_classes=len(label_columns))
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs = model.forward_inference(x.unsqueeze(0))

    print("Logits shape:", tuple(logits.shape))
    print("Probabilities shape:", tuple(probs.shape))


if __name__ == "__main__":
    main()
