from __future__ import annotations

from pathlib import Path

import pandas as pd


def inspect_metadata_file(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)

    label_cols = [c for c in df.columns if c.startswith("class_")]

    all_zero_rows = int(df[label_cols].fillna(0.0).eq(0.0).all(axis=1).sum()) if label_cols else 0

    # Data are encoded as percentages in [0, 100], often with intermediate values (15, 50, 80).
    # For a binary multi-label target required by BCEWithLogitsLoss, we normalize by 100 and
    # binarize with >0 -> 1.0, ==0 -> 0.0.
    detected_dat_col = next((c for c in df.columns if c.endswith("_dat_file")), None)
    detected_hea_col = next((c for c in df.columns if c.endswith("_hea_file")), None)
    detected_base_col = "local_record_base" if "local_record_base" in df.columns else None

    return {
        "label_columns": label_cols,
        "file_columns": {
            "base": detected_base_col,
            "dat": detected_dat_col,
            "hea": detected_hea_col,
        },
        "all_zero_rows": all_zero_rows,
    }


def inspect_all_metadata(data_root: Path) -> dict[str, object]:
    train_info = inspect_metadata_file(data_root / "train" / "train_metadata.csv")
    val_info = inspect_metadata_file(data_root / "val" / "val_metadata.csv")
    test_info = inspect_metadata_file(data_root / "test" / "test_metadata.csv")

    if train_info["label_columns"] != val_info["label_columns"] or train_info["label_columns"] != test_info["label_columns"]:
        raise ValueError("Label columns are not consistent across train/val/test metadata files.")

    return {
        "label_columns": train_info["label_columns"],
        "file_columns": train_info["file_columns"],
        "splits": {
            "train": train_info,
            "val": val_info,
            "test": test_info,
        },
    }




