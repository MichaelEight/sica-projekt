from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import infer_file_columns, infer_label_columns, missing_canonical_columns


def inspect_metadata_file(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)

    label_cols = infer_label_columns(df.columns)
    if not label_cols:
        raise ValueError(f"No class columns found in metadata file: {csv_path}")

    all_zero_rows = int(df[label_cols].fillna(0.0).eq(0.0).all(axis=1).sum()) if label_cols else 0

    # Metadane po `filter_data.py` zachowują procentowe etykiety w zakresie [0, 100].
    # W treningu są one normalizowane do [0, 1] i traktowane jako soft targets.
    file_columns = infer_file_columns(df.columns)

    return {
        "label_columns": label_cols,
        "file_columns": file_columns,
        "all_zero_rows": all_zero_rows,
        "missing_canonical_class_columns": missing_canonical_columns(df.columns),
    }


def inspect_all_metadata(data_root: Path) -> dict[str, object]:
    train_info = inspect_metadata_file(data_root / "train" / "train_metadata.csv")
    val_info = inspect_metadata_file(data_root / "val" / "val_metadata.csv")
    test_info = inspect_metadata_file(data_root / "test" / "test_metadata.csv")

    if train_info["label_columns"] != val_info["label_columns"] or train_info["label_columns"] != test_info["label_columns"]:
        raise ValueError("Label columns are not consistent across train/val/test metadata files.")

    if train_info["file_columns"] != val_info["file_columns"] or train_info["file_columns"] != test_info["file_columns"]:
        raise ValueError("File columns are not consistent across train/val/test metadata files.")

    return {
        "label_columns": train_info["label_columns"],
        "file_columns": train_info["file_columns"],
        "splits": {
            "train": train_info,
            "val": val_info,
            "test": test_info,
        },
    }




