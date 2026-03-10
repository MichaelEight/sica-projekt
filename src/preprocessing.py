"""
Ładowanie metadanych PTB-XL, mapowanie etykiet na 8 klas docelowych,
normalizacja sygnałów.
"""
import ast
import os
import numpy as np
import pandas as pd
import wfdb


# 8 klas docelowych
TARGET_CLASSES = ["NORM", "MI", "NST_", "ISC_", "LBBB", "RBBB", "LVH", "RVH"]

CLASS_NAMES_PL = {
    "NORM": "Zdrowy (NORM)",
    "MI": "Zawał mięśnia sercowego (MI)",
    "NST_": "Niespecyficzne zmiany ST/T (NST_)",
    "ISC_": "Niedokrwienne zmiany ST/T (ISC_)",
    "LBBB": "Całkowity blok lewej odnogi pęczka Hisa (LBBB)",
    "RBBB": "Całkowity blok prawej odnogi pęczka Hisa (RBBB)",
    "LVH": "Przerost lewej komory (LVH)",
    "RVH": "Przerost prawej komory (RVH)",
}

# SCP codes that map to each target class
# MI: all codes whose diagnostic_class == "MI"
MI_CODES = {
    "IMI", "ASMI", "ALMI", "AMI", "ILMI", "LMI", "PMI",
    "INJAL", "INJAS", "INJIL", "INJIN", "INJLA",
}
# ISC_: all ischemic ST/T codes
ISC_CODES = {"ISC_", "ISCA", "ISCAL", "ISCAS", "ISCIL", "ISCIN", "ISCLA"}


def _build_scp_to_target(scp_df: pd.DataFrame) -> dict:
    """Build mapping from individual SCP codes to our 8 target classes."""
    mapping = {}

    for _, row in scp_df.iterrows():
        code = row["Unnamed: 0"] if "Unnamed: 0" in row.index else row.name
        diag_class = str(row.get("diagnostic_class", ""))
        diag_sub = str(row.get("diagnostic_subclass", ""))

        if code == "NORM":
            mapping[code] = "NORM"
        elif code in MI_CODES or diag_class == "MI":
            mapping[code] = "MI"
        elif code == "NST_" or diag_sub == "NST_":
            mapping[code] = "NST_"
        elif code in ISC_CODES or diag_sub == "ISC_":
            mapping[code] = "ISC_"
        elif code == "CLBBB":
            mapping[code] = "LBBB"
        elif code == "CRBBB":
            mapping[code] = "RBBB"
        elif code == "LVH":
            mapping[code] = "LVH"
        elif code == "RVH":
            mapping[code] = "RVH"

    return mapping


def load_metadata(data_dir: str, sampling_rate: int = 500):
    """
    Load PTB-XL metadata and create 8-class multi-label targets.

    Returns:
        df: DataFrame with columns including 'labels' (8-dim numpy array)
        scp_to_target: dict mapping SCP codes to target class names
    """
    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    scp_path = os.path.join(data_dir, "scp_statements.csv")

    df = pd.read_csv(db_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_to_target = _build_scp_to_target(scp_df)

    # Build label vectors
    labels = []
    for _, row in df.iterrows():
        label_vec = np.zeros(len(TARGET_CLASSES), dtype=np.float32)
        for code, likelihood in row.scp_codes.items():
            if likelihood > 50 and code in scp_to_target:
                target = scp_to_target[code]
                idx = TARGET_CLASSES.index(target)
                label_vec[idx] = 1.0
        labels.append(label_vec)

    df["labels"] = labels

    # Filter: keep only records with at least one target class
    has_label = df["labels"].apply(lambda x: x.sum() > 0)
    df = df[has_label].copy()

    # Set correct filename column based on sampling rate
    if sampling_rate == 500:
        df["filename"] = df["filename_hr"]
    else:
        df["filename"] = df["filename_lr"]

    return df, scp_to_target


def load_signal(record_path: str) -> np.ndarray:
    """
    Load a single WFDB record.

    Args:
        record_path: path without extension (e.g., 'data/ptb-xl.../records500/00000/00001_hr')

    Returns:
        signal: numpy array of shape (n_samples, 12)
    """
    record = wfdb.rdrecord(record_path)
    return record.p_signal.astype(np.float32)


def compute_normalization_stats(df: pd.DataFrame, data_dir: str, train_folds=(1, 2, 3, 4, 5, 6, 7, 8)):
    """
    Compute per-lead mean and std from training folds.

    Returns:
        mean: shape (12,)
        std: shape (12,)
    """
    train_df = df[df["strat_fold"].isin(train_folds)]

    # Sample subset for efficiency (use all if feasible)
    n_samples = min(len(train_df), 2000)
    sample_df = train_df.sample(n=n_samples, random_state=42)

    all_signals = []
    for _, row in sample_df.iterrows():
        path = os.path.join(data_dir, row["filename"])
        try:
            sig = load_signal(path)
            all_signals.append(sig)
        except Exception:
            continue

    all_signals = np.concatenate(all_signals, axis=0)  # (total_samples, 12)
    mean = all_signals.mean(axis=0).astype(np.float32)
    std = all_signals.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0  # avoid division by zero

    return mean, std


def normalize_signal(signal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize signal per-lead using precomputed stats."""
    return (signal - mean) / std
