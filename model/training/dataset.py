from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset


class ECGWFDBDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        metadata_filename: str,
        label_columns: list[str],
        file_columns: dict[str, str | None],
        target_length: int = 5000,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.metadata_path = self.split_dir / metadata_filename
        self.label_columns = label_columns
        self.file_columns = file_columns
        self.target_length = target_length

        self.df = pd.read_csv(self.metadata_path)
        self.df = self._filter_rows(self.df)

    def _filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        nan_label_skips = 0
        unreadable_skips = 0
        printed_nan_examples = 0
        printed_unreadable_examples = 0
        max_examples = 5

        for idx, row in df.iterrows():
            label_values = row[self.label_columns]
            if label_values.isna().any():
                nan_label_skips += 1
                if printed_nan_examples < max_examples:
                    print(f"[WARN] Skip row {idx} in {self.metadata_path.name}: NaN in label column.")
                    printed_nan_examples += 1
                continue

            normalized_labels = self._normalize_labels(label_values.to_numpy(dtype=np.float32))
            if np.all(normalized_labels == 0.0):
                # Required exclusion rule: drop samples that contain no positive class.
                continue

            if not self._row_has_readable_record(row):
                unreadable_skips += 1
                if printed_unreadable_examples < max_examples:
                    print(f"[WARN] Skip row {idx} in {self.metadata_path.name}: missing/unreadable WFDB files.")
                    printed_unreadable_examples += 1
                continue

            rows.append(idx)

        if nan_label_skips or unreadable_skips:
            print(
                f"[INFO] {self.metadata_path.name}: kept={len(rows)} | "
                f"skipped_nan_labels={nan_label_skips} | skipped_unreadable={unreadable_skips}"
            )

        return df.loc[rows].reset_index(drop=True)

    @staticmethod
    def _normalize_labels(labels: np.ndarray) -> np.ndarray:
        # Labels in metadata are percentages [0, 100]. Keep soft targets in [0, 1].
        labels = labels / 100.0
        return np.clip(labels, 0.0, 1.0).astype(np.float32)

    def _record_base_from_row(self, row: pd.Series) -> str:
        candidates = self._candidate_record_bases(row)
        if not candidates:
            raise ValueError("Could not infer WFDB record base name from metadata columns.")
        return candidates[0]

    def _candidate_record_bases(self, row: pd.Series) -> list[str]:
        candidates: list[str] = []
        for key in ("base", "dat", "hea"):
            col = self.file_columns.get(key)
            if not col:
                continue
            raw_value = row.get(col)
            if not isinstance(raw_value, str):
                continue
            name = raw_value.strip()
            if name.lower().endswith(".dat") or name.lower().endswith(".hea"):
                name = name.rsplit(".", 1)[0]
            if name and name not in candidates:
                candidates.append(name)
        return candidates

    def _row_has_readable_record(self, row: pd.Series) -> bool:
        for base in self._candidate_record_bases(row):
            if self._record_is_readable(base):
                return True

        # Fallback: metadata często wskazuje poprawną parę .hea/.dat nawet gdy wpisy
        # wewnątrz nagłówka odwołują się do innej nazwy pliku .dat.
        dat_path, hea_path = self._metadata_paths_from_row(row)
        return dat_path is not None and hea_path is not None and dat_path.exists() and hea_path.exists()

    def _metadata_paths_from_row(self, row: pd.Series) -> tuple[Path | None, Path | None]:
        dat_col = self.file_columns.get("dat")
        hea_col = self.file_columns.get("hea")

        dat_name = row.get(dat_col) if dat_col else None
        hea_name = row.get(hea_col) if hea_col else None

        dat_path = self.split_dir / dat_name if isinstance(dat_name, str) else None
        hea_path = self.split_dir / hea_name if isinstance(hea_name, str) else None
        return dat_path, hea_path

    def _read_with_patched_header(self, row: pd.Series) -> np.ndarray:
        dat_path, hea_path = self._metadata_paths_from_row(row)
        if dat_path is None or hea_path is None:
            raise FileNotFoundError("Metadata does not provide local_dat_file/local_hea_file for fallback read.")
        if not dat_path.exists() or not hea_path.exists():
            raise FileNotFoundError(f"Missing fallback files: dat={dat_path} hea={hea_path}")

        cache_dir = Path(tempfile.gettempdir()) / "sica_wfdb_cache" / self.split_dir.name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Nazwa stabilna i zgodna z WFDB: tylko ASCII i bez spacji.
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in hea_path.stem)
        patched_base = cache_dir / f"patched_{safe_stem}"
        patched_hea = patched_base.with_suffix(".hea")

        source_lines = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not source_lines:
            raise ValueError(f"Empty header file: {hea_path}")

        first_parts = source_lines[0].split()
        if first_parts:
            first_parts[0] = patched_base.name
            source_lines[0] = " ".join(first_parts)

        cached_dat = cache_dir / f"{patched_base.name}.dat"
        if not cached_dat.exists():
            shutil.copy2(dat_path, cached_dat)

        dat_token = cached_dat.name
        nsig = int(first_parts[1]) if len(first_parts) > 1 and first_parts[1].isdigit() else 0
        last_signal_line = min(len(source_lines) - 1, nsig)
        for i in range(1, last_signal_line + 1):
            line = source_lines[i].strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                source_lines[i] = f"{dat_token} {parts[1]}"

        patched_hea.write_text("\n".join(source_lines) + "\n", encoding="utf-8")

        signal_arr, _ = wfdb.rdsamp(str(patched_base))
        return np.asarray(signal_arr, dtype=np.float32).T

    def _record_is_readable(self, record_base: str) -> bool:
        hea_path = self.split_dir / f"{record_base}.hea"
        if not hea_path.exists():
            return False

        try:
            header = wfdb.rdheader(str(self.split_dir / record_base))
        except Exception:
            return False

        dat_files = getattr(header, "file_name", None) or []
        if not dat_files:
            return False

        for dat_name in dat_files:
            if not (self.split_dir / dat_name).exists():
                return False
        return True

    def _fix_signal_length(self, signal: np.ndarray) -> np.ndarray:
        # signal shape expected: (12, N)
        n = signal.shape[1]
        if n > self.target_length:
            start = (n - self.target_length) // 2
            signal = signal[:, start : start + self.target_length]
        elif n < self.target_length:
            pad = self.target_length - n
            signal = np.pad(signal, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
        return signal

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        read_errors: list[str] = []
        signal = None
        record_base_used = None

        for record_base in self._candidate_record_bases(row):
            record_path_no_ext = str(self.split_dir / record_base)
            try:
                # wfdb.rdsamp() reads signal from .hea + .dat and returns ndarray shape (N, channels).
                signal_arr, _ = wfdb.rdsamp(record_path_no_ext)
                signal = np.asarray(signal_arr, dtype=np.float32).T  # to (channels, time)
                record_base_used = record_base
                break
            except Exception as exc:
                read_errors.append(f"{record_base}: {exc}")

        if signal is None:
            try:
                signal = self._read_with_patched_header(row)
                record_base_used = "patched_from_metadata"
            except Exception as exc:
                read_errors.append(f"patched_header: {exc}")
                raise FileNotFoundError(
                    f"Could not read WFDB record for row {idx} in {self.metadata_path.name}. "
                    f"Tried bases={self._candidate_record_bases(row)} | errors={read_errors}"
                )

        if signal.shape[0] != 12:
            raise ValueError(
                f"Expected 12 leads, got {signal.shape[0]} for record {record_base_used} in {self.split_dir}."
            )

        signal = self._fix_signal_length(signal)
        labels = self._normalize_labels(row[self.label_columns].to_numpy(dtype=np.float32))

        x = torch.from_numpy(signal).float()
        y = torch.from_numpy(labels).float()
        return x, y



def compute_label_stats(dataset: ECGWFDBDataset) -> dict[str, dict[str, float]]:
    labels_soft = dataset.df[dataset.label_columns].to_numpy(dtype=np.float32) / 100.0
    labels_soft = np.clip(labels_soft, 0.0, 1.0)

    labels_presence = (labels_soft > 0.0).astype(np.float32)
    count = labels_presence.sum(axis=0)
    ratio = count / max(len(labels_presence), 1)

    stats: dict[str, dict[str, float]] = {}
    for idx, col in enumerate(dataset.label_columns):
        stats[col] = {
            "positive_count": float(count[idx]),
            "positive_ratio": float(ratio[idx]),
            "mean_soft_label": float(labels_soft[:, idx].mean()) if len(labels_soft) else 0.0,
        }
    return stats


def compute_pos_weight_tensor(dataset: ECGWFDBDataset) -> torch.Tensor:
    stats = compute_label_stats(dataset)
    ratios = []
    for col in dataset.label_columns:
        r = stats[col]["positive_ratio"]
        r = min(max(r, 1e-6), 1.0 - 1e-6)
        ratios.append(r)

    ratios_arr = np.asarray(ratios, dtype=np.float32)
    pos_weight = (1.0 - ratios_arr) / ratios_arr
    return torch.tensor(pos_weight, dtype=torch.float32)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)




