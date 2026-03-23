from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb

from model.models import Inception1DNet


TARGET_FS = 500
MIN_SECONDS = 10
TARGET_LENGTH = TARGET_FS * MIN_SECONDS


def _normalize_input_shape(data: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert input into a float32 numpy array with shape (B, 12, N)."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    if arr.ndim == 2:
        if arr.shape[0] == 12:
            arr = arr[None, :, :]
        elif arr.shape[1] == 12:
            arr = arr.T[None, :, :]
        else:
            raise ValueError(f"Expected 12 leads in 2D input, got shape={arr.shape}.")
    elif arr.ndim == 3:
        if arr.shape[1] == 12:
            pass
        elif arr.shape[2] == 12:
            arr = np.transpose(arr, (0, 2, 1))
        else:
            raise ValueError(f"Expected 12 leads in 3D input, got shape={arr.shape}.")
    else:
        raise ValueError(f"Expected 2D or 3D input, got ndim={arr.ndim}.")

    return arr.astype(np.float32, copy=False)


def _resolve_wfdb_base(path_like: str | Path) -> str:
    path = Path(path_like)
    if path.suffix.lower() in {".hea", ".dat"}:
        path = path.with_suffix("")
    return str(path)


def _read_wfdb_record(path_like: str | Path) -> np.ndarray:
    """Read WFDB record and return array with shape (12, N)."""
    base = _resolve_wfdb_base(path_like)
    signal_arr, _ = wfdb.rdsamp(base)
    signal = np.asarray(signal_arr, dtype=np.float32).T
    if signal.shape[0] != 12:
        raise ValueError(f"Expected 12 leads, got {signal.shape[0]} for record: {path_like}")
    return signal


def _split_sample_windows(sample: np.ndarray, target_length: int = TARGET_LENGTH) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split one sample (12, N) into windows of target_length.

    For N > target_length, creates non-overlapping chunks and an extra last chunk
    ending at N when remainder exists.
    """
    n = sample.shape[1]
    if n < target_length:
        raise ValueError(
            f"Record is shorter than {MIN_SECONDS}s ({target_length} samples at {TARGET_FS} Hz): got {n}."
        )

    if n == target_length:
        return [sample], [(0, target_length)]

    starts = list(range(0, n - target_length + 1, target_length))
    last_start = n - target_length
    if starts[-1] != last_start:
        starts.append(last_start)

    windows = [sample[:, s : s + target_length] for s in starts]
    ranges = [(s, s + target_length) for s in starts]
    return windows, ranges


def _load_samples(data: np.ndarray | torch.Tensor | str | Path | list[str] | list[Path]) -> tuple[list[np.ndarray], list[str]]:
    """Load input into a list of arrays (12, N) and source ids."""
    if isinstance(data, (str, Path)):
        sample = _read_wfdb_record(data)
        return [sample], [str(data)]

    if isinstance(data, list) and data and isinstance(data[0], (str, Path)):
        samples = [_read_wfdb_record(p) for p in data]
        sources = [str(p) for p in data]
        return samples, sources

    arr = _normalize_input_shape(data)  # type: ignore[arg-type]
    samples = [arr[i] for i in range(arr.shape[0])]
    sources = [f"input_{i}" for i in range(arr.shape[0])]
    return samples, sources


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint_model(
    weights_path: str | Path,
    num_classes: int = 8,
    device: str | torch.device | None = None,
) -> tuple[Inception1DNet, torch.device]:
    """Load Inception1DNet and checkpoint weights.

    Supports both plain state_dict checkpoints and training checkpoints with
    a `model_state_dict` key.
    """
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    resolved_device = _resolve_device(device)
    model = Inception1DNet(num_classes=num_classes).to(resolved_device)

    checkpoint = torch.load(weights, map_location=resolved_device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model, resolved_device


def predict_with_model(
    model: Inception1DNet,
    data: np.ndarray | torch.Tensor | str | Path | list[str] | list[Path],
    threshold: float = 0.5,
    class_names: list[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run inference for tensor/ndarray input or WFDB record path(s).

    Input records can be longer than 10s and are automatically split into
    10-second windows (5000 samples). Predictions are aggregated per input item
    by mean probability across windows.
    """
    samples, sources = _load_samples(data)

    segment_arrays: list[np.ndarray] = []
    segment_meta: list[dict[str, Any]] = []
    for source_idx, sample in enumerate(samples):
        windows, ranges = _split_sample_windows(sample, target_length=TARGET_LENGTH)
        for win, (start, end) in zip(windows, ranges):
            segment_arrays.append(win)
            segment_meta.append(
                {
                    "source_index": source_idx,
                    "source_id": sources[source_idx],
                    "start": int(start),
                    "end": int(end),
                }
            )

    arr = np.stack(segment_arrays, axis=0).astype(np.float32, copy=False)

    if device is None or str(device) == "auto":
        resolved_device = next(model.parameters()).device
    else:
        resolved_device = _resolve_device(device)
        if next(model.parameters()).device != resolved_device:
            model = model.to(resolved_device)
    x = torch.from_numpy(arr).to(resolved_device)

    with torch.no_grad():
        probs = model.forward_inference(x).detach().cpu().numpy()

    segment_preds = (probs >= threshold).astype(np.int32)
    num_classes = probs.shape[1]
    classes = class_names if class_names is not None else [f"class_{i}" for i in range(num_classes)]

    per_input_probs: list[np.ndarray] = []
    for i in range(len(samples)):
        idxs = [k for k, m in enumerate(segment_meta) if int(m["source_index"]) == i]
        per_input_probs.append(probs[idxs].mean(axis=0))

    agg_probs = np.stack(per_input_probs, axis=0)
    agg_preds = (agg_probs >= threshold).astype(np.int32)

    positive_labels = [
        [classes[j] for j in range(num_classes) if agg_preds[i, j] == 1]
        for i in range(agg_preds.shape[0])
    ]

    return {
        "class_names": classes,
        "threshold": float(threshold),
        "probabilities": agg_probs.tolist(),
        "predictions": agg_preds.tolist(),
        "positive_labels": positive_labels,
        "source_ids": sources,
        "segment_probabilities": probs.tolist(),
        "segment_predictions": segment_preds.tolist(),
        "segments": segment_meta,
    }


def predict_from_checkpoint(
    weights_path: str | Path,
    data: np.ndarray | torch.Tensor | str | Path | list[str] | list[Path],
    threshold: float = 0.5,
    class_names: list[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Convenience API for external modules: weights + data -> predictions."""
    num_classes = len(class_names) if class_names else 8
    model, resolved_device = load_checkpoint_model(weights_path, num_classes=num_classes, device=device)
    return predict_with_model(
        model=model,
        data=data,
        threshold=threshold,
        class_names=class_names,
        device=resolved_device,
    )


