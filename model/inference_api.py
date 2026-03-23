from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.models import Inception1DNet


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


def _fix_length(batch: np.ndarray, target_length: int = 5000) -> np.ndarray:
    """Center-crop or right-pad each sample to target_length."""
    fixed = np.zeros((batch.shape[0], 12, target_length), dtype=np.float32)
    for i in range(batch.shape[0]):
        sample = batch[i]
        n = sample.shape[1]
        if n >= target_length:
            start = (n - target_length) // 2
            fixed[i] = sample[:, start : start + target_length]
        else:
            fixed[i, :, :n] = sample
    return fixed


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
    data: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    class_names: list[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run model inference and return probabilities plus binary predictions."""
    arr = _normalize_input_shape(data)
    arr = _fix_length(arr, target_length=5000)

    if device is None or str(device) == "auto":
        resolved_device = next(model.parameters()).device
    else:
        resolved_device = _resolve_device(device)
        if next(model.parameters()).device != resolved_device:
            model = model.to(resolved_device)
    x = torch.from_numpy(arr).to(resolved_device)

    with torch.no_grad():
        probs = model.forward_inference(x).detach().cpu().numpy()

    preds = (probs >= threshold).astype(np.int32)
    num_classes = probs.shape[1]
    classes = class_names if class_names is not None else [f"class_{i}" for i in range(num_classes)]

    positive_labels = [
        [classes[j] for j in range(num_classes) if preds[i, j] == 1]
        for i in range(preds.shape[0])
    ]

    return {
        "class_names": classes,
        "threshold": float(threshold),
        "probabilities": probs.tolist(),
        "predictions": preds.tolist(),
        "positive_labels": positive_labels,
    }


def predict_from_checkpoint(
    weights_path: str | Path,
    data: np.ndarray | torch.Tensor,
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


