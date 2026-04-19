from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb

from model.models import Inception1DNet
from model.training.schema import CANONICAL_CLASS_COLUMNS


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
    if class_names is not None:
        classes = class_names
    elif num_classes == len(CANONICAL_CLASS_COLUMNS):
        classes = CANONICAL_CLASS_COLUMNS
    else:
        classes = [f"class_{i}" for i in range(num_classes)]

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


def explain_prediction(
        model: Inception1DNet,
        data: np.ndarray | torch.Tensor | str | Path,
        target_classes: list[int] | list[str] | None = None,
        threshold: float = 0.5,
        class_names: list[str] | None = None,
        device: str | torch.device | None = None,
) -> dict[str, Any]:
    """
    Generuje wyjaśnienia XAI (Grad-CAM + Saliency) dla pojedynczego zapisu EKG.
    Funkcja zaprojektowana pod kątem łatwej integracji z frontendem (UI).

    Dla programisty UI:
    -------------------
    Zwraca słownik (łatwy do zamiany na JSON) o strukturze:
    {
        "source_id": "input_0",
        "explanations": [
            {
                "class_idx": 1,
                "class_name": "class_front_heart_attack",
                "probability": 0.95,

                # time_heatmap: tablica 5000 floatów [0.0 - 1.0].
                # UI powinno nałożyć cień na wykres EKG tam, gdzie wartości są > 0.1
                "time_heatmap": [0.0, 0.0, 0.12, 0.45, 0.8, 0.9, ...],

                # lead_importance_percent: słownik ważności poszczególnych odprowadzeń.
                # UI może z tego zbudować wykres słupkowy (suma zawsze = 100%).
                "lead_importance_percent": {
                    "I": 2.5, "II": 1.2, "III": 0.8, "aVR": 0.5,
                    "aVL": 1.0, "aVF": 1.5, "V1": 5.0, "V2": 45.0,
                    "V3": 35.0, "V4": 5.0, "V5": 1.5, "V6": 1.0
                },

                "top_lead": "V2" # Odprowadzenie z największym wpływem na decyzję
            },
            ...
        ]
    }
    """
    # 1. Przygotowanie danych (bierzemy tylko pierwsze 10 sekund dla spójności XAI)
    samples, sources = _load_samples(data)
    if not samples:
        raise ValueError("No valid data provided.")

    sample = samples[0]  # Operujemy na pierwszej dostarczonej próbce
    windows, _ = _split_sample_windows(sample, target_length=TARGET_LENGTH)
    x_numpy = windows[0]  # XAI liczymy dla pierwszego okna 5000 próbek

    # Obsługa nazw klas
    if class_names is None:
        class_names = CANONICAL_CLASS_COLUMNS

    # Obsługa urządzenia
    if device is None or str(device) == "auto":
        resolved_device = next(model.parameters()).device
    else:
        resolved_device = _resolve_device(device)
        model = model.to(resolved_device)

    x_tensor = torch.from_numpy(x_numpy).unsqueeze(0).to(resolved_device)

    # 2. Inicjalizacja Hooków (przechwytywanie wiedzy z wnętrza sieci)
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Podpinamy się pod ostatni blok konwolucyjny przed uśrednianiem (zdefiniowany w Inception1DNet)
    target_layer = model.block6
    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        # 3. Wykonanie Forward Pass (Musi wymagać gradientu na wejściu dla Lead Importance!)
        model.eval()
        x_tensor.requires_grad_(True)

        logits = model(x_tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        # 4. Ustalenie klas do wyjaśnienia (jeśli UI nie podało, bierzemy te powyżej threshold)
        if target_classes is None:
            active_indices = [i for i, p in enumerate(probs) if p >= threshold]
            # Jeśli brak patologii, wyjaśniamy klasę z największym prawdopodobieństwem
            if not active_indices:
                active_indices = [int(np.argmax(probs))]
        else:
            # Konwersja nazw klas na indeksy, jeśli UI wysłało stringi
            active_indices = []
            for c in target_classes:
                if isinstance(c, str):
                    if c in class_names:
                        active_indices.append(class_names.index(c))
                else:
                    active_indices.append(int(c))

        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        explanations = []

        # 5. Obliczanie XAI dla każdej wybranej klasy
        for class_idx in active_indices:
            model.zero_grad()
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            activations.clear()
            gradients.clear()

            # Puszczamy sygnał od nowa w trybie wymagającym gradientu
            out = model(x_tensor)

            # Wsteczna propagacja DLA KONKRETNEJ KLASY
            out[0, class_idx].backward(retain_graph=True)

            # --- A. GRAD-CAM (Kiedy sieć patrzyła?) ---
            acts = activations[0]
            grads = gradients[0]

            weights = torch.mean(grads, dim=2, keepdim=True)
            cam = torch.sum(weights * acts, dim=1).squeeze(0)
            cam = torch.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            time_heatmap = cam.detach().cpu().numpy().round(4).tolist()

            # --- B. LEAD IMPORTANCE (Gdzie sieć patrzyła?) ---
            input_grads = x_tensor.grad[0]  # kształt (12, 5000)
            lead_imp_raw = torch.sum(torch.abs(input_grads), dim=1)  # kształt (12,)

            lead_imp_sum = torch.sum(lead_imp_raw)
            if lead_imp_sum > 1e-8:
                lead_imp_pct = (lead_imp_raw / lead_imp_sum) * 100
            else:
                lead_imp_pct = torch.zeros_like(lead_imp_raw)

            lead_imp_pct = lead_imp_pct.detach().cpu().numpy().round(2)

            # Pakowanie wyników dla UI
            importance_dict = {lead_names[i]: float(lead_imp_pct[i]) for i in range(12)}
            top_lead_idx = int(np.argmax(lead_imp_pct))

            explanations.append({
                "class_idx": int(class_idx),
                "class_name": class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}",
                "probability": float(round(probs[class_idx], 4)),
                "time_heatmap": time_heatmap,
                "lead_importance_percent": importance_dict,
                "top_lead": lead_names[top_lead_idx]
            })

        return {
            "source_id": sources[0],
            "explanations": explanations
        }

    finally:
        # 6. Zawsze usuwamy hooki, by nie wyciekała pamięć i nie psuło to zwykłej inferencji!
        hook_f.remove()
        hook_b.remove()