"""
Inferencja pojedynczego pliku EKG.
Użycie: python -m src.predict --record_path path/to/record --model_path models/inception1d_best.pt
"""
import argparse
import numpy as np
import torch
import wfdb

from src.model import build_model
from src.preprocessing import TARGET_CLASSES, CLASS_NAMES_PL, normalize_signal
from src.grad_cam import GradCAM1D


def load_model(model_path, device=None):
    """Load trained model and normalization stats."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(input_channels=12, num_classes=len(TARGET_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return model, mean, std, device


def predict_record(record_path, model, mean, std, device):
    """
    Run inference on a single WFDB record.

    Args:
        record_path: path without extension
        model: loaded Inception1D model
        mean, std: normalization stats
        device: torch device

    Returns:
        probabilities: dict {class_name: probability}
        signal: raw signal array (n_samples, 12)
        leads: list of lead names
        fs: sampling frequency
    """
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.astype(np.float32)
    leads = record.sig_name
    fs = record.fs

    # Normalize
    signal_norm = normalize_signal(signal, mean, std)
    signal_norm = np.nan_to_num(signal_norm, nan=0.0)

    # To tensor: (1, 12, n_samples)
    x = torch.tensor(signal_norm.T, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    probabilities = {}
    for i, cls in enumerate(TARGET_CLASSES):
        probabilities[cls] = float(probs[i])

    return probabilities, signal, leads, fs, x


def predict_with_gradcam(record_path, model, mean, std, device):
    """
    Run inference with Grad-CAM explanations.

    Returns:
        probabilities: dict {class_name: probability}
        heatmaps: dict {class_name: heatmap_array}
        signal: raw signal array
        leads: lead names
        fs: sampling frequency
    """
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.astype(np.float32)
    leads = record.sig_name
    fs = record.fs

    signal_norm = normalize_signal(signal, mean, std)
    signal_norm = np.nan_to_num(signal_norm, nan=0.0)

    x = torch.tensor(signal_norm.T, dtype=torch.float32).unsqueeze(0).to(device)

    # Grad-CAM
    grad_cam = GradCAM1D(model)
    cam_heatmaps, cam_probs = grad_cam.generate_all_classes(x, signal_length=signal.shape[0])

    probabilities = {}
    heatmaps = {}
    for i, cls in enumerate(TARGET_CLASSES):
        probabilities[cls] = cam_probs[i]
        heatmaps[cls] = cam_heatmaps[i]

    return probabilities, heatmaps, signal, leads, fs


def main():
    parser = argparse.ArgumentParser(description="Predykcja EKG")
    parser.add_argument("--record_path", type=str, required=True, help="Ścieżka do rekordu WFDB (bez rozszerzenia)")
    parser.add_argument("--model_path", type=str, default="models/inception1d_best.pt")
    parser.add_argument("--gradcam", action="store_true", help="Generuj mapy Grad-CAM")
    args = parser.parse_args()

    model, mean, std, device = load_model(args.model_path)

    if args.gradcam:
        probs, heatmaps, signal, leads, fs = predict_with_gradcam(
            args.record_path, model, mean, std, device
        )
    else:
        probs, signal, leads, fs, _ = predict_record(
            args.record_path, model, mean, std, device
        )
        heatmaps = None

    print(f"\nWyniki klasyfikacji dla: {args.record_path}")
    print(f"Częstotliwość: {fs} Hz, Czas trwania: {signal.shape[0]/fs:.1f}s")
    print(f"Odprowadzenia: {', '.join(leads)}")
    print(f"\n{'Klasa':<50} {'Pewność':>10}")
    print("-" * 62)

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        name = CLASS_NAMES_PL[cls]
        bar = "█" * int(prob * 30)
        print(f"  {name:<48} {prob:>7.1%}  {bar}")


if __name__ == "__main__":
    main()
