"""
Ewaluacja modelu na zbiorze testowym PTB-XL (fold 10).
Użycie: python -m src.evaluate --data_dir data/ptb-xl-... --model_path models/inception1d_best.pt
"""
import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.preprocessing import load_metadata, TARGET_CLASSES, CLASS_NAMES_PL
from src.dataset import PTBXLDataset
from torch.utils.data import DataLoader
from src.model import build_model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    # Load data
    df, _ = load_metadata(args.data_dir, sampling_rate=500)
    test_df = df[df["strat_fold"] == 10]
    print(f"Próbek testowych: {len(test_df)}")

    test_ds = PTBXLDataset(test_df, args.data_dir, mean, std, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model
    model = build_model(input_channels=12, num_classes=len(TARGET_CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Ewaluacja"):
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Macro AUC
    macro_auc = roc_auc_score(all_labels, all_preds, average="macro")
    per_class_auc = roc_auc_score(all_labels, all_preds, average=None)

    print(f"\n{'='*60}")
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"{'='*60}")
    print(f"\nPer-class AUC:")
    for i, cls in enumerate(TARGET_CLASSES):
        name = CLASS_NAMES_PL[cls]
        print(f"  {name}: {per_class_auc[i]:.4f}")

    # Classification report (thresholded at 0.5)
    binary_preds = (all_preds >= 0.5).astype(int)
    print(f"\n{'='*60}")
    print("Classification Report (threshold=0.5):")
    print(f"{'='*60}")
    print(classification_report(
        all_labels, binary_preds,
        target_names=TARGET_CLASSES, zero_division=0,
    ))

    # Per-class confusion matrices
    print(f"\nMacierze pomyłek (per-class):")
    for i, cls in enumerate(TARGET_CLASSES):
        cm = confusion_matrix(all_labels[:, i], binary_preds[:, i])
        print(f"\n  {cls}:")
        print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja modelu Inception1D")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/inception1d_best.pt")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
