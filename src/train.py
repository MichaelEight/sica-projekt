"""
Pętla treningowa dla modelu Inception1D na zbiorze PTB-XL.
Użycie: python -m src.train --data_dir data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.preprocessing import load_metadata, compute_normalization_stats, TARGET_CLASSES
from src.dataset import get_dataloaders
from src.model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for signals, labels in tqdm(loader, desc="  Train", leave=False):
        signals, labels = signals.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except ValueError:
        auc = 0.0

    return avg_loss, auc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for signals, labels in tqdm(loader, desc="  Val  ", leave=False):
        signals, labels = signals.to(device), labels.to(device)

        logits = model(signals)
        loss = criterion(logits, labels)

        total_loss += loss.item() * signals.size(0)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        macro_auc = roc_auc_score(all_labels, all_preds, average="macro")
        per_class_auc = roc_auc_score(all_labels, all_preds, average=None)
    except ValueError:
        macro_auc = 0.0
        per_class_auc = np.zeros(all_labels.shape[1])

    return avg_loss, macro_auc, per_class_auc


def train(args):
    device = get_device()
    print(f"Urządzenie: {device}")

    # Load data
    print("Ładowanie metadanych...")
    df, scp_to_target = load_metadata(args.data_dir, sampling_rate=500)
    print(f"Rekordów po filtracji: {len(df)}")

    # Label distribution
    labels_matrix = np.stack(df["labels"].values)
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"  {cls}: {int(labels_matrix[:, i].sum())} próbek")

    # Normalization stats
    stats_path = os.path.join(args.save_dir, "norm_stats.npz")
    if os.path.exists(stats_path):
        print("Ładowanie statystyk normalizacji...")
        stats = np.load(stats_path)
        mean, std = stats["mean"], stats["std"]
    else:
        print("Obliczanie statystyk normalizacji...")
        mean, std = compute_normalization_stats(df, args.data_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        np.savez(stats_path, mean=mean, std=std)

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        df, args.data_dir, mean, std,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Model
    model = build_model(input_channels=12, num_classes=len(TARGET_CLASSES)).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parametry modelu: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True)

    # Training loop
    best_auc = 0.0
    patience_counter = 0
    best_path = os.path.join(args.save_dir, "inception1d_best.pt")

    print(f"\nRozpoczęcie treningu ({args.epochs} epok)...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, per_class_auc = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_auc)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoka {epoch:3d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
            f"Val loss: {val_loss:.4f}, AUC: {val_auc:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )

        # Per-class AUC
        if epoch % 5 == 0 or epoch == 1:
            print("  Per-class AUC:")
            for i, cls in enumerate(TARGET_CLASSES):
                print(f"    {cls}: {per_class_auc[i]:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "mean": mean,
                "std": std,
            }, best_path)
            print(f"  ★ Nowy najlepszy model (AUC={val_auc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping po {epoch} epokach (brak poprawy od {args.patience} epok).")
            break

    print(f"\nNajlepszy val AUC: {best_auc:.4f}")
    print(f"Model zapisany: {best_path}")

    # Final evaluation on test set
    print("\nEwaluacja na zbiorze testowym (fold 10)...")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_auc, test_per_class = eval_epoch(model, test_loader, criterion, device)
    print(f"Test macro-AUC: {test_auc:.4f}")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"  {cls}: {test_per_class[i]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Trening Inception1D na PTB-XL")
    parser.add_argument("--data_dir", type=str, required=True, help="Ścieżka do katalogu PTB-XL")
    parser.add_argument("--save_dir", type=str, default="models", help="Katalog na checkpointy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
