from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.models import Inception1DNet
from model.training.dataset import ECGWFDBDataset, compute_label_stats, compute_pos_weight_tensor, save_json
from model.training.evaluate import run_evaluation
from model.training.metadata_inspector import inspect_all_metadata
from model.training.metrics import safe_macro_auc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "training"
ANNOTATIONS_DIR = PROJECT_ROOT / "model" / "annotations"


class FocalLoss(nn.Module):
    """Focal Loss dla multi-label klasyfikacji ECG.
    
    Implementuje Focal Loss z artykułu: https://arxiv.org/abs/1708.02002
    Redukuje wagę łatwych przykładów, skupiając się na trudnych przypadkach.
    
    Parametry:
    - gamma: focusing parameter (domyślnie 2.0)
    - alpha: balancing parameter dla klas (tensor z wagami)
    - reduction: 'mean', 'sum', lub 'none'
    """
    
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) - raw model output
            targets: (batch, num_classes) - binary targets in [0, 1]
        
        Returns:
            loss: scalar or per-element loss
        """
        # Oblicz BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Oblicz prawd. dla focusing
        probs = torch.sigmoid(logits)
        
        # Focal loss = -alpha * (1 - p_t)^gamma * BCE
        # gdzie p_t to prawd. prawidłowej klasy
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = torch.pow(1 - p_t, self.gamma)
        focal_loss = focal_weight * bce_loss
        
        # Zastosuj alpha weighting jeśli dostępne
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Zastosuj reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TolerantImbalanceBCELoss(nn.Module):
    """BCEWithLogits z pos_weight + mniejsza kara dla błędów <= ok. 5 p.p."""

    def __init__(self, pos_weight: torch.Tensor, tolerance: float = 0.05, in_tolerance_weight: float = 0.15) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.tolerance = float(tolerance)
        self.in_tolerance_weight = float(in_tolerance_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        probs = torch.sigmoid(logits).detach()
        abs_err = torch.abs(probs - targets)

        # Dla błędów w okolicy 5 p.p. waga jest wyraźnie mniejsza, ale niezerowa.
        denom = max(2.0 * self.tolerance, 1e-6)
        ramp = torch.clamp(abs_err / denom, min=0.0, max=1.0)
        scale = self.in_tolerance_weight + (1.0 - self.in_tolerance_weight) * (ramp**2)

        return (base_loss * scale).mean()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG training pipeline")
    parser.add_argument("--sanity", action="store_true", help="Szybki przebieg: mniej epok i limit batchy.")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maksymalna liczba epok.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (val loss).")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit batchy treningowych na epokę.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit batchy walidacyjnych/testowych na epokę.")
    parser.add_argument("--skip-test-eval", action="store_true", help="Pomiń końcową ewaluację na teście.")
    parser.add_argument("--checkpoint-freq", type=int, default=1, help="Co ile epok zapisywać checkpoint (domyślnie 1 - każda epoka).")
    parser.add_argument("--log-freq", type=int, default=1, help="Co ile epok wypisywać logi na konsoli (domyślnie 1 - każda epoka).")
    parser.add_argument("--val-freq", type=int, default=1, help="Co ile epok robić walidację (domyślnie 1 - każda epoka).")
    parser.add_argument("--resume", type=str, default=None, help="Ścieżka do checkpointa do wznowienia treningu.")
    parser.add_argument("--skip-plots", action="store_true", help="Pomiń rysowanie grafik (tylko logi).")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "bce"], help="Funkcja straty: 'focal' (Focal Loss) lub 'bce' (BCEWithLogitsLoss z tolerancją)")
    parser.add_argument("--num-workers", type=int, default=4, help="Liczba workerów DataLoadera.")
    return parser.parse_args()


def _derive_columns_fast() -> tuple[list[str], dict[str, str | None]]:
    train_meta_path = DATA_ROOT / "train" / "train_metadata.csv"
    meta = pd.read_csv(train_meta_path, nrows=0)
    cols = list(meta.columns)
    label_columns = [c for c in cols if c.startswith("class_")]
    file_columns: dict[str, str | None] = {
        "base": "local_record_base" if "local_record_base" in cols else None,
        "dat": next((c for c in cols if c.endswith("_dat_file")), None),
        "hea": next((c for c in cols if c.endswith("_hea_file")), None),
    }
    return label_columns, file_columns



def _ensure_output_dirs() -> None:
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)



def _build_dataloaders(
    label_columns: list[str],
    file_columns: dict[str, str | None],
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, ECGWFDBDataset]:
    train_ds = ECGWFDBDataset(
        split_dir=DATA_ROOT / "train",
        metadata_filename="train_metadata.csv",
        label_columns=label_columns,
        file_columns=file_columns,
        target_length=5000,
    )
    val_ds = ECGWFDBDataset(
        split_dir=DATA_ROOT / "val",
        metadata_filename="val_metadata.csv",
        label_columns=label_columns,
        file_columns=file_columns,
        target_length=5000,
    )
    test_ds = ECGWFDBDataset(
        split_dir=DATA_ROOT / "test",
        metadata_filename="test_metadata.csv",
        label_columns=label_columns,
        file_columns=file_columns,
        target_length=5000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_ds



def _write_train_log_header(log_path: Path) -> None:
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_macro_auc"])



def _append_train_log(log_path: Path, epoch: int, train_loss: float, val_loss: float, val_auc: float) -> None:
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_auc:.6f}"])



def _plot_curves(history: dict[str, list[float]]) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANNOTATIONS_DIR / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_auc"], label="val_macro_auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation Macro AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANNOTATIONS_DIR / "auc_curve.png", dpi=150)
    plt.close()



def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def _cleanup_old_checkpoints(checkpoint_dir: Path) -> None:
    """Keep only best_model.pt and last_model.pt. Delete any other .pt files."""
    protected = {"best_model.pt", "last_model.pt"}
    for pt_file in checkpoint_dir.glob("*.pt"):
        if pt_file.name not in protected:
            try:
                pt_file.unlink()
                print(f"[CLEANUP] Removed stray checkpoint: {pt_file.name}")
            except OSError:
                pass



def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    sample_count = 0
    all_true = []
    all_prob = []

    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        probs = torch.sigmoid(logits)
        all_true.append(y.detach().cpu().numpy())
        all_prob.append(probs.detach().cpu().numpy())

    mean_loss = total_loss / max(sample_count, 1)
    if not all_true:
        return mean_loss, np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    return mean_loss, y_true, y_prob



def main() -> None:
    args = _parse_args()

    _ensure_output_dirs()

    # ===== GPU INFO =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] Karta graficzna: {gpu_name} | Pamięć: {gpu_memory:.2f} GB")
    else:
        print("[GPU] Brak karty graficznej - będzie użyty CPU")

    if args.sanity:
        label_columns, file_columns = _derive_columns_fast()
    else:
        cache_path = ANNOTATIONS_DIR / "metadata_inspection.json"
        if cache_path.exists():
            import json

            with cache_path.open(encoding="utf-8") as f:
                inspection = json.load(f)
            print("[CACHE] Loaded metadata inspection from cache.")
        else:
            inspection = inspect_all_metadata(DATA_ROOT)
            save_json(cache_path, inspection)
            print("[CACHE] Metadata inspection saved to cache.")
        label_columns = inspection["label_columns"]  # type: ignore[assignment]
        file_columns = inspection["file_columns"]  # type: ignore[assignment]

    train_loader, val_loader, test_loader, train_ds = _build_dataloaders(
        label_columns, file_columns, num_workers=args.num_workers
    )

    # BCEWithLogitsLoss is correct for independent multi-label targets.
    # CrossEntropyLoss/Softmax assume exactly one true class per sample, which is wrong here.
    pos_weight = compute_pos_weight_tensor(train_ds)
    label_stats = compute_label_stats(train_ds)

    save_json(ANNOTATIONS_DIR / "class_names.json", label_columns)

    model = Inception1DNet(num_classes=len(label_columns)).to(device)
    
    # Select loss function based on argument
    loss_type = args.loss
    if loss_type == "focal":
        pos_ratio = torch.tensor(
            [s["positive_ratio"] for s in label_stats.values()],
            dtype=torch.float32,
        ).clamp(min=1e-6)
        alpha = (1.0 / pos_ratio).to(device)
        alpha = alpha / alpha.sum() * len(pos_ratio)
        criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction="mean")
        print(f"[LOSS] Using Focal Loss (γ=2.0) with alpha weighting")
    else:
        # TolerantImbalanceBCELoss
        criterion = TolerantImbalanceBCELoss(
            pos_weight=pos_weight.to(device),
            tolerance=0.05,
            in_tolerance_weight=0.15,
        )
        print(f"[LOSS] Using TolerantImbalanceBCELoss")
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume from checkpoint if provided
    resume_checkpoint = args.resume
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"[RESUME] Loaded checkpoint from {resume_checkpoint}, starting from epoch {start_epoch}")
        else:
            print(f"[WARN] Checkpoint {resume_checkpoint} not found, starting fresh")
    
    max_epochs = args.max_epochs
    patience = args.patience
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    skip_test_eval = args.skip_test_eval
    checkpoint_freq = args.checkpoint_freq
    log_freq = args.log_freq
    val_freq = args.val_freq
    skip_plots = args.skip_plots

    if args.sanity:
        max_epochs = 1
        patience = 1
        max_train_batches = 8 if max_train_batches is None else max_train_batches
        max_val_batches = 4 if max_val_batches is None else max_val_batches
        skip_test_eval = True if not args.skip_test_eval else True
        print(
            "[SANITY] enabled | "
            f"max_epochs={max_epochs} | max_train_batches={max_train_batches} | "
            f"max_val_batches={max_val_batches} | skip_test_eval={skip_test_eval}"
        )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_macro_auc": []}
    log_csv = ANNOTATIONS_DIR / "train_log.csv"
    if start_epoch == 1:
        _write_train_log_header(log_csv)
    else:
        print(f"[RESUME] Appending to existing train log: {log_csv}")

    no_improve = 0

    if start_epoch == 1:
        best_path = ANNOTATIONS_DIR / "best_model.pt"
        if best_path.exists():
            import time

            ts = int(time.time())
            backup = ANNOTATIONS_DIR / f"best_model_backup_{ts}.pt"
            best_path.rename(backup)
            print(f"[WARN] Fresh start: existing best_model.pt backed up to {backup.name}")

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=max_train_batches,
        )

        # Walidacja co val_freq epok
        should_validate = (epoch % val_freq == 0) or (epoch == max_epochs)
        
        if should_validate:
            with torch.no_grad():
                val_loss, val_true, val_prob = _run_epoch(
                    model,
                    val_loader,
                    criterion,
                    None,
                    device,
                    max_batches=max_val_batches,
                )
            val_macro_auc = safe_macro_auc(val_true, val_prob) if val_true.size else float("nan")
        else:
            val_loss = float("nan")
            val_macro_auc = float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_auc"].append(val_macro_auc)
        
        if should_validate:
            _append_train_log(log_csv, epoch, train_loss, val_loss, val_macro_auc)

        # Logi na konsoli co log_freq epok
        if epoch % log_freq == 0 or epoch == max_epochs:
            ts = datetime.now().strftime("%H:%M:%S")
            if should_validate:
                print(
                    f"[{ts}][Epoch {epoch:03d}] "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_auc={val_macro_auc:.4f} | "
                    f"no_improve={no_improve}/{patience}"
                )
            else:
                print(f"[{ts}][Epoch {epoch:03d}] train_loss={train_loss:.4f}")

        # Checkpoint co checkpoint_freq epok
        if epoch % checkpoint_freq == 0 or epoch == max_epochs:
            _save_checkpoint(ANNOTATIONS_DIR / "last_model.pt", model, optimizer, epoch, best_val_loss)
            _cleanup_old_checkpoints(ANNOTATIONS_DIR)

        # Early stopping logic - zawsze gdy jest walidacja
        if should_validate:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                _save_checkpoint(ANNOTATIONS_DIR / "best_model.pt", model, optimizer, epoch, best_val_loss)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience}).")
                    break

    # Plot curves only if skip_plots is not set
    if not skip_plots:
        _plot_curves(history)
    else:
        print("[INFO] Skipping plot generation (--skip-plots enabled)")

    best_checkpoint = torch.load(ANNOTATIONS_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    if skip_test_eval:
        print("[INFO] Skipping full test evaluation.")
    else:
        run_evaluation(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=label_columns,
            output_path=ANNOTATIONS_DIR / "eval_results.txt",
            max_batches=max_val_batches,
        )


if __name__ == "__main__":
    main()



















