from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.inference_api import load_checkpoint_model
from model.training.dataset import ECGWFDBDataset
from model.training.metrics import per_class_metrics
from model.training.schema import infer_file_columns, infer_label_columns

DATA_ROOT = PROJECT_ROOT / "data" / "training"
SPLIT = "test"
SPLIT_DIR = DATA_ROOT / SPLIT
WEIGHTS_PATH = PROJECT_ROOT / "model" / "annotations" / "model-sota.pt"
REPORT_TXT_PATH = PROJECT_ROOT / "model" / "annotations" / "test_metrics_sota.txt"
REPORT_JSON_PATH = PROJECT_ROOT / "model" / "annotations" / "test_metrics_sota.json"
PLOTS_DIR = PROJECT_ROOT / "model" / "annotations" / "plots"
AUC_CLASS_PLOT_PATH = PLOTS_DIR / "test_metrics_sota_auc_per_class_pct.png"
F1_CLASS_PLOT_PATH = PLOTS_DIR / "test_metrics_sota_f1_per_class_pct.png"
MEAN_DOCTOR_GAP_CLASS_PLOT_PATH = PLOTS_DIR / "test_metrics_sota_mean_doctor_gap_per_class_pp.png"
MAX_DOCTOR_GAP_SPLIT_CLASS_PLOT_PATH = PLOTS_DIR / "test_metrics_sota_max_doctor_gap_split_per_class_pp.png"
THRESHOLD = 0.5
TARGET_LENGTH = 5000


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cuda_info() -> str:
    if not torch.cuda.is_available():
        return "CPU mode"

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    return f"cuda:{idx} | {name} | VRAM={total_gb:.1f}GB"


def _build_dataset() -> tuple[ECGWFDBDataset, list[str]]:
    metadata_path = SPLIT_DIR / f"{SPLIT}_metadata.csv"
    meta = pd.read_csv(metadata_path)
    label_columns = infer_label_columns(meta.columns)
    file_columns = infer_file_columns(meta.columns)

    dataset = ECGWFDBDataset(
        split_dir=SPLIT_DIR,
        metadata_filename=metadata_path.name,
        label_columns=label_columns,
        file_columns=file_columns,
        target_length=TARGET_LENGTH,
    )
    return dataset, label_columns


def _build_dataloader(dataset: ECGWFDBDataset, device: torch.device) -> DataLoader:
    workers = max(0, min(4, (os.cpu_count() or 2) - 1))
    # Ustawienia bezpieczne dla RTX 4050 45W, ale działające też na innych GPU/CPU.
    batch_size = 64 if device.type == "cuda" else 16
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )


def _collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_batches: list[np.ndarray] = []
    y_prob_batches: list[np.ndarray] = []

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader, start=1):
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            with amp_context:
                probs = model.forward_inference(x)

            y_true_batches.append(y.cpu().numpy())
            y_prob_batches.append(probs.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"[PROGRESS] batch={batch_idx}/{len(dataloader)}")

    if not y_true_batches:
        raise ValueError("Brak danych do ewaluacji.")

    y_true = np.concatenate(y_true_batches, axis=0)
    y_prob = np.concatenate(y_prob_batches, axis=0)
    return y_true, y_prob


def _safe_micro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true_bin = (y_true > 0.0).astype(np.int32)
    try:
        return float(roc_auc_score(y_true_bin.ravel(), y_prob.ravel()))
    except ValueError:
        return float("nan")


def _compute_doctor_distance_pp(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> tuple[list[dict[str, float]], float, float, float]:
    signed_error_pp = (y_prob - y_true) * 100.0
    abs_error_pp = np.abs(signed_error_pp)
    over_error_pp = np.maximum(signed_error_pp, 0.0)
    under_error_pp = np.maximum(-signed_error_pp, 0.0)

    mean_err = abs_error_pp.mean(axis=0)
    max_err = abs_error_pp.max(axis=0)
    max_over_err = over_error_pp.max(axis=0)
    max_under_err = under_error_pp.max(axis=0)

    rows = []
    for i, cls in enumerate(class_names):
        rows.append(
            {
                "class": str(cls),
                "mean_abs_diff_pp": float(mean_err[i]),
                "max_abs_diff_pp": float(max_err[i]),
                "max_overconfident_diff_pp": float(max_over_err[i]),
                "max_underconfident_diff_pp": float(max_under_err[i]),
            }
        )

    macro_mean = float(mean_err.mean()) if mean_err.size else float("nan")
    macro_max_over = float(max_over_err.mean()) if max_over_err.size else float("nan")
    macro_max_under = float(max_under_err.mean()) if max_under_err.size else float("nan")
    return rows, macro_mean, macro_max_over, macro_max_under


def _save_metric_plots(
    rows: list[dict[str, float]],
    doctor_distance_rows: list[dict[str, float]],
) -> dict[str, str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = [str(r["class"]) for r in rows]
    auc_pct_raw = [float(r["auc"]) * 100.0 for r in rows]
    f1_pct = np.asarray([float(r["f1"]) * 100.0 for r in rows], dtype=np.float32)

    auc_pct_plot = np.asarray([
        0.0 if np.isnan(v) else float(v)
        for v in auc_pct_raw
    ], dtype=np.float32)
    auc_nan_mask = np.asarray([np.isnan(v) for v in auc_pct_raw], dtype=bool)

    x = np.arange(len(class_names))

    # Wykres 1: AUC [%] per klasa (zgodny z tabelą).
    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, auc_pct_plot, color="#1f77b4", edgecolor="black")
    for i, bar in enumerate(bars):
        if auc_nan_mask[i]:
            bar.set_hatch("//")
            plt.text(bar.get_x() + bar.get_width() / 2, 2.0, "NaN", ha="center", va="bottom", fontsize=9)
        else:
            value = float(auc_pct_plot[i])
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.axhline(100.0, color="gray", linestyle="--", linewidth=1.2, label="Cel 100%")
    plt.title("AUC per klasa [%]")
    plt.ylabel("AUC [%]")
    plt.xticks(x, class_names, rotation=25, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(AUC_CLASS_PLOT_PATH, dpi=150)
    plt.close()

    # Wykres 2: F1 [%] per klasa (zgodny z tabelą).
    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, f1_pct, color="#ff7f0e", edgecolor="black")
    for bar, value in zip(bars, f1_pct):
        val = float(value)
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.axhline(100.0, color="gray", linestyle="--", linewidth=1.2, label="Cel 100%")
    plt.title("F1 per klasa [%]")
    plt.ylabel("F1 [%]")
    plt.xticks(x, class_names, rotation=25, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(F1_CLASS_PLOT_PATH, dpi=150)
    plt.close()

    # Wykres 3: srednia roznica model vs adnotacja lekarza [pp] per klasa.
    mean_gap_pp = np.asarray([float(r["mean_abs_diff_pp"]) for r in doctor_distance_rows], dtype=np.float32)
    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, mean_gap_pp, color="#2ca02c", edgecolor="black")
    for bar, value in zip(bars, mean_gap_pp):
        val = float(value)
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.title("Srednia roznica model vs lekarz per klasa [pp]")
    plt.ylabel("Srednia |pred - adnotacja| [pp]")
    plt.xticks(x, class_names, rotation=25, ha="right")
    plt.ylim(0, float(max(mean_gap_pp.max() + 5.0, 10.0)))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(MEAN_DOCTOR_GAP_CLASS_PLOT_PATH, dpi=150)
    plt.close()

    # Wykres 4: maksymalna roznica rozdzielona na kierunki [pp] per klasa.
    max_over_pp = np.asarray([float(r["max_overconfident_diff_pp"]) for r in doctor_distance_rows], dtype=np.float32)
    max_under_pp = np.asarray([float(r["max_underconfident_diff_pp"]) for r in doctor_distance_rows], dtype=np.float32)
    w = 0.38
    plt.figure(figsize=(14, 6))
    bars_over = plt.bar(
        x - w / 2,
        max_over_pp,
        width=w,
        color="#d62728",
        edgecolor="black",
        label="Model zbyt pewny (pred > adnotacja)",
    )
    bars_under = plt.bar(
        x + w / 2,
        max_under_pp,
        width=w,
        color="#9467bd",
        edgecolor="black",
        label="Model za malo pewny (pred < adnotacja)",
    )
    for bar, value in zip(bars_over, max_over_pp):
        val = float(value)
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, value in zip(bars_under, max_under_pp):
        val = float(value)
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.title("Maksymalna roznica model vs lekarz per klasa [pp] (kierunki)")
    plt.ylabel("Maks. roznica [pp]")
    plt.xticks(x, class_names, rotation=25, ha="right")
    ymax = float(max(max(max_over_pp.max(), max_under_pp.max()) + 5.0, 20.0))
    plt.ylim(0, ymax)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(MAX_DOCTOR_GAP_SPLIT_CLASS_PLOT_PATH, dpi=150)
    plt.close()

    return {
        "auc_per_class_pct": str(AUC_CLASS_PLOT_PATH),
        "f1_per_class_pct": str(F1_CLASS_PLOT_PATH),
        "mean_doctor_gap_per_class_pp": str(MEAN_DOCTOR_GAP_CLASS_PLOT_PATH),
        "max_doctor_gap_split_per_class_pp": str(MAX_DOCTOR_GAP_SPLIT_CLASS_PLOT_PATH),
    }


def main() -> None:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono wag: {WEIGHTS_PATH}")

    print("[INFO] Start pełnej ewaluacji testowej (AUC + F1).")
    print(f"[INFO] Device: {_cuda_info()}")
    print(f"[INFO] Weights: {WEIGHTS_PATH}")
    print(f"[INFO] Split dir: {SPLIT_DIR}")

    dataset, label_columns = _build_dataset()
    print(f"[INFO] Usable samples: {len(dataset)}")

    device = _resolve_device()
    model, model_device = load_checkpoint_model(WEIGHTS_PATH, num_classes=len(label_columns), device=device)

    if model_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataloader = _build_dataloader(dataset, model_device)
    print(f"[INFO] Batches: {len(dataloader)}")

    y_true, y_prob = _collect_predictions(model, dataloader, model_device)
    y_pred = (y_prob >= THRESHOLD).astype(np.int32)

    rows, macro_auc, macro_f1 = per_class_metrics(y_true, y_prob, y_pred, label_columns)
    doctor_distance_rows, doctor_gap_mean_macro, doctor_gap_max_over_macro, doctor_gap_max_under_macro = _compute_doctor_distance_pp(
        y_true=y_true,
        y_prob=y_prob,
        class_names=label_columns,
    )

    y_true_bin = (y_true > 0.0).astype(np.int32)
    micro_f1 = float(f1_score(y_true_bin.ravel(), y_pred.ravel(), zero_division=0))
    micro_auc = _safe_micro_auc(y_true, y_prob)
    plot_paths = _save_metric_plots(rows, doctor_distance_rows)

    header = f"{'class':45s} {'AUC':>10s} {'F1':>10s}"
    lines = [header, "-" * len(header)]
    for row in rows:
        auc = row["auc"]
        f1 = row["f1"]
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "nan"
        lines.append(f"{str(row['class'])[:45]:45s} {auc_str:>10s} {f1:.4f}")

    lines.append("-" * len(header))
    lines.append(f"{'MACRO':45s} {macro_auc:.4f} {macro_f1:.4f}")
    lines.append(f"{'MICRO':45s} {micro_auc:.4f} {micro_f1:.4f}")
    lines.append(f"{'MEAN |pred-adnotacja| [pp]':45s} {doctor_gap_mean_macro:.4f}")
    lines.append(f"{'MEAN MAX (zbyt pewny) [pp]':45s} {doctor_gap_max_over_macro:.4f}")
    lines.append(f"{'MEAN MAX (za malo pewny) [pp]':45s} {doctor_gap_max_under_macro:.4f}")
    report = "\n".join(lines)

    print("\n" + report)

    payload = {
        "weights": str(WEIGHTS_PATH),
        "split": SPLIT,
        "samples": int(len(dataset)),
        "threshold": THRESHOLD,
        "device": str(model_device),
        "macro_auc": float(macro_auc),
        "macro_f1": float(macro_f1),
        "micro_auc": float(micro_auc),
        "micro_f1": float(micro_f1),
        "per_class": rows,
        "per_class_percent": [
            {
                "class": str(row["class"]),
                "auc_percent": None if np.isnan(float(row["auc"])) else float(row["auc"]) * 100.0,
                "f1_percent": float(row["f1"]) * 100.0,
            }
            for row in rows
        ],
        "per_class_doctor_distance_pp": doctor_distance_rows,
        "doctor_distance_mean_macro_pp": float(doctor_gap_mean_macro),
        "doctor_distance_max_over_macro_pp": float(doctor_gap_max_over_macro),
        "doctor_distance_max_under_macro_pp": float(doctor_gap_max_under_macro),
        "doctor_distance_mean_max_macro_pp_deprecated": float(
            np.mean([float(r["max_abs_diff_pp"]) for r in doctor_distance_rows]) if doctor_distance_rows else float("nan")
        ),
        "per_class_gap_to_100_deprecated": [
            {
                "class": str(row["class"]),
                "auc_gap_percent": None if np.isnan(float(row["auc"])) else 100.0 - float(row["auc"]) * 100.0,
                "f1_gap_percent": 100.0 - float(row["f1"]) * 100.0,
            }
            for row in rows
        ],
        "plots": plot_paths,
    }

    REPORT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TXT_PATH.write_text(report + "\n", encoding="utf-8")
    REPORT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n[OK] Raport TXT: {REPORT_TXT_PATH}")
    print(f"[OK] Raport JSON: {REPORT_JSON_PATH}")
    print(f"[OK] Wykres AUC per klasa [%]: {AUC_CLASS_PLOT_PATH}")
    print(f"[OK] Wykres F1 per klasa [%]: {F1_CLASS_PLOT_PATH}")
    print(f"[OK] Wykres sredniej roznicy model vs lekarz [pp]: {MEAN_DOCTOR_GAP_CLASS_PLOT_PATH}")
    print(f"[OK] Wykres maksymalnej roznicy model vs lekarz [pp] (kierunki): {MAX_DOCTOR_GAP_SPLIT_CLASS_PLOT_PATH}")


if __name__ == "__main__":
    main()



