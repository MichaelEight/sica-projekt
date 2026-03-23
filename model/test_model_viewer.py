from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model.inference_api import load_checkpoint_model, predict_with_model
from model.training.dataset import ECGWFDBDataset, save_json
from model.training.metadata_inspector import inspect_all_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "training"
ANNOTATIONS_DIR = PROJECT_ROOT / "model" / "annotations"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive test viewer: select checkpoint and sample index to compare CSV labels vs prediction."
    )
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint path (.pt). If missing, choose from menu.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary predictions.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split.")
    return parser.parse_args()


def _load_inspection() -> dict[str, object]:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ANNOTATIONS_DIR / "metadata_inspection.json"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    inspection = inspect_all_metadata(DATA_ROOT)
    save_json(cache_path, inspection)
    return inspection


def _choose_checkpoint(weights_arg: str | None) -> Path:
    if weights_arg:
        path = Path(weights_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    candidates = sorted(ANNOTATIONS_DIR.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {ANNOTATIONS_DIR}")

    print("Available checkpoints:")
    for i, ckpt in enumerate(candidates):
        print(f"  [{i}] {ckpt.name}")

    while True:
        raw = input("Choose checkpoint index: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Invalid number. Try again.")
            continue

        if 0 <= idx < len(candidates):
            return candidates[idx]
        print("Index out of range. Try again.")


def _print_prediction_block(
    sample_idx: int,
    row_dict: dict[str, object],
    class_names: list[str],
    true_soft: np.ndarray,
    pred_probs: list[float],
    pred_bins: list[int],
) -> None:
    print("\n" + "=" * 100)
    print(f"Sample index: {sample_idx}")
    print("CSV row:")
    for key, value in row_dict.items():
        print(f"  {key}: {value}")

    print("\nPrediction:")
    print(f"{'class':30s} {'csv_label':>12s} {'prob':>12s} {'pred':>8s}")
    print("-" * 70)
    for i, cls in enumerate(class_names):
        print(f"{cls:30s} {true_soft[i]:12.4f} {pred_probs[i]:12.4f} {pred_bins[i]:8d}")
    print("=" * 100)


def main() -> None:
    args = _parse_args()
    inspection = _load_inspection()

    label_columns = list(inspection["label_columns"])  # type: ignore[index]
    file_columns = dict(inspection["file_columns"])  # type: ignore[index]

    ckpt = _choose_checkpoint(args.weights)
    model, device = load_checkpoint_model(ckpt, num_classes=len(label_columns), device=args.device)
    print(f"[MODEL] Loaded: {ckpt}")
    print(f"[DEVICE] Using: {device}")

    split_dir = DATA_ROOT / args.split
    ds = ECGWFDBDataset(
        split_dir=split_dir,
        metadata_filename=f"{args.split}_metadata.csv",
        label_columns=label_columns,
        file_columns=file_columns,
        target_length=5000,
    )
    print(f"[DATA] Split={args.split} | usable rows={len(ds)}")
    print("Type sample index (0-based) or 'q' to exit.")

    while True:
        raw = input("Sample index> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break

        try:
            idx = int(raw)
        except ValueError:
            print("Invalid number. Try again.")
            continue

        if idx < 0 or idx >= len(ds):
            print(f"Index out of range. Valid range: 0..{len(ds)-1}")
            continue

        x, y = ds[idx]
        result = predict_with_model(
            model=model,
            data=x.unsqueeze(0),
            threshold=args.threshold,
            class_names=label_columns,
            device=device,
        )

        row = ds.df.iloc[idx].to_dict()
        probs = result["probabilities"][0]
        preds = result["predictions"][0]
        _print_prediction_block(
            sample_idx=idx,
            row_dict=row,
            class_names=label_columns,
            true_soft=y.numpy(),
            pred_probs=probs,
            pred_bins=preds,
        )


if __name__ == "__main__":
    main()

