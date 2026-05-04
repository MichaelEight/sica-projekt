from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from model.training.train_pipeline import (
    RUNS_DIR,
    _build_run_name,
    _derive_columns_fast,
    _load_or_refresh_metadata_inspection,
    _run_training,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = PROJECT_ROOT / "model" / "annotations"
LOCAL_SEARCH_DIR = ANNOTATIONS_DIR / "local_search"


def _parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(val.strip()) for val in raw.split(",") if val.strip()]


def _parse_kernel_sizes(raw: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in raw.replace("x", "-").replace(":", "-").split("-") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Kernel sizes musza miec format np. 9-19-39")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _load_base_params(base_run_path: Path) -> tuple[int, tuple[int, int, int]]:
    with base_run_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    width = int(payload.get("nb_filters", 32))
    kernel_sizes = tuple(payload.get("kernel_sizes", [9, 19, 39]))
    return width, (int(kernel_sizes[0]), int(kernel_sizes[1]), int(kernel_sizes[2]))


def _build_candidates(base_value: int, deltas: list[int], include_base: bool, min_value: int) -> list[int]:
    values = []
    if include_base:
        values.append(base_value)
    for delta in deltas:
        values.append(base_value + delta)
    values = [v for v in values if v >= min_value]
    return sorted(set(values))


def _build_kernel_candidates(
    base_kernels: tuple[int, int, int],
    deltas: list[int],
    include_base: bool,
) -> list[tuple[int, int, int]]:
    candidates = []
    if include_base:
        candidates.append(base_kernels)
    for delta in deltas:
        k1, k2, k3 = base_kernels
        k1 = max(3, k1 + delta)
        k2 = max(3, k2 + delta)
        k3 = max(3, k3 + delta)
        candidates.append((k1, k2, k3))
    return sorted(set(candidates))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local search around best hyperparameters (focal loss only)",
    )
    parser.add_argument("--base-run", type=str, default="", help="Sciezka do best_run.json lub run_config.json")
    parser.add_argument("--base-width", type=int, default=32, help="Bazowa szerokosc (nb_filters)")
    parser.add_argument("--base-kernels", type=str, default="9-19-39", help="Bazowe kernle, np. 9-19-39")
    parser.add_argument("--width-deltas", type=str, default="-8,-4,4", help="5 zmian szerokosci")
    parser.add_argument("--kernel-deltas", type=str, default="-4,-2,2", help="5 zmian kerneli")
    parser.add_argument("--include-base", action="store_true", help="Uwzglednij wartosci bazowe")
    parser.add_argument("--max-epochs", type=int, default=200, help="Maksymalna liczba epok")
    parser.add_argument("--patience", type=int, default=7, help="Patience na val_auc")
    parser.add_argument("--num-workers", type=int, default=4, help="Liczba workerow DataLoadera")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--val-freq", type=int, default=1)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=1)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--sanity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] Karta graficzna: {gpu_name} | Pamiec: {gpu_memory:.2f} GB")
    else:
        print("[GPU] Brak karty graficznej - bedzie uzyty CPU")

    base_run = None
    if args.base_run:
        base_run = Path(args.base_run)
    else:
        candidate = RUNS_DIR / "best_run.json"
        if candidate.exists():
            base_run = candidate

    if base_run and base_run.exists():
        base_width, base_kernels = _load_base_params(base_run)
        print(f"[BASE] from {base_run}: width={base_width}, kernels={base_kernels}")
    else:
        base_width = args.base_width
        base_kernels = _parse_kernel_sizes(args.base_kernels)
        print(f"[BASE] from args: width={base_width}, kernels={base_kernels}")

    width_deltas = _parse_int_list(args.width_deltas)
    kernel_deltas = _parse_int_list(args.kernel_deltas)
    widths = _build_candidates(base_width, width_deltas, args.include_base, min_value=8)
    kernel_sets = _build_kernel_candidates(base_kernels, kernel_deltas, args.include_base)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = LOCAL_SEARCH_DIR / f"local_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    if args.sanity:
        label_columns, file_columns = _derive_columns_fast()
    else:
        inspection = _load_or_refresh_metadata_inspection(ANNOTATIONS_DIR)
        label_columns = inspection["label_columns"]  # type: ignore[assignment]
        file_columns = inspection["file_columns"]  # type: ignore[assignment]

    results: list[dict[str, object]] = []

    for idx, (width, kernels) in enumerate(((w, k) for w in widths for k in kernel_sets), start=1):
        run_name = _build_run_name(width, kernels, "focal", f"{timestamp}_{idx:02d}", prefix="local")
        output_dir = batch_dir / run_name

        run_args = argparse.Namespace(
            max_epochs=args.max_epochs,
            patience=args.patience,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            skip_test_eval=False,
            checkpoint_freq=args.checkpoint_freq,
            log_freq=args.log_freq,
            val_freq=args.val_freq,
            skip_plots=args.skip_plots,
            num_workers=args.num_workers,
            sanity=args.sanity,
            resume=None,
            loss="focal",
        )

        print(f"[RUN] {run_name} | width={width} | kernels={kernels} | loss=focal")

        result = _run_training(
            output_dir=output_dir,
            label_columns=label_columns,
            file_columns=file_columns,
            device=device,
            args=run_args,
            nb_filters=width,
            kernel_sizes=kernels,
            loss_type="focal",
            run_name=run_name,
            allow_resume=False,
        )

        result["nb_filters"] = width
        result["kernel_sizes"] = list(kernels)
        result["loss"] = "focal"
        results.append(result)

    if not results:
        print("[WARN] Brak wynikow do podsumowania.")
        return

    def _rank_key(item: dict[str, object]) -> float:
        value = item.get("test_macro_auc")
        if value is None:
            return float("-inf")
        try:
            val = float(value)
        except (TypeError, ValueError):
            return float("-inf")
        if not np.isfinite(val):
            return float("-inf")
        return val

    results_sorted = sorted(results, key=_rank_key, reverse=True)
    best = results_sorted[0]

    summary_path = batch_dir / "local_search_summary.json"
    summary_path.write_text(json.dumps({"results": results_sorted}, indent=2), encoding="utf-8")

    ranking_path = batch_dir / "ranking_test_auc.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as f:
        f.write("run_name,output_dir,nb_filters,kernel_sizes,test_macro_auc\n")
        for item in results_sorted:
            f.write(
                f"{item.get('run_name')},{item.get('output_dir')},{item.get('nb_filters')},"
                f"{item.get('kernel_sizes')},{float(item.get('test_macro_auc', float('nan'))):.6f}\n"
            )

    print(
        "\n[BEST LOCAL] "
        f"name={best.get('run_name')} | test_auc={float(best.get('test_macro_auc', float('nan'))):.4f} | "
        f"dir={best.get('output_dir')}"
    )
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] Ranking: {ranking_path}")


if __name__ == "__main__":
    main()

