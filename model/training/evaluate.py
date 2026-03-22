from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import per_class_metrics


def run_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
    output_path: Path,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    y_true_batches = []
    y_prob_batches = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            probs = model.forward_inference(x)
            y_true_batches.append(y.cpu().numpy())
            y_prob_batches.append(probs.cpu().numpy())

    if not y_true_batches:
        raise ValueError("No batches were evaluated. Increase max_batches or check dataset.")

    y_true = np.concatenate(y_true_batches, axis=0)
    y_prob = np.concatenate(y_prob_batches, axis=0)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    rows, macro_auc, macro_f1 = per_class_metrics(y_true, y_prob, y_pred, class_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    header = f"{'class':50s} {'AUC':>10s} {'F1':>10s}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        class_name = str(r["class"])
        auc_value = float(r["auc"])
        f1_value = float(r["f1"])
        auc_text = f"{auc_value:.4f}" if not np.isnan(auc_value) else "nan"
        lines.append(f"{class_name[:50]:50s} {auc_text:>10s} {f1_value:.4f}")

    lines.append("-" * len(header))
    lines.append(f"{'MACRO':50s} {macro_auc:.4f} {macro_f1:.4f}")

    report = "\n".join(lines)
    print("\n" + report)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(report + "\n")

    return {
        "rows": rows,
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
    }

