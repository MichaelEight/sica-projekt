from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _binarize_targets(y_true: np.ndarray) -> np.ndarray:
    # Miękkie etykiety [0,1] zamieniamy na obecność klasy dla metryk klasyfikacyjnych.
    return (y_true > 0.0).astype(np.int32)


def safe_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true_bin = _binarize_targets(y_true)
    try:
        return float(roc_auc_score(y_true_bin, y_prob, average="macro"))
    except ValueError:
        # Happens if at least one class has only one label value in the evaluated split.
        per_class = []
        for i in range(y_true_bin.shape[1]):
            cls_true = y_true_bin[:, i]
            if np.unique(cls_true).size < 2:
                continue
            per_class.append(float(roc_auc_score(cls_true, y_prob[:, i])))
        return float(np.mean(per_class)) if per_class else float("nan")


def per_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> tuple[list[dict[str, float]], float, float]:
    y_true_bin = _binarize_targets(y_true)
    rows: list[dict[str, float]] = []

    aucs = []
    f1s = []

    for i, name in enumerate(class_names):
        cls_true = y_true_bin[:, i]
        cls_prob = y_prob[:, i]
        cls_pred = y_pred[:, i]

        if np.unique(cls_true).size < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(cls_true, cls_prob))

        f1 = float(f1_score(cls_true, cls_pred, zero_division=0))
        rows.append({"class": name, "auc": auc, "f1": f1})

        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(f1)

    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return rows, macro_auc, macro_f1


