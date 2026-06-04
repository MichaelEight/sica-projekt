from __future__ import annotations

import os

import numpy as np

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]

_ASSET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "healthy_reference.npy")
_REFERENCE: np.ndarray | None = None


def load_healthy_reference() -> np.ndarray:
    global _REFERENCE
    if _REFERENCE is None:
        try:
            _REFERENCE = np.load(_ASSET_PATH).astype(np.float32, copy=False)
        except Exception:
            _REFERENCE = np.zeros((5000, 12), dtype=np.float32)
    return _REFERENCE


def fill_missing_leads(window: np.ndarray, present_leads) -> np.ndarray:
    if present_leads is not None and len(present_leads) >= len(CANONICAL_LEADS):
        return window
    ref = load_healthy_reference()
    n = window.shape[0]
    if ref.shape[0] != n:
        reps = -(-n // ref.shape[0])
        ref = np.tile(ref, (reps, 1))[:n]
    out = window.copy()
    for ci, cl in enumerate(CANONICAL_LEADS):
        if present_leads is None or cl not in present_leads:
            out[:, ci] = ref[:, ci]
    return out


def filter_lead_importance(importance: dict | None, present_leads) -> dict | None:
    if not importance or present_leads is None:
        return importance
    kept = {k: v for k, v in importance.items() if k in present_leads}
    total = sum(kept.values())
    if total > 0:
        kept = {k: round(v / total * 100.0, 2) for k, v in kept.items()}
    return kept
