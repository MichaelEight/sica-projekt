from __future__ import annotations

import os

import numpy as np

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]

_ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_ASSET_PATH = os.path.join(_ASSET_DIR, "healthy_reference.npy")
_MEDIANS_PATH = os.path.join(_ASSET_DIR, "lead_medians.npy")
_REFERENCE: np.ndarray | None = None
_MEDIANS: np.ndarray | None = None


def load_healthy_reference() -> np.ndarray:
    global _REFERENCE
    if _REFERENCE is None:
        try:
            _REFERENCE = np.load(_ASSET_PATH).astype(np.float32, copy=False)
        except Exception:
            _REFERENCE = np.zeros((5000, 12), dtype=np.float32)
    return _REFERENCE


def load_lead_medians() -> np.ndarray:
    """Per-lead population median amplitude (12,), CANONICAL_LEADS order."""
    global _MEDIANS
    if _MEDIANS is None:
        try:
            _MEDIANS = np.load(_MEDIANS_PATH).astype(np.float32, copy=False)
        except Exception:
            _MEDIANS = np.zeros(len(CANONICAL_LEADS), dtype=np.float32)
    return _MEDIANS


def fill_missing_leads(window: np.ndarray, present_leads) -> np.ndarray:
    if present_leads is not None and len(present_leads) >= len(CANONICAL_LEADS):
        return window

    from model.lead_recovery import derive_dependent_limb_leads  # lazy: avoid import cycle

    present = set(present_leads) if present_leads is not None else set()
    out, recovered = derive_dependent_limb_leads(window, present)
    med = load_lead_medians()
    for ci, cl in enumerate(CANONICAL_LEADS):
        if cl not in recovered:
            out[:, ci] = med[ci]
    return out


def filter_lead_importance(importance: dict | None, present_leads) -> dict | None:
    if not importance or present_leads is None:
        return importance
    kept = {k: v for k, v in importance.items() if k in present_leads}
    total = sum(kept.values())
    if total > 0:
        kept = {k: round(v / total * 100.0, 2) for k, v in kept.items()}
    return kept
