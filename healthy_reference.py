"""Hardcoded healthy 12-lead reference, used to fill missing leads for the model.

The classifier is fixed to 12 leads. Records may lack leads (never recorded, or
the user deselected them for a "selected leads only" analysis). The old fix was
zero-filling the gaps, which the model reads as a flatline and misinterprets.

Instead we fill each missing lead with the matching lead from a real patient the
model confidently calls healthy (``assets/healthy_reference.npy``, a 5000-sample
/ 500 Hz / 12-lead window in canonical order). These filler leads exist ONLY so
the model has all 12 inputs:
  * they are never displayed,
  * they are excluded from the XAI lead-importance readout.

Qt-free on purpose so both the viewer and the headless batch path can use it.
``CANONICAL_LEADS`` mirrors ``ui.ekg_canvas.ALL_LEADS_ORDER``.
"""
from __future__ import annotations

import os

import numpy as np

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]

_ASSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "assets", "healthy_reference.npy")
_REFERENCE: np.ndarray | None = None


def load_healthy_reference() -> np.ndarray:
    """Return the bundled healthy window as (5000, 12) float32, canonical order.

    Cached after first load. Falls back to zeros if the asset is missing so a
    broken install degrades to the old behaviour rather than crashing.
    """
    global _REFERENCE
    if _REFERENCE is None:
        try:
            _REFERENCE = np.load(_ASSET_PATH).astype(np.float32, copy=False)
        except Exception:
            _REFERENCE = np.zeros((5000, 12), dtype=np.float32)
    return _REFERENCE


def fill_missing_leads(window: np.ndarray, present_leads) -> np.ndarray:
    """Replace every canonical lead NOT in ``present_leads`` with the healthy one.

    ``window`` is a (N, 12) array in canonical lead order (the post-resample
    model window). Columns for present leads are left untouched; missing ones are
    overwritten with the healthy reference (length-matched by trim/tile). Returns
    a copy; the input is not mutated.
    """
    if present_leads is not None and len(present_leads) >= len(CANONICAL_LEADS):
        return window  # nothing missing — avoid the copy
    ref = load_healthy_reference()
    n = window.shape[0]
    if ref.shape[0] != n:
        reps = -(-n // ref.shape[0])  # ceil-div tile, then trim
        ref = np.tile(ref, (reps, 1))[:n]
    out = window.copy()
    for ci, cl in enumerate(CANONICAL_LEADS):
        if present_leads is None or cl not in present_leads:
            out[:, ci] = ref[:, ci]
    return out


def filter_lead_importance(importance: dict | None, present_leads) -> dict | None:
    """Drop filler leads from an XAI lead-importance dict and renormalise to 100%.

    The model attributes importance across all 12 leads, but the filler leads are
    synthetic, so we only ever surface importance for leads the patient actually
    has. Remaining percentages are rescaled to sum to 100.
    """
    if not importance or present_leads is None:
        return importance
    kept = {k: v for k, v in importance.items() if k in present_leads}
    total = sum(kept.values())
    if total > 0:
        kept = {k: round(v / total * 100.0, 2) for k, v in kept.items()}
    return kept
