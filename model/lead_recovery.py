"""Exact reconstruction of dependent limb leads (no model change).

A 12-lead ECG has only 8 independent signals. The 6 limb leads carry just 2
degrees of freedom: given any two of {I, II, III}, the remaining limb leads are
exact linear combinations (Einthoven + Goldberger):

    III = II - I
    aVR = -(I + II) / 2
    aVL =  I - II / 2
    aVF =  II - I / 2

So a missing limb lead should never be zero-/healthy-filled when >=2 limb leads
are present -- it can be recovered exactly. Precordial leads (V1..V6) are NOT
recoverable this way; they remain genuinely missing.

Operates on (N, 12) windows in CANONICAL_LEADS order (matches fill_missing_leads).
"""
from __future__ import annotations

import numpy as np

from model.healthy_reference import CANONICAL_LEADS

_LIMB = ("I", "II", "III")
_IDX = {l: i for i, l in enumerate(CANONICAL_LEADS)}


def derive_dependent_limb_leads(window: np.ndarray, present_leads):
    """Fill exactly-recoverable limb leads from present ones.

    Args:
        window: (N, 12) signal in canonical lead order.
        present_leads: iterable of canonical lead names actually measured
            (None -> nothing present).

    Returns:
        (out, recovered): out is a copy with derivable limb leads filled;
        recovered is the set of leads now known (measured + reconstructed).
        Precordial leads are untouched.
    """
    present = set(present_leads) if present_leads is not None else set()
    out = window.copy()
    have = [l for l in _LIMB if l in present]
    if len(have) < 2:
        return out, present  # < 2 limb leads -> frontal plane underdetermined

    rec = set(present)
    I = out[:, _IDX["I"]] if "I" in present else None
    II = out[:, _IDX["II"]] if "II" in present else None
    III = out[:, _IDX["III"]] if "III" in present else None

    if I is None:      # have II, III
        I = II - III
        out[:, _IDX["I"]] = I
        rec.add("I")
    if II is None:     # have I, III
        II = I + III
        out[:, _IDX["II"]] = II
        rec.add("II")
    if III is None:    # have I, II
        III = II - I
        out[:, _IDX["III"]] = III
        rec.add("III")

    if "aVR" not in present:
        out[:, _IDX["aVR"]] = -(I + II) / 2.0
        rec.add("aVR")
    if "aVL" not in present:
        out[:, _IDX["aVL"]] = I - II / 2.0
        rec.add("aVL")
    if "aVF" not in present:
        out[:, _IDX["aVF"]] = II - I / 2.0
        rec.add("aVF")

    return out, rec
