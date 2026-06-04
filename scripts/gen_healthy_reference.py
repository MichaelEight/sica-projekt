"""One-off generator for the hardcoded healthy 12-lead reference window.

We need a clean 10 s / 500 Hz / 12-lead signal that the model confidently calls
healthy. It is bundled as ``assets/healthy_reference.npy`` and used to fill
missing/deselected leads at analysis time (so the fixed-12-lead model always has
all inputs) instead of the old zero-fill, which confused the model.

Run:  python -m scripts.gen_healthy_reference
"""
from __future__ import annotations

import os

import numpy as np
import wfdb

from model.healthy_reference import CANONICAL_LEADS
from model.inference_api import load_checkpoint_model, predict_with_model
from ui.lead_names import normalize_lead_names
from ui.theme import TARGET_CLASSES

MODEL_PATH = "model/annotations/model-sota.pt"
OUT_PATH = "assets/healthy_reference.npy"

# PTB-XL records labelled NORM (first 10 s, 500 Hz). We pick whichever the model
# calls healthy most confidently.
CANDIDATES = [
    "data/ptb-xl/records500/00000/00001_hr",
    "data/ptb-xl/records500/00000/00002_hr",
    "data/ptb-xl/records500/00000/00005_hr",
    "data/ptb-xl/records500/00000/00006_hr",
    "data/ptb-xl/records500/00000/00007_hr",
    "data/ptb-xl/records500/00000/00009_hr",
]


def _load_canonical(base: str) -> np.ndarray | None:
    """Return (5000, 12) in canonical lead order, or None if unusable."""
    if not (os.path.exists(base + ".dat") and os.path.exists(base + ".hea")):
        return None
    rec = wfdb.rdrecord(base)
    sig = np.asarray(rec.p_signal, dtype=np.float32)  # (N, n_leads)
    leads = normalize_lead_names(rec.sig_name)
    if sig.shape[0] < 5000:
        return None
    out = np.zeros((5000, 12), dtype=np.float32)
    for ci, cl in enumerate(CANONICAL_LEADS):
        if cl in leads:
            out[:, ci] = sig[:5000, leads.index(cl)]
    return out


def main() -> None:
    model, device = load_checkpoint_model(MODEL_PATH, num_classes=len(TARGET_CLASSES))
    best = None  # (healthy_prob, base, window)
    for base in CANDIDATES:
        win = _load_canonical(base)
        if win is None:
            print(f"skip  {base} (missing/short)")
            continue
        res = predict_with_model(
            model=model, data=win, threshold=0.5,
            class_names=TARGET_CLASSES, device=device,
        )
        probs = res["probabilities"][0]
        pd = {c: float(probs[j]) for j, c in enumerate(TARGET_CLASSES)}
        healthy = pd["class_healthy"]
        top = max(pd.items(), key=lambda kv: kv[1])
        print(f"{os.path.basename(base):14s} healthy={healthy:.3f}  top={top[0]}={top[1]:.3f}")
        if best is None or healthy > best[0]:
            best = (healthy, base, win)

    if best is None:
        raise SystemExit("No usable candidate record found.")

    healthy, base, win = best
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, win)
    print(f"\nSaved {OUT_PATH} from {os.path.basename(base)} "
          f"(healthy={healthy:.3f}, shape={win.shape}, dtype={win.dtype})")


if __name__ == "__main__":
    main()
