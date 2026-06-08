from __future__ import annotations

import glob
import os

import numpy as np
import wfdb

from model.healthy_reference import CANONICAL_LEADS
from ui.lead_names import normalize_lead_names

RECORDS_GLOB = "data/ptb-xl/records500/*/*_hr.hea"
OUT_PATH = "assets/lead_medians.npy"
MAX_RECORDS = 300


def _load_canonical(base: str) -> np.ndarray | None:
    rec = wfdb.rdrecord(base)
    sig = np.asarray(rec.p_signal, dtype=np.float32)  # (N, n_leads)
    leads = normalize_lead_names(rec.sig_name)
    if sig.shape[0] < 5000:
        return None
    out = np.full((5000, 12), np.nan, dtype=np.float32)
    for ci, cl in enumerate(CANONICAL_LEADS):
        if cl in leads:
            out[:, ci] = sig[:5000, leads.index(cl)]
    return out


def main() -> None:
    headers = sorted(glob.glob(RECORDS_GLOB))[:MAX_RECORDS]
    if not headers:
        raise SystemExit(f"No records found at {RECORDS_GLOB}")

    cols: list[np.ndarray] = []
    used = 0
    for hea in headers:
        base = hea[:-4]
        win = _load_canonical(base)
        if win is None:
            continue
        cols.append(win)
        used += 1

    stacked = np.concatenate(cols, axis=0)  # (used*5000, 12)
    medians = np.nanmedian(stacked, axis=0).astype(np.float32)  # (12,)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, medians)
    print(f"Saved {OUT_PATH} from {used} records, shape={medians.shape}")
    for cl, m in zip(CANONICAL_LEADS, medians):
        print(f"  {cl:4s} {m:+.5f} mV")


if __name__ == "__main__":
    main()
