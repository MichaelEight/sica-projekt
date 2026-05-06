"""Resample arbitrary-fs ECG signal to fixed 500 Hz / 10 s window."""
from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

TARGET_FS = 500
TARGET_DURATION_S = 10.0
TARGET_SAMPLES = int(TARGET_FS * TARGET_DURATION_S)  # 5000

_log = logging.getLogger("resample")


def resample_to_target(
    signal: np.ndarray, fs_in: float
) -> tuple[np.ndarray, int, str]:
    """Force signal to shape (TARGET_SAMPLES, n_leads) at TARGET_FS.

    Identity (no copy beyond dtype cast) when input is already 500 Hz and
    exactly 5000 samples. Otherwise:
      1. Resample frequency to TARGET_FS via polyphase FIR (resample_poly).
      2. Trim to first TARGET_SAMPLES if longer.
      3. Edge-pad (replicate last sample) if shorter.

    Returns:
        (signal_out, TARGET_FS, status_msg).
        status_msg is empty when nothing changed; otherwise reports each
        transformation with its before/after sample counts.
    """
    if signal.ndim != 2:
        raise ValueError(
            f"expected (n_samples, n_leads) 2D array, got shape {signal.shape}"
        )
    if fs_in <= 0:
        raise ValueError(f"fs_in must be positive, got {fs_in}")

    n_in = signal.shape[0]
    dur_in = n_in / float(fs_in)
    msg_parts: list[str] = []

    # Step 1: frequency adjustment ---------------------------------------
    if abs(fs_in - TARGET_FS) < 1e-6:
        out = signal.astype(np.float32, copy=False)
    else:
        ratio = Fraction(TARGET_FS / float(fs_in)).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
        out = resample_poly(signal, up, down, axis=0).astype(np.float32)
        n_after = out.shape[0]
        delta = n_after - n_in
        direction = "upsample" if fs_in < TARGET_FS else "downsample"
        sign = "+" if delta >= 0 else ""
        msg_parts.append(
            f"{direction} polyphase-FIR {fs_in:g}Hz->{TARGET_FS}Hz "
            f"L/M={up}/{down}, in={n_in}pts({dur_in:.3f}s) "
            f"out={n_after}pts({n_after / TARGET_FS:.3f}s) Δ={sign}{delta}pts"
        )

    # Step 2: length enforcement -----------------------------------------
    n = out.shape[0]
    if n > TARGET_SAMPLES:
        dropped = n - TARGET_SAMPLES
        out = out[:TARGET_SAMPLES].copy()
        msg_parts.append(
            f"trim head-keep {n}->{TARGET_SAMPLES}pts (dropped {dropped})"
        )
    elif n < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - n
        edge = np.repeat(out[-1:], pad, axis=0)
        out = np.concatenate([out, edge], axis=0)
        msg_parts.append(
            f"edge-pad {n}->{TARGET_SAMPLES}pts (added {pad}, input <{TARGET_DURATION_S:g}s)"
        )

    if msg_parts:
        _log.info("resample-window: %s", " | ".join(msg_parts))
    else:
        _log.info(
            "resample-window: passthrough %dHz / %dpts (no change)",
            TARGET_FS, n_in,
        )

    return out, TARGET_FS, " | ".join(msg_parts)
