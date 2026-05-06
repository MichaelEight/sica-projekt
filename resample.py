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
        direction = "upsampled" if fs_in < TARGET_FS else "downsampled"
        sign = "+" if delta >= 0 else ""
        msg_parts.append(
            f"{direction} {fs_in:g}Hz->{TARGET_FS}Hz "
            f"(ratio {up}/{down}, {n_in} -> {n_after} pts, {sign}{delta})"
        )
        _log.info(
            "resample: %s from %g Hz to %d Hz, ratio=%d/%d, %d -> %d samples (delta %s%d)",
            direction, fs_in, TARGET_FS, up, down, n_in, n_after, sign, delta,
        )

    # Step 2: length enforcement -----------------------------------------
    n = out.shape[0]
    if n > TARGET_SAMPLES:
        dropped = n - TARGET_SAMPLES
        out = out[:TARGET_SAMPLES].copy()
        msg_parts.append(f"trimmed {dropped} pts ({n} -> {TARGET_SAMPLES})")
        _log.info("resample: trimmed %d samples (%d -> %d)", dropped, n, TARGET_SAMPLES)
    elif n < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - n
        edge = np.repeat(out[-1:], pad, axis=0)
        out = np.concatenate([out, edge], axis=0)
        msg_parts.append(
            f"edge-padded {pad} pts ({n} -> {TARGET_SAMPLES}, input <{TARGET_DURATION_S:g}s)"
        )
        _log.warning(
            "resample: edge-padded %d samples (%d -> %d). Input shorter than %g s.",
            pad, n, TARGET_SAMPLES, TARGET_DURATION_S,
        )

    if not msg_parts:
        _log.info(
            "resample: passthrough — input already %d Hz / %d samples (no change)",
            TARGET_FS, n_in,
        )

    return out, TARGET_FS, "; ".join(msg_parts)
