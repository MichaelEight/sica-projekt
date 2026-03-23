"""Generate a 20-second demo WFDB file by concatenating two PTB-XL records.

First 10s: NORM (record 00001_hr)
Last 10s:  MI / front heart attack (record 00414_hr)

This allows testing window-based analysis — placing the window on the first half
should predict NORM, placing it on the second half should predict MI.

Usage:
    python data/generate_demo.py
"""
import os
import numpy as np
import wfdb

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "demo")

RECORD_NORM = os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00001_hr")
RECORD_MI = os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00414_hr")


def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    r1 = wfdb.rdrecord(RECORD_NORM)
    r2 = wfdb.rdrecord(RECORD_MI)

    # Concatenate signals: 10s NORM + 10s MI = 20s
    signal = np.vstack([r1.p_signal, r2.p_signal]).astype(np.float64)

    output_path = os.path.join(OUTPUT_DIR, "norm_mi_20s")
    wfdb.wrsamp(
        record_name="norm_mi_20s",
        fs=r1.fs,
        units=r1.units,
        sig_name=r1.sig_name,
        p_signal=signal,
        write_dir=OUTPUT_DIR,
    )

    print(f"Generated: {output_path}.dat + .hea")
    print(f"  Shape: {signal.shape} ({signal.shape[0] / r1.fs:.1f}s)")
    print(f"  First 10s: NORM (record 00001)")
    print(f"  Last  10s: MI   (record 00414)")
    return output_path


if __name__ == "__main__":
    generate()
