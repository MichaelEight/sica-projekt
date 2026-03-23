"""Generate a 20-second demo WFDB file by concatenating two PTB-XL records.

First 10s: NORM (record 00001_hr)
Last 10s:  MI / front heart attack (record 00414_hr)

Also writes a .annotations.json sidecar with per-window ground truth,
so the app can show the correct annotation depending on which 10s window
the doctor selects for analysis.

Usage:
    python data/generate_demo.py
"""
import json
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

    # Write per-window ground truth sidecar
    # Values are 0.0 (absent) or 1.0 (present) — binary, not probabilities
    annotations = {
        "windows": [
            {
                "start": 0.0,
                "end": 10.0,
                "ground_truth": {
                    "class_healthy": 1.0,
                    "class_front_heart_attack": 0.0,
                    "class_side_heart_attack": 0.0,
                    "class_bottom_heart_attack": 0.0,
                    "class_back_heart_attack": 0.0,
                    "class_complete_right_conduction_disorder": 0.0,
                    "class_incomplete_right_conduction_disorder": 0.0,
                    "class_complete_left_conduction_disorder": 0.0,
                },
            },
            {
                "start": 10.0,
                "end": 20.0,
                "ground_truth": {
                    "class_healthy": 0.0,
                    "class_front_heart_attack": 1.0,
                    "class_side_heart_attack": 0.0,
                    "class_bottom_heart_attack": 0.0,
                    "class_back_heart_attack": 0.0,
                    "class_complete_right_conduction_disorder": 0.0,
                    "class_incomplete_right_conduction_disorder": 0.0,
                    "class_complete_left_conduction_disorder": 0.0,
                },
            },
        ]
    }

    json_path = output_path + ".annotations.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Generated: {output_path}.dat + .hea")
    print(f"Generated: {json_path}")
    print(f"  Shape: {signal.shape} ({signal.shape[0] / r1.fs:.1f}s)")
    print(f"  First 10s: NORM (record 00001)")
    print(f"  Last  10s: MI   (record 00414)")
    return output_path


if __name__ == "__main__":
    generate()
