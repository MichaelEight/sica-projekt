"""Generate demo WFDB files by concatenating PTB-XL records.

Short demo (20s):
    First 10s: NORM (record 00001_hr)
    Last 10s:  MI / front heart attack (record 00414_hr)

Long demo (90s):
    9 segments x 10s = 90s
    Pattern: NORM, NORM, ILL, NORM, NORM, NORM, ILL, NORM, NORM
    ~78% healthy, no adjacent pathological segments.

Also writes .annotations.json sidecars with per-window ground truth,
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

# Additional NORM records for the long demo
RECORDS_NORM_LONG = [
    os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00001_hr"),
    os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00002_hr"),
    os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00003_hr"),
    os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00005_hr"),
    os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00006_hr"),
    os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00007_hr"),
    os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00009_hr"),
]

# Pathological records for the long demo
RECORD_MI_FRONT = os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00414_hr")  # INJAS/INJAL
RECORD_MI_BOTTOM = os.path.join(DATA_DIR, "ptb-xl", "records500", "00000", "00008_hr")  # IMI

# Ground-truth template: all zeros
_GT_ZEROS = {
    "class_healthy": 0.0,
    "class_front_heart_attack": 0.0,
    "class_side_heart_attack": 0.0,
    "class_bottom_heart_attack": 0.0,
    "class_back_heart_attack": 0.0,
    "class_complete_right_conduction_disorder": 0.0,
    "class_incomplete_right_conduction_disorder": 0.0,
    "class_complete_left_conduction_disorder": 0.0,
}


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


def generate_long():
    """Generate a ~90s demo WFDB file with 9 segments (7 NORM + 2 pathological).

    Pattern: NORM, NORM, ILL, NORM, NORM, NORM, ILL, NORM, NORM
    Segment 2 (idx 2): front heart attack (record 00414)
    Segment 6 (idx 6): bottom heart attack (record 00008 / IMI)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Indices into RECORDS_NORM_LONG for the 7 healthy segments
    norm_records = [wfdb.rdrecord(p) for p in RECORDS_NORM_LONG]
    ill_front = wfdb.rdrecord(RECORD_MI_FRONT)
    ill_bottom = wfdb.rdrecord(RECORD_MI_BOTTOM)

    # Segment order: N N ILL N N N ILL N N
    # Map: index -> (record, label)
    # Healthy slots use norm_records[0..6] in order
    norm_idx = 0
    segments = []
    labels = []
    for i in range(9):
        if i == 2:
            segments.append(ill_front.p_signal)
            gt = dict(_GT_ZEROS)
            gt["class_front_heart_attack"] = 1.0
            labels.append(gt)
        elif i == 6:
            segments.append(ill_bottom.p_signal)
            gt = dict(_GT_ZEROS)
            gt["class_bottom_heart_attack"] = 1.0
            labels.append(gt)
        else:
            segments.append(norm_records[norm_idx].p_signal)
            gt = dict(_GT_ZEROS)
            gt["class_healthy"] = 1.0
            labels.append(gt)
            norm_idx += 1

    signal = np.vstack(segments).astype(np.float64)

    ref_record = norm_records[0]
    record_name = "long_demo_90s"
    output_path = os.path.join(OUTPUT_DIR, record_name)
    wfdb.wrsamp(
        record_name=record_name,
        fs=ref_record.fs,
        units=ref_record.units,
        sig_name=ref_record.sig_name,
        p_signal=signal,
        write_dir=OUTPUT_DIR,
    )

    # Build per-window annotations
    annotations = {"windows": []}
    for i in range(9):
        annotations["windows"].append({
            "start": float(i * 10),
            "end": float((i + 1) * 10),
            "ground_truth": labels[i],
        })

    json_path = output_path + ".annotations.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Generated: {output_path}.dat + .hea")
    print(f"Generated: {json_path}")
    print(f"  Shape: {signal.shape} ({signal.shape[0] / ref_record.fs:.1f}s)")
    print(f"  Pattern: NORM, NORM, ILL(front), NORM, NORM, NORM, ILL(bottom), NORM, NORM")
    print(f"  Healthy segments: 7/9 ({7/9*100:.0f}%)")
    return output_path


if __name__ == "__main__":
    generate()
    generate_long()
