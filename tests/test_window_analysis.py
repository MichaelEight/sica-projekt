"""Test that window placement on the 20s demo file produces different predictions.

First 10s = NORM, Last 10s = MI (front heart attack).
The model should predict accordingly based on which window is selected.

Usage:
    python tests/test_window_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import wfdb
from model.inference_api import load_checkpoint_model, predict_with_model
from ui.theme import TARGET_CLASSES, CLASS_NAMES_PL

DEMO_RECORD = "data/demo/norm_mi_20s"
MODEL_PATH = "model/annotations/model-sota.pt"
FS = 500


def test_window_analysis():
    # Generate demo file if it doesn't exist
    if not os.path.exists(DEMO_RECORD + ".dat"):
        from data.generate_demo import generate
        generate()

    record = wfdb.rdrecord(DEMO_RECORD)
    signal = record.p_signal.astype(np.float32)
    assert signal.shape == (10000, 12), f"Expected (10000, 12), got {signal.shape}"
    assert record.fs == FS

    model, device = load_checkpoint_model(MODEL_PATH, num_classes=len(TARGET_CLASSES))

    # Window 1: first 10s (0-10s) — should be NORM
    w1 = signal[:5000]
    r1 = predict_with_model(model=model, data=w1, threshold=0.5,
                            class_names=TARGET_CLASSES, device=device)
    p1 = {cls: float(r1["probabilities"][0][i]) for i, cls in enumerate(TARGET_CLASSES)}

    # Window 2: last 10s (10-20s) — should be MI
    w2 = signal[5000:]
    r2 = predict_with_model(model=model, data=w2, threshold=0.5,
                            class_names=TARGET_CLASSES, device=device)
    p2 = {cls: float(r2["probabilities"][0][i]) for i, cls in enumerate(TARGET_CLASSES)}

    print("=" * 70)
    print("Window 1 (0–10s, expected: NORM)")
    print("-" * 70)
    top1 = max(p1, key=p1.get)
    for cls in sorted(p1, key=p1.get, reverse=True):
        marker = " <--" if cls == top1 else ""
        print(f"  {CLASS_NAMES_PL[cls]:50s} {p1[cls]*100:6.1f}%{marker}")

    print()
    print("Window 2 (10–20s, expected: MI / front heart attack)")
    print("-" * 70)
    top2 = max(p2, key=p2.get)
    for cls in sorted(p2, key=p2.get, reverse=True):
        marker = " <--" if cls == top2 else ""
        print(f"  {CLASS_NAMES_PL[cls]:50s} {p2[cls]*100:6.1f}%{marker}")

    print()
    print("=" * 70)

    # Assertions
    assert top1 == "class_healthy", \
        f"Window 1 (NORM): expected class_healthy, got {top1} ({CLASS_NAMES_PL[top1]})"
    assert p1["class_healthy"] > 0.5, \
        f"Window 1 (NORM): healthy confidence too low: {p1['class_healthy']:.3f}"

    assert top2 == "class_front_heart_attack", \
        f"Window 2 (MI): expected class_front_heart_attack, got {top2} ({CLASS_NAMES_PL[top2]})"
    assert p2["class_front_heart_attack"] > 0.3, \
        f"Window 2 (MI): front_heart_attack confidence too low: {p2['class_front_heart_attack']:.3f}"

    print("ALL TESTS PASSED")
    print(f"  Window 1 top: {CLASS_NAMES_PL[top1]} ({p1[top1]*100:.1f}%)")
    print(f"  Window 2 top: {CLASS_NAMES_PL[top2]} ({p2[top2]*100:.1f}%)")


if __name__ == "__main__":
    test_window_analysis()
