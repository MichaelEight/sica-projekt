"""Generate a demo 12-lead WFDB record for testing the UI.

Usage:
    python data/generate_demo.py
"""
import numpy as np
import wfdb

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
DURATION = 10.0


def synth_ekg(t, seed=0.0, amp=1.0):
    period = 0.833  # ~72 bpm
    phase = ((t % period) / period) * 2 * np.pi
    v = 0.12 * np.exp(-((phase - 0.9) ** 2) / 0.03)
    v -= 0.08 * np.exp(-((phase - 1.55) ** 2) / 0.004)
    v += (0.9 + seed * 0.08) * np.exp(-((phase - 1.65) ** 2) / 0.005)
    v -= 0.15 * np.exp(-((phase - 1.78) ** 2) / 0.004)
    v += 0.2 * np.exp(-((phase - 2.8) ** 2) / 0.06)
    # Add small noise
    v += np.random.default_rng(int(abs(seed * 1000))).normal(0, 0.02, len(t))
    return v * amp


SEEDS = {"I": 0, "II": 0.5, "III": 1, "aVR": -0.8, "aVL": -0.3, "aVF": 0.8,
         "V1": -0.6, "V2": -0.2, "V3": 0.2, "V4": 0.6, "V5": 0.8, "V6": 0.4}
AMPS = {"I": 1.0, "II": 1.2, "III": 0.8, "aVR": 0.7, "aVL": 0.6, "aVF": 1.0,
        "V1": 0.9, "V2": 1.3, "V3": 1.5, "V4": 1.4, "V5": 1.1, "V6": 0.9}


def main():
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    signal = np.zeros((n, len(LEADS)))

    for i, lead in enumerate(LEADS):
        signal[:, i] = synth_ekg(t, SEEDS[lead], AMPS[lead])

    wfdb.wrsamp(
        "demo_ekg",
        fs=FS,
        units=["mV"] * 12,
        sig_name=LEADS,
        p_signal=signal.astype(np.float64),
        write_dir="data",
    )
    print("Wygenerowano: data/demo_ekg.dat + data/demo_ekg.hea")


if __name__ == "__main__":
    main()
