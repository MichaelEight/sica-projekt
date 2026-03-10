"""
ECG Viewer — Load and plot ECG from .dat/.hea (WFDB) files.

Usage:
    python ecg_viewer.py path/to/record_name

    (no file extension — just the base name, e.g. python ecg_viewer.py 00888_lr)

Requirements:
    pip install wfdb matplotlib numpy
"""

import sys
import wfdb
import numpy as np
import matplotlib.pyplot as plt


def load_ecg(record_path: str) -> tuple:
    """
    Load ECG record from .dat/.hea files.

    Args:
        record_path: path without extension, e.g. "data/00888_lr"

    Returns:
        signal: numpy array of shape (samples, leads)
        leads:  list of lead names, e.g. ["I", "II", ..., "V6"]
        fs:     sampling frequency in Hz
    """
    record = wfdb.rdrecord(record_path)
    return record.p_signal, record.sig_name, record.fs


def plot_ecg(signal: np.ndarray, leads: list, fs: int, title: str = "ECG Record", save_path: str = None):
    """
    Plot all leads of an ECG signal.

    Args:
        signal:    numpy array of shape (samples, leads)
        leads:     list of lead names
        fs:        sampling frequency in Hz
        title:     plot title
        save_path: if provided, saves the plot to this path instead of showing it
    """
    num_leads = signal.shape[1]
    t = np.arange(signal.shape[0]) / fs

    fig, axes = plt.subplots(num_leads, 1, figsize=(14, num_leads * 1.5), sharex=True)
    fig.suptitle(f"{title} — {num_leads} leads, {fs} Hz, {t[-1]:.1f}s", fontsize=14, fontweight="bold")

    if num_leads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, signal[:, i], color="#1a1a2e", linewidth=0.8)
        ax.set_ylabel(leads[i], fontsize=11, rotation=0, labelpad=30)
        ax.grid(True, alpha=0.3, color="red", linewidth=0.5)
        ax.set_xlim(0, t[-1])
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ecg_viewer.py path/to/record_name")
        print("Example: python ecg_viewer.py 00888_lr")
        sys.exit(1)

    record_path = sys.argv[1]

    # Optional: pass --save filename.png to save instead of showing
    save_path = None
    if "--save" in sys.argv:
        save_idx = sys.argv.index("--save")
        if save_idx + 1 < len(sys.argv):
            save_path = sys.argv[save_idx + 1]
        else:
            save_path = f"{record_path}_ecg.png"

    signal, leads, fs = load_ecg(record_path)

    print(f"Loaded: {record_path}")
    print(f"  Shape: {signal.shape}  ({signal.shape[0]} samples x {signal.shape[1]} leads)")
    print(f"  Leads: {leads}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Duration: {signal.shape[0] / fs:.1f} seconds")

    plot_ecg(signal, leads, fs, title=record_path, save_path=save_path)