"""ECG measurement algorithms for 12-lead EKG signals.

Implements Pan-Tompkins R-peak detection, heart rate, QRS duration,
PR interval, QT/QTc intervals, and electrical axis computation.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------------------------------
# Helper: Bandpass filter
# ---------------------------------------------------------------------------

def _bandpass(signal_1d, lowcut, highcut, fs, order=2):
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal_1d)


# ---------------------------------------------------------------------------
# R-peak detection (simplified Pan-Tompkins)
# ---------------------------------------------------------------------------

def _detect_r_peaks(signal, fs, lead_idx):
    """Detect R-peaks in a single lead using simplified Pan-Tompkins.

    Returns array of sample indices for detected R-peaks.
    """
    sig = signal[:, lead_idx].copy()

    # 1. Bandpass 5-15 Hz
    filtered = _bandpass(sig, 5.0, 15.0, fs)

    # 2. Differentiate
    diff = np.diff(filtered)

    # 3. Square
    squared = diff ** 2

    # 4. Moving window integration (150 ms = 75 samples at 500 Hz)
    win_size = int(0.15 * fs)
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Find peaks
    height_thresh = 0.3 * np.max(integrated)
    min_distance = int(0.4 * fs)  # 400 ms -> max ~150 bpm
    peaks, _ = find_peaks(integrated, distance=min_distance, height=height_thresh)

    return peaks


def _find_r_peaks(signal, fs):
    """Try multiple leads for R-peak detection. Returns peak indices."""
    # Lead priority: II (1), I (0), V5 (10)
    for lead_idx in [1, 0, 10]:
        if lead_idx >= signal.shape[1]:
            continue
        peaks = _detect_r_peaks(signal, fs, lead_idx)
        if len(peaks) >= 3:
            return peaks
    return np.array([], dtype=int)


# ---------------------------------------------------------------------------
# Heart Rate
# ---------------------------------------------------------------------------

def _compute_hr(r_peaks, fs):
    """Compute heart rate from median R-R interval.

    Returns (hr_string, hr_value) tuple.
    """
    if len(r_peaks) < 2:
        return "N/A", None

    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    median_rr = np.median(rr_intervals)

    if median_rr <= 0:
        return "N/A", None

    hr = 60.0 / median_rr

    if hr < 30 or hr > 220:
        return "N/A", None

    return f"{int(round(hr))} bpm", float(hr)


# ---------------------------------------------------------------------------
# QRS onset/offset detection
# ---------------------------------------------------------------------------

def _find_qrs_boundaries(signal_lead, r_peaks, fs):
    """Find QRS onset and offset for each R-peak in a single lead.

    Returns (onsets, offsets) arrays of sample indices.
    """
    max_back = int(0.06 * fs)   # 60 ms
    max_fwd = int(0.08 * fs)    # 80 ms

    onsets = []
    offsets = []

    for rp in r_peaks:
        # --- QRS onset: search backward ---
        start = max(0, rp - max_back)
        segment = signal_lead[start:rp + 1]
        if len(segment) < 3:
            onsets.append(rp - int(0.04 * fs))
            offsets.append(rp + int(0.04 * fs))
            continue

        deriv = np.abs(np.diff(segment))
        max_deriv = np.max(deriv)
        if max_deriv == 0:
            onsets.append(start)
        else:
            threshold = 0.15 * max_deriv
            # Search from the R-peak backwards
            onset_idx = 0
            for i in range(len(deriv) - 1, -1, -1):
                if deriv[i] < threshold:
                    onset_idx = i
                    break
            onsets.append(start + onset_idx)

        # --- QRS offset: search forward ---
        end = min(len(signal_lead), rp + max_fwd + 1)
        segment = signal_lead[rp:end]
        if len(segment) < 3:
            offsets.append(min(rp + int(0.04 * fs), len(signal_lead) - 1))
            continue

        deriv = np.abs(np.diff(segment))
        max_deriv = np.max(deriv)
        if max_deriv == 0:
            offsets.append(rp + len(segment) - 1)
        else:
            threshold = 0.15 * max_deriv
            offset_idx = len(deriv) - 1
            for i in range(len(deriv)):
                if deriv[i] < threshold:
                    offset_idx = i
                    break
            offsets.append(rp + offset_idx)

    return np.array(onsets, dtype=int), np.array(offsets, dtype=int)


# ---------------------------------------------------------------------------
# QRS Duration
# ---------------------------------------------------------------------------

def _compute_qrs(signal, r_peaks, fs, lead_idx=1):
    """Compute QRS duration in ms from median across beats."""
    if len(r_peaks) < 2:
        return "N/A", None, None

    if lead_idx >= signal.shape[1]:
        lead_idx = 0

    sig_lead = signal[:, lead_idx]
    onsets, offsets = _find_qrs_boundaries(sig_lead, r_peaks, fs)

    durations = (offsets - onsets) / fs * 1000.0  # ms
    # Filter valid
    valid = durations[(durations >= 60) & (durations <= 200)]

    if len(valid) == 0:
        return "N/A", onsets, offsets

    median_qrs = np.median(valid)
    return f"{int(round(median_qrs))} ms", onsets, offsets


# ---------------------------------------------------------------------------
# PR Interval
# ---------------------------------------------------------------------------

def _compute_pr(signal, r_peaks, qrs_onsets, fs, lead_idx=1):
    """Compute PR interval from P-wave onset to QRS onset."""
    if qrs_onsets is None or len(qrs_onsets) < 2:
        return "N/A"

    if lead_idx >= signal.shape[1]:
        lead_idx = 0

    sig_lead = signal[:, lead_idx]
    pr_intervals = []

    for i, rp in enumerate(r_peaks):
        if i >= len(qrs_onsets):
            break
        qrs_on = qrs_onsets[i]

        # Window: 200 ms to 50 ms before QRS onset
        win_start = max(0, qrs_on - int(0.200 * fs))
        win_end = max(0, qrs_on - int(0.050 * fs))

        if win_end <= win_start or win_end - win_start < 5:
            continue

        segment = sig_lead[win_start:win_end]

        # P-peak: max positive deflection
        p_peak_rel = np.argmax(segment)
        p_peak_abs = win_start + p_peak_rel

        # P-onset: search backward from P-peak for low-slope point
        search_start = max(0, p_peak_abs - int(0.06 * fs))
        onset_seg = sig_lead[search_start:p_peak_abs + 1]
        if len(onset_seg) < 3:
            continue

        deriv = np.abs(np.diff(onset_seg))
        max_d = np.max(deriv)
        if max_d == 0:
            continue

        threshold = 0.15 * max_d
        p_onset = search_start
        for j in range(len(deriv) - 1, -1, -1):
            if deriv[j] < threshold:
                p_onset = search_start + j
                break

        pr_ms = (qrs_on - p_onset) / fs * 1000.0
        if 80 <= pr_ms <= 400:
            pr_intervals.append(pr_ms)

    if len(pr_intervals) == 0:
        return "N/A"

    median_pr = np.median(pr_intervals)
    return f"{int(round(median_pr))} ms"


# ---------------------------------------------------------------------------
# QT Interval
# ---------------------------------------------------------------------------

def _compute_qt(signal, r_peaks, qrs_onsets, qrs_offsets, fs):
    """Compute QT interval using T-wave end detection."""
    if qrs_onsets is None or qrs_offsets is None or len(qrs_onsets) < 2:
        return "N/A", None

    # Use lead II (index 1) primarily, fallback V5 (index 10)
    lead_idx = 1 if signal.shape[1] > 1 else 0
    sig_lead = signal[:, lead_idx]

    qt_intervals = []

    for i in range(len(r_peaks)):
        if i >= len(qrs_onsets) or i >= len(qrs_offsets):
            break

        qrs_on = qrs_onsets[i]
        qrs_off = qrs_offsets[i]

        # T-wave search window: 100-400 ms after QRS offset
        t_start = qrs_off + int(0.100 * fs)
        t_end = min(len(sig_lead), qrs_off + int(0.400 * fs))

        if t_start >= t_end or t_start >= len(sig_lead):
            continue

        t_segment = sig_lead[t_start:t_end]
        if len(t_segment) < 10:
            continue

        # Find T-peak
        t_peak_rel = np.argmax(t_segment)
        t_peak_abs = t_start + t_peak_rel

        # Find T-end using fallback method: signal returns to within 10% of baseline
        baseline = np.mean(sig_lead[max(0, qrs_on - int(0.05 * fs)):qrs_on])
        t_amplitude = sig_lead[t_peak_abs] - baseline
        if abs(t_amplitude) < 1e-6:
            continue

        threshold = baseline + 0.10 * t_amplitude

        # Search after T-peak for T-end
        search_end = min(len(sig_lead), t_peak_abs + int(0.300 * fs))
        t_end_abs = search_end  # default

        # Try tangent method first
        downslope = sig_lead[t_peak_abs:search_end]
        if len(downslope) > 5:
            deriv = np.diff(downslope)
            if len(deriv) > 0:
                steepest = np.argmin(deriv)  # most negative slope
                slope = deriv[steepest]
                if abs(slope) > 1e-8:
                    # Tangent line: y = slope * (x - steepest) + downslope[steepest]
                    # Find where y = baseline
                    x_intercept = steepest + (baseline - downslope[steepest]) / slope
                    if 0 <= x_intercept < len(downslope):
                        t_end_abs = t_peak_abs + int(x_intercept)

        # Fallback: threshold crossing
        if t_end_abs >= search_end or t_end_abs <= t_peak_abs:
            for j in range(len(downslope)):
                if t_amplitude > 0 and downslope[j] < threshold:
                    t_end_abs = t_peak_abs + j
                    break
                elif t_amplitude < 0 and downslope[j] > threshold:
                    t_end_abs = t_peak_abs + j
                    break

        qt_ms = (t_end_abs - qrs_on) / fs * 1000.0
        if 200 <= qt_ms <= 600:
            qt_intervals.append(qt_ms)

    if len(qt_intervals) == 0:
        return "N/A", None

    median_qt = np.median(qt_intervals)
    return f"{int(round(median_qt))} ms", float(median_qt)


# ---------------------------------------------------------------------------
# QTc Correction
# ---------------------------------------------------------------------------

def _compute_qtc(qt_ms, hr_value, sex):
    """Compute corrected QT using Bazett and Fridericia methods.

    Returns (qtc_string, qtc_value, qtc_method, qtc_status).
    """
    if qt_ms is None or hr_value is None or hr_value <= 0:
        return "N/A", None, "Bazett", "N/A"

    rr_sec = 60.0 / hr_value
    if rr_sec <= 0:
        return "N/A", None, "Bazett", "N/A"

    qt_sec = qt_ms / 1000.0

    # Bazett: QTc = QT / sqrt(RR)
    qtc_bazett = qt_sec / np.sqrt(rr_sec) * 1000.0

    # Fridericia: QTc = QT / cbrt(RR)
    qtc_fridericia = qt_sec / np.cbrt(rr_sec) * 1000.0

    # Use Bazett as default display
    qtc_val = qtc_bazett

    # Determine status using Bazett value
    if sex and str(sex).strip().lower() in ("f", "female", "k", "kobieta"):
        if qtc_val < 340:
            status = "Skrócony"
        elif qtc_val < 460:
            status = "Norma"
        elif qtc_val <= 480:
            status = "Norma"  # borderline, still acceptable
        else:
            status = "Wydłużony"
    else:
        # Male or unknown (more conservative)
        if qtc_val < 340:
            status = "Skrócony"
        elif qtc_val < 450:
            status = "Norma"
        elif qtc_val <= 470:
            status = "Norma"  # borderline
        else:
            status = "Wydłużony"

    return f"{int(round(qtc_val))} ms", float(qtc_val), "Bazett", status


# ---------------------------------------------------------------------------
# Electrical Axis
# ---------------------------------------------------------------------------

def _compute_axis(signal, qrs_onsets, qrs_offsets, fs):
    """Compute electrical axis from net QRS area in leads I and aVF."""
    if qrs_onsets is None or qrs_offsets is None or len(qrs_onsets) < 2:
        return "N/A"

    # Lead I = index 0, aVF = index 5
    if signal.shape[1] < 6:
        return "N/A"

    lead_I = signal[:, 0]
    lead_aVF = signal[:, 5]

    areas_I = []
    areas_aVF = []

    for i in range(len(qrs_onsets)):
        if i >= len(qrs_offsets):
            break
        on = qrs_onsets[i]
        off = qrs_offsets[i]
        if on >= off or off >= len(lead_I):
            continue
        areas_I.append(np.trapz(lead_I[on:off + 1]))
        areas_aVF.append(np.trapz(lead_aVF[on:off + 1]))

    if len(areas_I) == 0:
        return "N/A"

    mean_area_I = np.mean(areas_I)
    mean_area_aVF = np.mean(areas_aVF)

    axis_deg = np.degrees(np.arctan2(mean_area_aVF, mean_area_I))
    axis_deg = round(axis_deg)

    sign = "+" if axis_deg >= 0 else ""
    return f"{sign}{axis_deg}°"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_measurements(signal, fs=500, sex=None):
    """Compute ECG measurements from a 12-lead signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Shape (N, 12) ECG signal. Lead order: I, II, III, aVR, aVL, aVF, V1-V6.
    fs : int
        Sampling frequency in Hz (default 500).
    sex : str or None
        Patient sex for QTc thresholds ('M'/'F' or None).

    Returns
    -------
    dict
        Measurement results with string-formatted values and raw numbers.
    """
    result = {
        "hr": "N/A",
        "pr": "N/A",
        "qrs": "N/A",
        "qt": "N/A",
        "qtc": "N/A",
        "axis": "N/A",
        "hr_value": None,
        "qtc_value": None,
        "qtc_method": "Bazett",
        "qtc_status": "N/A",
    }

    if signal is None or signal.ndim != 2 or signal.shape[0] < fs:
        return result

    try:
        r_peaks = _find_r_peaks(signal, fs)
    except Exception:
        return result

    # Heart Rate
    try:
        result["hr"], result["hr_value"] = _compute_hr(r_peaks, fs)
    except Exception:
        pass

    # QRS Duration
    qrs_onsets = None
    qrs_offsets = None
    try:
        result["qrs"], qrs_onsets, qrs_offsets = _compute_qrs(signal, r_peaks, fs)
    except Exception:
        pass

    # PR Interval
    try:
        result["pr"] = _compute_pr(signal, r_peaks, qrs_onsets, fs)
    except Exception:
        pass

    # QT Interval
    qt_value = None
    try:
        result["qt"], qt_value = _compute_qt(signal, r_peaks, qrs_onsets, qrs_offsets, fs)
    except Exception:
        pass

    # QTc
    try:
        result["qtc"], result["qtc_value"], result["qtc_method"], result["qtc_status"] = (
            _compute_qtc(qt_value, result["hr_value"], sex)
        )
    except Exception:
        pass

    # Axis
    try:
        result["axis"] = _compute_axis(signal, qrs_onsets, qrs_offsets, fs)
    except Exception:
        pass

    return result
