"""Headless batch processing for WFDB records.

Runs the same sliding-window autoscan as ViewerPage but without any Qt
dependencies, so it can safely run in a worker thread. Writes results to
`.ann` sidecars and the autoscan JSON cache, so the viewer can render the
AI markings transparently when the file is opened.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Callable

import numpy as np

from marking_store import Marking, MarkingStore
from ui.theme import TARGET_CLASSES


_LEAD_ALIASES = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3",
    "v4": "V4", "v5": "V5", "v6": "V6",
}


def _normalize_lead_names(names: list[str]) -> list[str]:
    return [_LEAD_ALIASES.get(n.lower(), n) for n in names]


def load_wfdb_record(base_path: str) -> tuple[np.ndarray, list[str], int] | None:
    """Load a WFDB record from a base path (no extension).

    Returns (signal, leads, fs) or None if the record can't be read.
    """
    try:
        import wfdb
    except ImportError:
        return None
    if not (os.path.exists(base_path + ".dat") and os.path.exists(base_path + ".hea")):
        return None
    try:
        record = wfdb.rdrecord(base_path)
        signal = record.p_signal.astype(np.float32)
        leads = _normalize_lead_names(record.sig_name)
        fs = int(record.fs)
        return signal, leads, fs
    except Exception:
        return None


def scan_signal(signal: np.ndarray, fs: int, model, device) -> list[dict]:
    """Run sliding 10s/5s window autoscan on a signal.

    Returns the raw window results (one per window).
    """
    from model.inference_api import predict_with_model

    window_sec = 10.0
    step_sec = 5.0
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    total = signal.shape[0]

    if total < window_samples:
        return []

    starts = list(range(0, total - window_samples + 1, step_samples))
    last_start = total - window_samples
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    results: list[dict] = []
    for s in starts:
        window = signal[s:s + window_samples]
        t_start = s / fs
        t_end = (s + window_samples) / fs
        try:
            res = predict_with_model(
                model=model, data=window, threshold=0.5,
                class_names=TARGET_CLASSES, device=device,
            )
            probs = res["probabilities"][0]
            prob_dict = {cls: float(probs[j]) for j, cls in enumerate(TARGET_CLASSES)}
        except Exception:
            prob_dict = {cls: 0.0 for cls in TARGET_CLASSES}

        top_cls = max(prob_dict, key=prob_dict.get)
        top_prob = prob_dict[top_cls]
        if top_cls == "class_healthy" and top_prob >= 0.5:
            color = 0
        elif top_cls != "class_healthy" and top_prob >= 0.5:
            color = 2
        else:
            color = 1
        results.append({
            "t_start": t_start,
            "t_end": t_end,
            "color": color,
            "probs": prob_dict,
        })
    return results


def merge_regions(raw_windows: list[dict]) -> list[dict]:
    """Collapse overlapping windows into non-overlapping merged regions.

    Priority: red > yellow > none. Healthy segments are dropped. Adjacent
    segments with the same color + top class + similar probability are merged.
    """
    if not raw_windows:
        return []

    bounds = sorted({r["t_start"] for r in raw_windows} | {r["t_end"] for r in raw_windows})
    atoms: list[dict] = []
    prob_tol = 0.10

    def _non_healthy_top(r):
        probs = r.get("probs") or {}
        items = [(c, p) for c, p in probs.items() if c != "class_healthy"]
        if not items:
            return ("", 0.0)
        return max(items, key=lambda x: x[1])

    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 1e-6:
            continue
        mid = 0.5 * (a + b)
        covering = [r for r in raw_windows if r["t_start"] <= mid < r["t_end"]]
        if not covering:
            continue
        max_color = max(r.get("color", 0) for r in covering)
        if max_color == 0:
            continue
        candidates = [r for r in covering if r.get("color", 0) == max_color]
        rep = max(candidates, key=lambda r: _non_healthy_top(r)[1])
        top_cls, top_prob = _non_healthy_top(rep)
        atoms.append({
            "t_start": a,
            "t_end": b,
            "color": max_color,
            "top_cls": top_cls,
            "top_prob": top_prob,
            "probs": rep.get("probs") or {},
        })

    merged: list[dict] = []
    for seg in atoms:
        if merged:
            prev = merged[-1]
            can_merge = (
                abs(prev["t_end"] - seg["t_start"]) < 1e-6
                and prev["color"] == seg["color"]
                and prev["top_cls"] == seg["top_cls"]
                and abs(prev["top_prob"] - seg["top_prob"]) <= prob_tol
            )
            if can_merge:
                prev["t_end"] = seg["t_end"]
                if seg["top_prob"] > prev["top_prob"]:
                    prev["top_prob"] = seg["top_prob"]
                    prev["probs"] = seg["probs"]
                continue
        merged.append(dict(seg))
    return merged


def autoscan_cache_path(base_path: str, model_path: str) -> str:
    file_key = base_path or ""
    model_key = model_path or ""
    key = f"{file_key}:{model_key}"
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(".cache", "autoscan", f"{h}.json")


def write_autoscan_cache(base_path: str, model_path: str, windows: list[dict]) -> None:
    path = autoscan_cache_path(base_path, model_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump({"windows": windows}, f)
    except Exception:
        pass


def save_ai_annotations(base_path: str, merged: list[dict], patient: dict | None) -> None:
    """Write a .ann sidecar containing synthetic scan markings."""
    if not base_path:
        return
    store = MarkingStore()
    for seg in merged:
        m = Marking(
            type="scan",
            lead="all",
            t1=seg["t_start"],
            t2=seg["t_end"],
            probs=seg.get("probs"),
            color_code=seg.get("color", 0),
            source="ai",
        )
        store.add(m)
    store.save_ann(base_path + ".ann", patient=patient)


def _lookup_patient_and_gt(base_path: str, gt_lookup: dict) -> tuple[dict | None, dict | list | None]:
    """Replicate MainWindow._lookup_csv_entry / _lookup_ground_truth
    without any Qt dependencies.
    """
    json_path = base_path + ".annotations.json"
    gt: dict | list | None = None
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                data = json.load(f)
            gt = data.get("windows", None)
        except Exception:
            gt = None

    patient: dict | None = None
    if gt_lookup:
        path = base_path.replace("\\", "/")
        entry = None
        for suffix in ["records500/", "records100/"]:
            idx = path.find(suffix)
            if idx >= 0:
                entry = gt_lookup.get(path[idx:])
                break
        if entry:
            patient = entry.get("patient")
            if gt is None:
                gt = entry.get("ground_truth")
    return patient, gt


def _count_colors(merged: list[dict]) -> tuple[int, int]:
    red = sum(1 for s in merged if s.get("color") == 2)
    yellow = sum(1 for s in merged if s.get("color") == 1)
    return red, yellow


def run_batch(
    base_paths: list[str],
    model,
    device,
    model_path: str,
    gt_lookup: dict,
    progress: dict,
    cancel_flag: threading.Event,
) -> list[dict]:
    """Worker entry point — run autoscan over many records.

    `progress` is a plain dict mutated in place; the caller reads it from
    the main thread via a QTimer. Keys:
        file_idx, file_total, window_idx, window_total, current_name, done

    Returns a list of result rows:
        {base_path, name, duration, red_count, yellow_count, patient, merged}
    """
    results: list[dict] = []
    total = len(base_paths)
    progress["file_total"] = total
    progress["window_total"] = 0
    progress["window_idx"] = 0
    progress["current_name"] = ""
    progress["done"] = False

    for i, base_path in enumerate(base_paths):
        if cancel_flag.is_set():
            break
        progress["file_idx"] = i + 1
        progress["current_name"] = os.path.basename(base_path)
        progress["window_idx"] = 0
        progress["window_total"] = 0

        loaded = load_wfdb_record(base_path)
        if loaded is None:
            continue
        signal, leads, fs = loaded
        duration = signal.shape[0] / fs if fs else 0.0

        windows = _scan_with_progress(signal, fs, model, device, progress, cancel_flag)
        if cancel_flag.is_set():
            break

        merged = merge_regions(windows)
        patient, _gt = _lookup_patient_and_gt(base_path, gt_lookup)

        save_ai_annotations(base_path, merged, patient)
        write_autoscan_cache(base_path, model_path, windows)

        red, yellow = _count_colors(merged)
        results.append({
            "base_path": base_path,
            "name": os.path.basename(base_path),
            "duration": duration,
            "red_count": red,
            "yellow_count": yellow,
            "patient": patient or {},
            "merged": merged,
        })

    progress["done"] = True
    return results


def _scan_with_progress(
    signal: np.ndarray,
    fs: int,
    model,
    device,
    progress: dict,
    cancel_flag: threading.Event,
) -> list[dict]:
    """scan_signal variant that updates `progress` per window."""
    from model.inference_api import predict_with_model

    window_sec = 10.0
    step_sec = 5.0
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    total = signal.shape[0]
    if total < window_samples:
        progress["window_total"] = 0
        return []

    starts = list(range(0, total - window_samples + 1, step_samples))
    last_start = total - window_samples
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    progress["window_total"] = len(starts)
    results: list[dict] = []
    for wi, s in enumerate(starts):
        if cancel_flag.is_set():
            return results
        progress["window_idx"] = wi + 1
        window = signal[s:s + window_samples]
        t_start = s / fs
        t_end = (s + window_samples) / fs
        try:
            res = predict_with_model(
                model=model, data=window, threshold=0.5,
                class_names=TARGET_CLASSES, device=device,
            )
            probs = res["probabilities"][0]
            prob_dict = {cls: float(probs[j]) for j, cls in enumerate(TARGET_CLASSES)}
        except Exception:
            prob_dict = {cls: 0.0 for cls in TARGET_CLASSES}

        top_cls = max(prob_dict, key=prob_dict.get)
        top_prob = prob_dict[top_cls]
        if top_cls == "class_healthy" and top_prob >= 0.5:
            color = 0
        elif top_cls != "class_healthy" and top_prob >= 0.5:
            color = 2
        else:
            color = 1
        results.append({
            "t_start": t_start, "t_end": t_end,
            "color": color, "probs": prob_dict,
        })
    return results
