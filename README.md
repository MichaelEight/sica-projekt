# EKG Assistant

A desktop application for viewing and analyzing 12-lead ECG signals using a deep learning model (Inception1D) to detect cardiac abnormalities.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Load a WFDB file (`.dat` + `.hea` pair), explore the signal, and run AI classification.

## Screens

- **12-Lead Grid** — all 12 leads displayed in a standard 4x3 layout
- **Single Lead** — zoomed view of one lead with caliper measurements and annotation tools
- **Monitor** — real-time sweep playback simulating a cardiac monitor

## Detected conditions

| Code  | Condition                     |
| ----- | ----------------------------- |
| NORM  | Normal heart rhythm           |
| MI    | Myocardial infarction         |
| RBBB  | Right bundle branch block     |
| LBBB  | Left bundle branch block      |
| LVH   | Left ventricular hypertrophy  |
| RVH   | Right ventricular hypertrophy |
| ISC\_ | Ischemia                      |
| NST\_ | Non-specific ST changes       |
