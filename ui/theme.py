"""Color palette and stylesheet matching v2 designs."""

# ── Colors ──────────────────────────────────────
BG = "#f5f6f8"
TOPBAR = "#1a1a2e"
ACCENT = "#4a9eff"
TEXT = "#1a1a2e"
TEXT_MUTED = "#6b7280"
TEXT_DIM = "#9ca3af"
BORDER = "#e5e7eb"
BORDER_LIGHT = "#f3f4f6"
WHITE = "#ffffff"
BTN_DARK = "#2a2a40"
GRID_MINOR = "#FCE4E4"
GRID_MAJOR = "#F0BFBF"
SIGNAL_COLOR = "#1A1A2E"
GREEN = "#059669"
YELLOW = "#d97706"
AMBER_BG = "#fefce8"
AMBER_BORDER = "#fde68a"
AMBER_TEXT = "#92400e"
AMBER_SUB = "#a16207"
PURPLE = "#8b5cf6"
RED = "#e74c3c"

# ── Lead definitions ────────────────────────────
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
LEAD_SEEDS = {"I": 0, "II": 0.5, "III": 0.3, "aVR": -1, "aVL": 0.2, "aVF": 0.6,
              "V1": -0.5, "V2": 0.1, "V3": 0.8, "V4": 1.2, "V5": 1.0, "V6": 0.7}
LEAD_AMPS = {"I": 0.7, "II": 1.0, "III": 0.5, "aVR": -0.6, "aVL": 0.4, "aVF": 0.8,
             "V1": -0.3, "V2": 0.5, "V3": 1.0, "V4": 1.3, "V5": 1.1, "V6": 0.8}

# ── Stylesheet ──────────────────────────────────
STYLESHEET = f"""
QMainWindow, QWidget#centralWidget {{
    background: {BG};
}}
QLabel {{
    color: {TEXT};
    font-family: "Helvetica Neue";
}}
QLabel#muted {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel#dim {{
    color: {TEXT_DIM};
    font-size: 11px;
}}
QLabel#mono {{
    font-family: Menlo;
    font-size: 12px;
    color: {TEXT_MUTED};
}}
QPushButton {{
    font-family: "Helvetica Neue";
    font-size: 12px;
    border: none;
    border-radius: 5px;
    padding: 6px 12px;
    font-weight: 500;
}}
QPushButton#primary {{
    background: {ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: #3a8eef;
}}
QPushButton#secondary {{
    background: {BTN_DARK};
    color: #ccc;
}}
QPushButton#secondary:hover {{
    background: #3a3a50;
}}
QPushButton#toolbar {{
    background: {BTN_DARK};
    color: #ccc;
    padding: 6px 10px;
}}
QPushButton#toolbar:hover {{
    background: #3a3a50;
}}
QPushButton#toolbarActive {{
    background: {ACCENT};
    color: white;
    padding: 6px 10px;
}}
QPushButton#nav {{
    height: 32px;
    padding: 0 10px;
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: white;
    color: #4b5563;
    font-weight: 500;
}}
QPushButton#nav:hover {{
    background: #f9fafb;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
"""
