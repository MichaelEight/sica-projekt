"""Color palette and stylesheet for light and dark mode."""

_dark_mode = False


def is_dark_mode() -> bool:
    return _dark_mode


def set_dark_mode(dark: bool):
    global _dark_mode
    global BG, BG_SECONDARY, TOPBAR, ACCENT, ACCENT_TEXT, TEXT, TEXT_SECONDARY
    global TEXT_MUTED, TEXT_DIM, BORDER, BORDER_LIGHT, WHITE, BTN_DARK, BTN_TEXT
    global GRID_MINOR, GRID_MAJOR, SIGNAL_COLOR
    global GREEN, YELLOW, AMBER_BG, AMBER_BORDER, AMBER_TEXT, AMBER_SUB
    global PURPLE, RED, SEPARATOR, BADGE_NORM_BG, BADGE_NORM_TEXT
    global BADGE_WARN_BG, BADGE_WARN_TEXT, TAG_BG, ICON_BG
    global GREEN_BG, GREEN_BORDER, BLUE_BG, PURPLE_BG
    global BADGE_BLUE_BG, BADGE_BLUE_TEXT, AMBER, BAR_BG, BORDER_DASHED
    global STYLESHEET

    _dark_mode = dark
    if dark:
        BG = "#0a0a0a"
        BG_SECONDARY = "#111111"
        TOPBAR = "#111111"
        ACCENT = "#00e676"
        ACCENT_TEXT = "#000000"       # black text on bright green
        TEXT = "#e0e0e0"
        TEXT_SECONDARY = "#b0b0b0"
        TEXT_MUTED = "#9e9e9e"
        TEXT_DIM = "#616161"
        BORDER = "#2a2a2a"
        BORDER_LIGHT = "#1e1e1e"
        WHITE = "#141414"
        BTN_DARK = "#1e1e1e"
        BTN_TEXT = "#cccccc"
        GRID_MINOR = "#1a2e1a"
        GRID_MAJOR = "#2a4a2a"
        SIGNAL_COLOR = "#00e676"
        GREEN = "#00e676"
        YELLOW = "#ffd600"
        AMBER_BG = "#1a1800"
        AMBER_BORDER = "#4a4000"
        AMBER_TEXT = "#ffd600"
        AMBER_SUB = "#ccaa00"
        PURPLE = "#bb86fc"
        RED = "#ff5252"
        SEPARATOR = "#3a3a3a"
        BADGE_NORM_BG = "#0a2e1a"
        BADGE_NORM_TEXT = "#00e676"
        BADGE_WARN_BG = "#2e2a0a"
        BADGE_WARN_TEXT = "#ffd600"
        TAG_BG = "#1e1e1e"
        ICON_BG = "#0a1e0a"
        GREEN_BG = "#0a2e1a"
        GREEN_BORDER = "#1a4a2a"
        BLUE_BG = "#0a1e3a"
        PURPLE_BG = "#1a0a2e"
        BADGE_BLUE_BG = "#1e3a5f"
        BADGE_BLUE_TEXT = "#93c5fd"
        AMBER = "#f59e0b"
        BAR_BG = "#3a3a3a"
        BORDER_DASHED = "#3a3a3a"
    else:
        BG = "#f5f6f8"
        BG_SECONDARY = "#f9fafb"
        TOPBAR = "#1a1a2e"
        ACCENT = "#4a9eff"
        ACCENT_TEXT = "#ffffff"       # white text on blue
        TEXT = "#1a1a2e"
        TEXT_SECONDARY = "#4b5563"
        TEXT_MUTED = "#6b7280"
        TEXT_DIM = "#9ca3af"
        BORDER = "#e5e7eb"
        BORDER_LIGHT = "#f3f4f6"
        WHITE = "#ffffff"
        BTN_DARK = "#2a2a40"
        BTN_TEXT = "#cccccc"
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
        SEPARATOR = "#6b7280"
        BADGE_NORM_BG = "#d1fae5"
        BADGE_NORM_TEXT = "#065f46"
        BADGE_WARN_BG = "#fef3c7"
        BADGE_WARN_TEXT = "#92400e"
        TAG_BG = "#f0f1f3"
        ICON_BG = "#eef4ff"
        GREEN_BG = "#f0fdf4"
        GREEN_BORDER = "#bbf7d0"
        BLUE_BG = "#eff6ff"
        PURPLE_BG = "#f5f3ff"
        BADGE_BLUE_BG = "#dbeafe"
        BADGE_BLUE_TEXT = "#1e40af"
        AMBER = "#f59e0b"
        BAR_BG = "#d1d5db"
        BORDER_DASHED = "#c5cad3"

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
    color: {TEXT};
}}
QPushButton#primary {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: {'#00c864' if dark else '#3a8eef'};
}}
QPushButton#secondary {{
    background: {BTN_DARK};
    color: {BTN_TEXT};
}}
QPushButton#secondary:hover {{
    background: {'#333333' if dark else '#4a4a60'};
    color: {'#ffffff' if dark else '#ffffff'};
}}
QPushButton#toolbar {{
    background: {BTN_DARK};
    color: {BTN_TEXT};
    padding: 6px 10px;
}}
QPushButton#toolbar:hover {{
    background: {'#2a2a2a' if dark else '#3a3a50'};
}}
QPushButton#toolbarActive {{
    background: {ACCENT};
    color: {ACCENT_TEXT};
    padding: 6px 10px;
}}
QPushButton#nav {{
    height: 32px;
    padding: 0 10px;
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {WHITE};
    color: {TEXT_MUTED};
    font-weight: 500;
}}
QPushButton#nav:hover {{
    background: {BORDER_LIGHT};
    border: 1px solid {TEXT_DIM};
    color: {TEXT};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QComboBox {{
    color: {TEXT};
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}
QComboBox QAbstractItemView {{
    color: {TEXT};
    background: {WHITE};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}
QTextEdit {{
    color: {TEXT};
    background: {WHITE};
}}
"""


set_dark_mode(False)

# AI model class definitions
TARGET_CLASSES = [
    "class_healthy",
    "class_front_heart_attack",
    "class_side_heart_attack",
    "class_bottom_heart_attack",
    "class_back_heart_attack",
    "class_complete_right_conduction_disorder",
    "class_incomplete_right_conduction_disorder",
    "class_complete_left_conduction_disorder",
]

CLASS_NAMES_PL = {
    "class_healthy": "Zdrowy (NORM)",
    "class_front_heart_attack": "Zawał przedniej ściany",
    "class_side_heart_attack": "Zawał ściany bocznej",
    "class_bottom_heart_attack": "Zawał ściany dolnej",
    "class_back_heart_attack": "Zawał ściany tylnej",
    "class_complete_right_conduction_disorder": "Całkowity blok prawej odnogi (CRBBB)",
    "class_incomplete_right_conduction_disorder": "Niepełny blok prawej odnogi (IRBBB)",
    "class_complete_left_conduction_disorder": "Całkowity blok lewej odnogi (CLBBB)",
}

# Lead definitions
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
LEAD_SEEDS = {"I": 0, "II": 0.5, "III": 0.3, "aVR": -1, "aVL": 0.2, "aVF": 0.6,
              "V1": -0.5, "V2": 0.1, "V3": 0.8, "V4": 1.2, "V5": 1.0, "V6": 0.7}
LEAD_AMPS = {"I": 0.7, "II": 1.0, "III": 0.5, "aVR": -0.6, "aVL": 0.4, "aVF": 0.8,
             "V1": -0.3, "V2": 0.5, "V3": 1.0, "V4": 1.3, "V5": 1.1, "V6": 0.8}
