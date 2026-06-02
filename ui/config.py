"""User preferences persisted across app runs via QSettings."""
from PySide6.QtCore import QSettings

_ORG = "EKG Assistant"
_APP = "EKG Assistant"

_KEY_DARK_MODE = "ui/dark_mode"
_KEY_T_HIGH = "ai/threshold_high"
_KEY_T_LOW = "ai/threshold_low"

# AI confidence thresholds (probability 0..1) controlling autoscan band colors.
#   illness prob >= T_HIGH        -> red    ("bardzo pewne" — disease)
#   illness prob in [T_LOW, HIGH) -> yellow ("do sprawdzenia" — borderline)
#   model says healthy but its prob < T_HEALTHY -> yellow ("Niepewne")
#   otherwise                     -> no band (healthy / ignored)
# These were previously hardcoded across viewer_page.py / batch.py; now they are
# user-tunable sliders persisted here and applied live by re-filtering the
# cached per-window probabilities (no re-inference needed).
DEFAULT_T_HIGH = 0.70
DEFAULT_T_LOW = 0.40
T_HEALTHY = 0.50  # fixed: how confident "healthy" must be to NOT flag the window


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def get_dark_mode() -> bool:
    val = _settings().value(_KEY_DARK_MODE, False)
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def set_dark_mode(enabled: bool) -> None:
    s = _settings()
    s.setValue(_KEY_DARK_MODE, bool(enabled))
    s.sync()


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _read_float(key: str, default: float) -> float:
    val = _settings().value(key, default)
    try:
        return _clamp01(float(val))
    except (TypeError, ValueError):
        return default


def get_threshold_high() -> float:
    """Red/illness threshold ('bardzo pewne'), default 0.70."""
    return _read_float(_KEY_T_HIGH, DEFAULT_T_HIGH)


def get_threshold_low() -> float:
    """Yellow/borderline threshold ('do sprawdzenia'), default 0.40."""
    return _read_float(_KEY_T_LOW, DEFAULT_T_LOW)


def set_thresholds(t_low: float, t_high: float) -> None:
    """Persist both thresholds, enforcing 0 <= low <= high <= 1."""
    t_low = _clamp01(float(t_low))
    t_high = _clamp01(float(t_high))
    if t_low > t_high:
        t_low, t_high = t_high, t_low
    s = _settings()
    s.setValue(_KEY_T_LOW, t_low)
    s.setValue(_KEY_T_HIGH, t_high)
    s.sync()


def reset_thresholds() -> None:
    set_thresholds(DEFAULT_T_LOW, DEFAULT_T_HIGH)


def classify_window(prob_dict: dict, t_high: float | None = None,
                    t_low: float | None = None,
                    t_healthy: float = T_HEALTHY) -> int:
    """Map a per-class probability dict to an autoscan band code.

    Returns 2 (red/illness), 1 (yellow/borderline or uncertain-healthy) or
    0 (no band). Single source of truth shared by every scan call site.
    When thresholds are omitted, the user's persisted values are used.
    """
    if t_high is None:
        t_high = get_threshold_high()
    if t_low is None:
        t_low = get_threshold_low()
    if not prob_dict:
        return 0
    p_healthy = prob_dict.get("class_healthy", 0.0)
    non_healthy = [p for c, p in prob_dict.items() if c != "class_healthy"]
    p_ill = max(non_healthy) if non_healthy else 0.0
    true_top = max(prob_dict, key=prob_dict.get)
    if p_ill >= t_high:
        return 2
    if p_ill >= t_low:
        return 1
    if true_top == "class_healthy" and p_healthy < t_healthy:
        return 1
    return 0
