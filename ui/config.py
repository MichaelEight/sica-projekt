"""User preferences persisted across app runs via QSettings."""
from PySide6.QtCore import QSettings

_ORG = "EKG Assistant"
_APP = "EKG Assistant"

_KEY_DARK_MODE = "ui/dark_mode"


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
