"""Kardioskop — desktop EKG analysis application.

Usage:
    python main.py
"""
import sys
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle
from PySide6.QtGui import QFont

from ui.app_logger import get_logger
from ui.main_window import MainWindow


class _FastTooltipStyle(QProxyStyle):
    """Shorten Qt's ~700 ms tooltip wake-up delay so hints appear quickly."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return 250
        if hint == QStyle.SH_ToolTip_FallAsleepDelay:
            return 200
        return super().styleHint(hint, option, widget, returnData)


def main():
    log = get_logger("main")
    log.info("starting Kardioskop")

    app = QApplication(sys.argv)
    app.setApplicationName("Kardioskop")
    app.setOrganizationName("Kardioskop")
    app.setStyle(_FastTooltipStyle())
    app.setFont(QFont(".AppleSystemUIFont", 13))

    # Restore persisted theme before constructing widgets
    from ui import config
    from ui.theme import set_dark_mode
    set_dark_mode(config.get_dark_mode())

    window = MainWindow()
    window.show()
    log.info("main window shown, entering event loop")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
