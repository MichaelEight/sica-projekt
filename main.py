"""EKG Assistant — Desktop Application.

Usage:
    python main.py
"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from ui.app_logger import get_logger
from ui.main_window import MainWindow


def main():
    log = get_logger("main")
    log.info("starting EKG Assistant")

    app = QApplication(sys.argv)
    app.setApplicationName("EKG Assistant")
    app.setOrganizationName("EKG Assistant")
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
