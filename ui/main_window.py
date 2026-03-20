"""Main application window orchestrating all pages."""
import os
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget, QMessageBox

from ui.theme import STYLESHEET, STANDARD_LEADS, BG
from ui.upload_page import UploadPage, add_recent
from ui.viewer_page import ViewerPage
from ui.report_page import ReportPage
from ui.ekg_canvas import generate_demo_signal

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False


# Map common WFDB lead name variants to our standard names
_LEAD_ALIASES = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3",
    "v4": "V4", "v5": "V5", "v6": "V6",
}


def _normalize_lead_names(names: list[str]) -> list[str]:
    """Normalize WFDB lead names to standard form (I, II, aVR, V1, etc.)."""
    return [_LEAD_ALIASES.get(n.lower(), n) for n in names]


class MainWindow(QMainWindow):
    """Main application window with Upload → Viewer → Report flow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EKG Assistant")
        self.setMinimumSize(1200, 800)
        self.resize(1440, 900)

        # Apply theme
        self.setStyleSheet(STYLESHEET)

        # Central stacked widget
        self.stack = QStackedWidget()
        self.stack.setObjectName("centralWidget")
        self.setCentralWidget(self.stack)

        # Pages
        self.upload_page = UploadPage()
        self.viewer_page = ViewerPage()
        self.report_page = ReportPage()

        self.stack.addWidget(self.upload_page)   # 0
        self.stack.addWidget(self.viewer_page)   # 1
        self.stack.addWidget(self.report_page)   # 2

        # Connect signals
        self.upload_page.file_selected.connect(self._load_file)
        self.viewer_page.open_file.connect(self._go_upload)
        self.viewer_page.show_report.connect(self._go_report)
        self.report_page.go_back.connect(self._go_viewer)

        # Show upload page
        self.stack.setCurrentIndex(0)

        # Current data
        self._signal = None
        self._leads = STANDARD_LEADS
        self._fs = 500
        self._filename = ""

    def _load_file(self, base_path: str):
        """Load a WFDB record or generate demo data."""
        dat_path = base_path + ".dat"
        hea_path = base_path + ".hea"

        if os.path.exists(dat_path) and os.path.exists(hea_path) and HAS_WFDB:
            try:
                record = wfdb.rdrecord(base_path)
                self._signal = record.p_signal.astype(np.float32)
                self._leads = _normalize_lead_names(record.sig_name)
                self._fs = record.fs
                self._filename = os.path.basename(base_path) + ".dat"

                info = f"{self._fs} Hz \u00b7 {len(self._leads)} odprowadze\u0144 \u00b7 {self._signal.shape[0] / self._fs:.1f} s"
                add_recent(base_path, info)
            except Exception as e:
                QMessageBox.warning(self, "B\u0142\u0105d", f"Nie uda\u0142o si\u0119 wczyta\u0107 pliku:\n{e}")
                return
        else:
            # Generate demo signal
            self._signal = generate_demo_signal(STANDARD_LEADS, fs=500, duration=10.0)
            self._leads = STANDARD_LEADS
            self._fs = 500
            self._filename = os.path.basename(base_path) + ".dat" if base_path else "demo.dat"
            add_recent(base_path or "demo", f"{self._fs} Hz \u00b7 12 odprowadze\u0144 \u00b7 10.0 s")

        # Update pages
        self.viewer_page.set_signal(self._signal, self._leads, self._fs, self._filename)
        self.report_page.set_signal(self._signal, self._leads, self._fs, self._filename)

        # Navigate to viewer
        self.stack.setCurrentIndex(1)

    def _go_upload(self):
        self.upload_page.refresh()
        self.stack.setCurrentIndex(0)

    def _go_viewer(self):
        self.stack.setCurrentIndex(1)

    def _go_report(self):
        self.stack.setCurrentIndex(2)

    def keyPressEvent(self, event):
        """Global keyboard shortcuts matching the v2 status bar hints."""
        key = event.key()
        mods = event.modifiers()

        if self.stack.currentIndex() == 1:  # Viewer page
            # View mode shortcuts
            if key == Qt.Key_1:
                self.viewer_page.view_seg.set_active(0)
            elif key == Qt.Key_2:
                self.viewer_page.view_seg.set_active(1)
            elif key == Qt.Key_3:
                self.viewer_page.view_seg.set_active(2)
            # Tool shortcuts
            elif key == Qt.Key_V:
                self.viewer_page._on_tool_mode(0)
            elif key == Qt.Key_C:
                self.viewer_page._on_tool_mode(1)
            elif key == Qt.Key_A:
                self.viewer_page._on_tool_mode(2)
            # Navigation
            elif key == Qt.Key_Left:
                self.viewer_page._nav_step(-0.2)
            elif key == Qt.Key_Right:
                self.viewer_page._nav_step(0.2)
            elif key == Qt.Key_Home:
                self.viewer_page._nav_start()
            elif key == Qt.Key_End:
                self.viewer_page._nav_end()
            # Monitor pause
            elif key == Qt.Key_Space and self.viewer_page._view_mode == 2:
                self.viewer_page._monitor_playing = not self.viewer_page._monitor_playing
                if self.viewer_page._monitor_playing:
                    self.viewer_page._monitor_timer.start()
                else:
                    self.viewer_page._monitor_timer.stop()
            # Report shortcut
            elif key == Qt.Key_E and mods & Qt.ControlModifier:
                self._go_report()
            # Analyze shortcut
            elif key == Qt.Key_Return and mods & Qt.ControlModifier:
                self.viewer_page._on_analyze()
            # Escape
            elif key == Qt.Key_Escape:
                if self.viewer_page._view_mode == 2:
                    self.viewer_page.view_seg.set_active(0)
                elif self.viewer_page._tool_mode != 0:
                    self.viewer_page._on_tool_mode(0)

        elif self.stack.currentIndex() == 2:  # Report page
            if key == Qt.Key_Escape:
                self._go_viewer()

        super().keyPressEvent(event)
