"""Upload / Welcome screen matching v2 01-upload design."""
import os
import json
from datetime import datetime
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFileDialog, QFrame)

import ui.theme as T
from ui.widgets import make_logo

RECENT_FILE = os.path.expanduser("~/.ekg_assistant_recent.json")


def load_recent() -> list[dict]:
    try:
        with open(RECENT_FILE) as f:
            return json.load(f)[:10]
    except Exception:
        return []


def save_recent(entries: list[dict]):
    try:
        with open(RECENT_FILE, "w") as f:
            json.dump(entries[:10], f)
    except Exception:
        pass


def add_recent(filepath: str, info: str = ""):
    entries = load_recent()
    entry = {"path": filepath, "name": os.path.basename(filepath),
             "info": info, "date": datetime.now().strftime("%d.%m.%Y")}
    entries = [e for e in entries if e["path"] != filepath]
    entries.insert(0, entry)
    save_recent(entries[:10])


class UploadPage(QWidget):
    """Welcome screen with file picker and recent files."""

    file_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {T.BG};")
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(48)
        topbar.setStyleSheet(f"background: {T.TOPBAR};")
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(20, 0, 20, 0)
        logo = make_logo(15)
        tb_layout.addWidget(logo)
        tb_layout.addStretch()
        outer.addWidget(topbar)

        # Main content
        center = QWidget()
        center.setStyleSheet(f"background: {T.BG};")
        main = QVBoxLayout(center)
        main.setAlignment(Qt.AlignCenter)
        main.setSpacing(24)

        title = QLabel("Wczytaj sygnał EKG")
        title.setFont(QFont(".AppleSystemUIFont", 22, QFont.DemiBold))
        title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        subtitle = QLabel("Aplikacja analizuje sygnały EKG zapisane w formacie WFDB")
        subtitle.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignCenter)
        main.addWidget(subtitle)

        # File picker zone
        self.dropzone = QFrame()
        self.dropzone.setFixedSize(520, 240)
        self.dropzone.setStyleSheet(f"""
            QFrame {{
                background: {T.WHITE};
                border: 1px solid {T.BORDER};
                border-radius: 12px;
            }}
        """)
        dz_layout = QVBoxLayout(self.dropzone)
        dz_layout.setAlignment(Qt.AlignCenter)
        dz_layout.setSpacing(12)

        step_label = QLabel("Wskaż plik .dat lub .hea z nagrania EKG")
        step_label.setStyleSheet("font-size: 14px; font-weight: 500; border: none;")
        step_label.setAlignment(Qt.AlignCenter)
        dz_layout.addWidget(step_label)

        browse_btn = QPushButton("Wybierz plik")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {T.ACCENT}; color: white;
                font-size: 14px; font-weight: 600;
                padding: 10px 32px; border-radius: 8px; border: none;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """)
        browse_btn.clicked.connect(self._browse)
        dz_layout.addWidget(browse_btn, alignment=Qt.AlignCenter)

        hint = QLabel("Oba pliki (.dat i .hea) muszą mieć tę samą nazwę i znajdować się w tym samym folderze")
        hint.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px; border: none;")
        hint.setAlignment(Qt.AlignCenter)
        dz_layout.addWidget(hint)

        main.addWidget(self.dropzone, alignment=Qt.AlignCenter)

        # Recent files
        recent_container = QWidget()
        recent_container.setFixedWidth(520)
        recent_container.setStyleSheet("background: transparent;")
        rc_layout = QVBoxLayout(recent_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(8)

        rc_header = QLabel("OSTATNIE PLIKI")
        rc_header.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {T.TEXT_DIM}; letter-spacing: 0.5px;")
        rc_layout.addWidget(rc_header)

        self.recent_list = QVBoxLayout()
        self.recent_list.setSpacing(0)
        self.recent_frame = QFrame()
        self.recent_frame.setStyleSheet(f"background: {T.WHITE}; border: 1px solid {T.BORDER}; border-radius: 8px;")
        self.recent_frame.setLayout(self.recent_list)
        rc_layout.addWidget(self.recent_frame)
        main.addWidget(recent_container, alignment=Qt.AlignCenter)

        outer.addWidget(center, stretch=1)

        # Status bar
        statusbar = QWidget()
        statusbar.setFixedHeight(32)
        statusbar.setStyleSheet(f"background: {T.WHITE}; border-top: 1px solid {T.BORDER};")
        sb_layout = QHBoxLayout(statusbar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_label = QLabel("Cyfrowy asystent wspomagający pracę specjalistów z zakresu elektrofizjologii")
        sb_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px;")
        sb_label.setAlignment(Qt.AlignCenter)
        sb_layout.addWidget(sb_label)
        outer.addWidget(statusbar)

        self._refresh_recent()

    def _refresh_recent(self):
        while self.recent_list.count():
            item = self.recent_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        entries = load_recent()
        if not entries:
            lbl = QLabel("  Brak ostatnich plików")
            lbl.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 13px; padding: 12px; border: none;")
            self.recent_list.addWidget(lbl)
            return

        for i, entry in enumerate(entries[:5]):
            row = self._make_recent_row(entry, is_last=(i == min(len(entries), 5) - 1))
            self.recent_list.addWidget(row)

    def _make_recent_row(self, entry: dict, is_last: bool = False) -> QWidget:
        row = QWidget()
        row.setCursor(Qt.PointingHandCursor)
        border_style = "" if is_last else f"border-bottom: 1px solid {T.BORDER_LIGHT};"
        row.setStyleSheet(f"padding: 8px 12px; {border_style}")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        icon = QLabel("E")
        icon.setFixedSize(32, 32)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"""
            background: {T.ICON_BG}; border-radius: 6px;
            color: {T.ACCENT}; font-size: 14px; font-weight: 700; border: none;
        """)
        layout.addWidget(icon)

        info = QVBoxLayout()
        info.setSpacing(2)
        name = QLabel(entry.get("name", ""))
        name.setStyleSheet("font-size: 13px; font-weight: 600; font-family: Menlo; border: none;")
        info.addWidget(name)
        meta = QLabel(entry.get("info", ""))
        meta.setStyleSheet(f"font-size: 11px; color: {T.TEXT_DIM}; border: none;")
        info.addWidget(meta)
        layout.addLayout(info, stretch=1)

        date = QLabel(entry.get("date", ""))
        date.setStyleSheet(f"font-size: 12px; color: {T.TEXT_DIM}; border: none;")
        layout.addWidget(date)

        path = entry.get("path", "")
        row.mousePressEvent = lambda e, p=path: self.file_selected.emit(p)
        return row

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik EKG", "",
            "Pliki WFDB (*.dat *.hea);;Wszystkie pliki (*)"
        )
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext not in ('.dat', '.hea'):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "Nieobsługiwany format",
                    f"Plik \"{os.path.basename(path)}\" nie jest w formacie WFDB.\n"
                    "Wybierz plik z rozszerzeniem .dat lub .hea.")
                return
            base, _ = os.path.splitext(path)
            dat_path = base + ".dat"
            hea_path = base + ".hea"
            missing = []
            if not os.path.isfile(dat_path):
                missing.append(".dat")
            if not os.path.isfile(hea_path):
                missing.append(".hea")
            if missing:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self, "Brak pliku",
                    f"Nie znaleziono pliku {', '.join(missing)} w tym samym folderze.\n\n"
                    "Format WFDB wymaga obu plików (.dat i .hea)\n"
                    "o tej samej nazwie, w tym samym folderze.")
                return
            self.file_selected.emit(base)

    def refresh(self):
        self._refresh_recent()
