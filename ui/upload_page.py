"""Upload / Welcome screen matching v2 01-upload design."""
import os
import json
from datetime import datetime
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFileDialog, QFrame)

import ui.theme as T

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

    file_selected = Signal(str)  # emits the base path (without extension)
    batch_selected = Signal(list)  # emits list of base paths for batch analysis

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {T.BG};")
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top bar ──
        topbar = QWidget()
        topbar.setFixedHeight(48)
        topbar.setStyleSheet(f"background: {T.TOPBAR};")
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(20, 0, 20, 0)
        logo = QLabel()
        logo.setText('<span style="color:#4a9eff; font-weight:600;">EKG</span>'
                     ' <span style="color:white; font-weight:600;">Assistant</span>')
        logo.setFont(QFont(".AppleSystemUIFont", 15))
        logo.setTextFormat(Qt.RichText)
        tb_layout.addWidget(logo)
        tb_layout.addStretch()
        outer.addWidget(topbar)

        # ── Main content ──
        center = QWidget()
        center.setStyleSheet(f"background: {T.BG};")
        main = QVBoxLayout(center)
        main.setAlignment(Qt.AlignCenter)
        main.setSpacing(24)

        # Title
        title = QLabel("Wczytaj sygna\u0142 EKG")
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

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.setAlignment(Qt.AlignCenter)

        dark = T.is_dark_mode()
        accent_hover = "#047857" if dark else "#3a8eef"
        accent_pressed = "#065f46" if dark else "#2563eb"
        secondary_hover_bg = T.BORDER_LIGHT
        secondary_pressed_bg = T.BORDER

        browse_btn = QPushButton("Wybierz plik")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                font-size: 14px; font-weight: 600;
                padding: 10px 32px; border-radius: 8px; border: none;
            }}
            QPushButton:hover {{
                background: {accent_hover};
            }}
            QPushButton:pressed {{
                background: {accent_pressed};
            }}
        """)
        browse_btn.clicked.connect(self._browse)
        btn_row.addWidget(browse_btn)

        batch_btn = QPushButton("Wybierz wiele plików")
        batch_btn.setCursor(Qt.PointingHandCursor)
        batch_btn.setStyleSheet(f"""
            QPushButton {{
                background: {T.WHITE}; color: {T.TEXT};
                font-size: 14px; font-weight: 600;
                padding: 10px 28px; border-radius: 8px;
                border: 1px solid {T.BORDER};
            }}
            QPushButton:hover {{
                background: {secondary_hover_bg};
                border: 1px solid {T.ACCENT}; color: {T.ACCENT};
            }}
            QPushButton:pressed {{
                background: {secondary_pressed_bg};
                border: 1px solid {T.ACCENT}; color: {T.ACCENT};
            }}
        """)
        batch_btn.clicked.connect(self._browse_batch)
        btn_row.addWidget(batch_btn)

        btn_wrap = QWidget()
        btn_wrap.setStyleSheet("background: transparent; border: none;")
        btn_wrap.setLayout(btn_row)
        dz_layout.addWidget(btn_wrap, alignment=Qt.AlignCenter)

        hint = QLabel("Oba pliki (.dat i .hea) muszą mieć tę samą nazwę i znajdować się w tym samym folderze")
        hint.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px; border: none;")
        hint.setAlignment(Qt.AlignCenter)
        dz_layout.addWidget(hint)

        main.addWidget(self.dropzone, alignment=Qt.AlignCenter)

        # ── Recent files ──
        recent_container = QWidget()
        recent_container.setFixedWidth(520)
        recent_container.setStyleSheet("background: transparent;")
        rc_layout = QVBoxLayout(recent_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(8)

        rc_header_row = QHBoxLayout()
        rc_header = QLabel("OSTATNIE PLIKI")
        rc_header.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {T.TEXT_DIM}; letter-spacing: 0.5px;")
        rc_header_row.addWidget(rc_header)
        rc_header_row.addStretch()
        self._clear_recent_btn = QPushButton("Wyczyść")
        self._clear_recent_btn.setCursor(Qt.PointingHandCursor)
        self._clear_recent_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 11px; color: {T.TEXT_DIM}; background: transparent;
                border: none; padding: 2px 6px;
            }}
            QPushButton:hover {{ color: {T.RED}; }}
        """)
        self._clear_recent_btn.clicked.connect(self._clear_recent)
        rc_header_row.addWidget(self._clear_recent_btn)
        rc_layout.addLayout(rc_header_row)

        self.recent_list = QVBoxLayout()
        self.recent_list.setSpacing(0)
        self.recent_frame = QFrame()
        self.recent_frame.setStyleSheet(f"background: {T.WHITE}; border: 1px solid {T.BORDER}; border-radius: 8px;")
        self.recent_frame.setLayout(self.recent_list)
        rc_layout.addWidget(self.recent_frame)
        main.addWidget(recent_container, alignment=Qt.AlignCenter)

        outer.addWidget(center, stretch=1)

        # ── Status bar ──
        statusbar = QWidget()
        statusbar.setFixedHeight(32)
        statusbar.setStyleSheet(f"background: {T.WHITE}; border-top: 1px solid {T.BORDER};")
        sb_layout = QHBoxLayout(statusbar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_label = QLabel("Cyfrowy asystent wspomagaj\u0105cy prac\u0119 specjalist\u00f3w z zakresu elektrofizjologii")
        sb_label.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 12px;")
        sb_label.setAlignment(Qt.AlignCenter)
        sb_layout.addWidget(sb_label)
        outer.addWidget(statusbar)

        self._refresh_recent()

    def _clear_recent(self):
        save_recent([])
        self._refresh_recent()

    def _refresh_recent(self):
        # Clear existing
        while self.recent_list.count():
            item = self.recent_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        entries = load_recent()
        if not entries:
            lbl = QLabel("  Brak ostatnich plik\u00f3w")
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

        # Icon
        icon = QLabel("E")
        icon.setFixedSize(32, 32)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"""
            background: {T.ICON_BG}; border-radius: 6px;
            color: {T.ACCENT}; font-size: 14px; font-weight: 700; border: none;
        """)
        layout.addWidget(icon)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)
        name = QLabel(entry.get("name", ""))
        name.setStyleSheet("font-size: 13px; font-weight: 600; font-family: Menlo; border: none;")
        info.addWidget(name)
        # Clean old info format — keep only duration (last "X.X s" part)
        raw_info = entry.get("info", "")
        import re
        duration_match = re.search(r'(\d+\.?\d*\s*s)\s*$', raw_info)
        clean_info = duration_match.group(1) if duration_match else raw_info
        meta = QLabel(clean_info)
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

    def _browse_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Wybierz nagrania WFDB", "",
            "Nagłówki WFDB (*.hea)"
        )
        if not paths:
            return
        seen = set()
        base_paths: list[str] = []
        skipped = 0
        for p in paths:
            base, ext = os.path.splitext(p)
            if ext.lower() != ".hea":
                skipped += 1
                continue
            if base in seen:
                continue
            seen.add(base)
            if not os.path.isfile(base + ".dat"):
                skipped += 1
                continue
            base_paths.append(base)

        if not base_paths:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Brak plików do analizy",
                "Nie wybrano żadnych poprawnych par .hea + .dat."
            )
            return

        self.batch_selected.emit(base_paths)

    def refresh(self):
        self._refresh_recent()
