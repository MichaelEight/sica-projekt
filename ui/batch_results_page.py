"""Batch autoscan results list — sorted worst-first, click to open."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QScrollArea, QFrame)

import ui.theme as T


class BatchResultsPage(QWidget):
    """Lists files analyzed by batch processing, sorted by severity."""

    file_selected = Signal(str)   # emits base path
    go_back = Signal()            # back to upload

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[dict] = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar
        self._topbar = QWidget()
        self._topbar.setFixedHeight(52)
        self._topbar.setStyleSheet(f"background: {T.TOPBAR};")
        tb = QHBoxLayout(self._topbar)
        tb.setContentsMargins(16, 0, 16, 0)
        tb.setSpacing(12)

        self._back_btn = QPushButton("\u2190 Wczytaj inny plik")
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.setStyleSheet(self._back_btn_style())
        self._back_btn.clicked.connect(self.go_back.emit)
        tb.addWidget(self._back_btn)

        self._title = QLabel("Wyniki analizy wielu plików")
        self._title.setStyleSheet(
            "color: white; font-size: 14px; font-weight: 600;"
        )
        tb.addWidget(self._title)
        tb.addStretch()

        self._count_badge = QLabel("")
        self._count_badge.setStyleSheet(self._badge_style())
        tb.addWidget(self._count_badge)
        outer.addWidget(self._topbar)

        # Scrollable content
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"border: none; background: {T.BG};")
        self._content = QWidget()
        self._content.setStyleSheet(f"background: {T.BG};")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(40, 24, 40, 24)
        self._content_layout.setSpacing(10)
        self._content_layout.setAlignment(Qt.AlignTop)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, stretch=1)

    def _back_btn_style(self) -> str:
        return (
            "QPushButton { background: transparent; color: #cbd5e1; "
            "font-size: 12px; padding: 6px 12px; border-radius: 6px; border: none; }"
            "QPushButton:hover { background: rgba(255,255,255,0.1); color: white; }"
        )

    def _badge_style(self) -> str:
        return (
            "color: #cbd5e1; font-size: 11px; font-family: Menlo;"
            "background: rgba(255,255,255,0.08); border-radius: 4px;"
            "padding: 4px 10px;"
        )

    def _card_style(self) -> str:
        return (
            f"QFrame#batchCard {{ background: {T.WHITE}; border: 1px solid {T.BORDER};"
            f" border-radius: 10px; }}"
            f"QFrame#batchCard:hover {{ border: 1px solid {T.ACCENT}; }}"
        )

    def set_results(self, results: list[dict], cancelled: bool = False):
        self._results = list(results or [])
        self._results.sort(
            key=lambda r: (-int(r.get("red_count", 0)),
                           -int(r.get("yellow_count", 0)),
                           r.get("name", "")),
        )
        self._rebuild()
        n = len(self._results)
        suffix = " (anulowano)" if cancelled else ""
        self._count_badge.setText(f"{n} nagrań{suffix}")

    def _rebuild(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not self._results:
            lbl = QLabel("Brak wyników do wyświetlenia.")
            lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 13px;")
            lbl.setAlignment(Qt.AlignCenter)
            self._content_layout.addWidget(lbl)
            return

        for r in self._results:
            self._content_layout.addWidget(self._make_card(r))

    def _make_card(self, r: dict) -> QWidget:
        card = QFrame()
        card.setObjectName("batchCard")
        card.setCursor(Qt.PointingHandCursor)
        card.setStyleSheet(self._card_style())

        lay = QHBoxLayout(card)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(16)

        # Filename + duration + patient
        info = QVBoxLayout()
        info.setSpacing(2)
        name = QLabel(r.get("name", ""))
        name.setFont(QFont("Menlo", 13, QFont.DemiBold))
        name.setStyleSheet(f"color: {T.TEXT}; background: transparent; border: none;")
        info.addWidget(name)

        duration = r.get("duration", 0.0)
        patient = r.get("patient") or {}
        meta_bits = [f"{duration:.1f} s"]
        if patient.get("id"):
            meta_bits.append(f"ID {patient['id']}")
        if patient.get("age"):
            meta_bits.append(f"{patient['age']} lat")
        if patient.get("sex"):
            meta_bits.append(patient["sex"])
        meta = QLabel(" • ".join(meta_bits))
        meta.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px; background: transparent; border: none;")
        info.addWidget(meta)
        lay.addLayout(info, stretch=1)

        # Severity badges
        red = int(r.get("red_count", 0))
        yellow = int(r.get("yellow_count", 0))
        lay.addWidget(self._sev_badge(red, "red"))
        lay.addWidget(self._sev_badge(yellow, "yellow"))

        # Open button
        open_btn = QPushButton("Otwórz \u2192")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                border: none; border-radius: 6px;
                padding: 7px 14px; font-size: 12px; font-weight: 600;
            }}
            """
        )
        base_path = r.get("base_path", "")
        open_btn.clicked.connect(lambda _=False, p=base_path: self.file_selected.emit(p))
        lay.addWidget(open_btn)

        card.mousePressEvent = lambda e, p=base_path: self.file_selected.emit(p)
        return card

    def _sev_badge(self, count: int, kind: str) -> QLabel:
        if kind == "red":
            bg, fg = "#fee2e2", "#991b1b"
            label = f"\u26a0 {count} wys."
        else:
            bg, fg = "#fef3c7", "#92400e"
            label = f"\u26a1 {count} śr."
        if count == 0:
            bg = T.BORDER_LIGHT
            fg = T.TEXT_DIM
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"background: {bg}; color: {fg}; padding: 4px 10px;"
            f"border-radius: 4px; font-size: 11px; font-weight: 600;"
            f"font-family: Menlo; border: none;"
        )
        return lbl

    def apply_theme(self):
        self._topbar.setStyleSheet(f"background: {T.TOPBAR};")
        self._back_btn.setStyleSheet(self._back_btn_style())
        self._count_badge.setStyleSheet(self._badge_style())
        self._scroll.setStyleSheet(f"border: none; background: {T.BG};")
        self._content.setStyleSheet(f"background: {T.BG};")
        self._rebuild()
