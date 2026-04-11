"""Modal progress overlay for batch processing."""
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar

import ui.theme as T


class BatchOverlay(QDialog):
    """Modal overlay shown during batch autoscan processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedSize(420, 260)

        self._cancel_callback = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        self._spinner = QLabel("\u23f3")
        self._spinner.setAlignment(Qt.AlignCenter)
        self._spinner.setStyleSheet("font-size: 40px; background: transparent;")
        layout.addWidget(self._spinner)

        self._title = QLabel("Analiza wielu plików")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #1e293b; background: transparent;"
        )
        layout.addWidget(self._title)

        self._file_label = QLabel("Plik 0 / 0")
        self._file_label.setAlignment(Qt.AlignCenter)
        self._file_label.setStyleSheet(
            "font-size: 12px; color: #475569; background: transparent;"
        )
        layout.addWidget(self._file_label)

        self._name_label = QLabel("")
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setStyleSheet(
            "font-size: 12px; font-family: Menlo; color: #334155; background: transparent;"
        )
        layout.addWidget(self._name_label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(
            """
            QProgressBar { background: #e2e8f0; border: none; border-radius: 4px; }
            QProgressBar::chunk { background: #3a8eef; border-radius: 4px; }
            """
        )
        layout.addWidget(self._bar)

        self._window_label = QLabel("")
        self._window_label.setAlignment(Qt.AlignCenter)
        self._window_label.setStyleSheet(
            "font-size: 11px; color: #64748b; background: transparent;"
        )
        layout.addWidget(self._window_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QPushButton("Anuluj")
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.setStyleSheet(
            """
            QPushButton {
                background: #e2e8f0; color: #1e293b;
                border: none; border-radius: 6px;
                padding: 8px 22px; font-size: 12px; font-weight: 600;
            }
            QPushButton:hover { background: #cbd5e1; }
            """
        )
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(500)
        self._pulse_frames = ["\u23f3", "\u231b"]
        self._pulse_idx = 0
        self._pulse_timer.timeout.connect(self._pulse_tick)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(QColor(255, 255, 255, 244))
        p.setPen(QPen(QColor("#e2e8f0"), 1))
        p.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 16, 16)
        p.end()

    def set_cancel_callback(self, fn):
        self._cancel_callback = fn

    def _on_cancel(self):
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Anulowanie...")
        if self._cancel_callback:
            self._cancel_callback()

    def show_loading(self):
        self._pulse_timer.start()
        self._center_on_parent()
        self.show()

    def update(self, file_idx: int, file_total: int,
               window_idx: int, window_total: int, name: str):
        self._file_label.setText(f"Plik {file_idx} / {file_total}")
        self._name_label.setText(name or "")
        if window_total > 0:
            pct = int(window_idx / window_total * 100)
            self._bar.setValue(pct)
            self._window_label.setText(f"Okno {window_idx} / {window_total}")
        else:
            self._bar.setValue(0)
            self._window_label.setText("")

    def show_done(self, cancelled: bool = False):
        self._pulse_timer.stop()
        self._spinner.setText("\u2713" if not cancelled else "\u26d4")
        self._title.setText("Anulowano" if cancelled else "Zakończono")
        self._cancel_btn.hide()
        QTimer.singleShot(800, self.accept)

    def _pulse_tick(self):
        self._pulse_idx = (self._pulse_idx + 1) % len(self._pulse_frames)
        self._spinner.setText(self._pulse_frames[self._pulse_idx])

    def _center_on_parent(self):
        parent = self.parent()
        if parent:
            pr = parent.rect()
            x = pr.center().x() - self.width() // 2
            y = pr.center().y() - self.height() // 2
            self.move(parent.mapToGlobal(QPoint(x, y)))
