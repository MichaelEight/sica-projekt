"""Shared UI helpers used across pages."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFrame

import ui.theme as T


class LeadImportanceBar(QWidget):
    """Bar chart of lead importances (XAI). Shows top 3 leads that influenced model.

    Per professor 2026-04-14 + 11mat11: explain which leads drove the AI's decision.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._importances: dict[str, float] = {}
        self._title = ""

    def set_data(self, lead_importance: dict | None, title: str = ""):
        self._importances = lead_importance or {}
        self._title = title
        self.update()
        self.setVisible(bool(self._importances))

    def paintEvent(self, _event):
        if not self._importances:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, QColor(T.BG_SECONDARY))

        # Title
        p.setPen(QColor(T.TEXT))
        p.setFont(QFont(".AppleSystemUIFont", 10, QFont.DemiBold))
        p.drawText(8, 14, "Wpływ odprowadzeń na decyzję")
        if self._title:
            p.setPen(QColor(T.TEXT_DIM))
            p.setFont(QFont(".AppleSystemUIFont", 9))
            p.drawText(8, 28, self._title)

        # Top 3 leads by importance
        sorted_leads = sorted(self._importances.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if not sorted_leads:
            return
        max_v = max(v for _, v in sorted_leads) or 1.0

        bar_h = 22
        gap = 6
        y0 = 38
        label_w = 36
        right_pad = 50
        bar_area_w = w - label_w - right_pad - 16

        for i, (lead, val) in enumerate(sorted_leads):
            y = y0 + i * (bar_h + gap)
            # Lead label
            p.setPen(QColor(T.TEXT))
            p.setFont(QFont("Menlo", 11, QFont.Bold))
            p.drawText(8, y + bar_h - 6, lead)
            # Bar
            bw = int(val / max_v * bar_area_w)
            color = QColor(T.ACCENT) if i == 0 else QColor("#94a3b8")
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(color))
            p.drawRoundedRect(label_w + 8, y, bw, bar_h, 4, 4)
            # Percentage
            p.setPen(QColor(T.TEXT))
            p.setFont(QFont("Menlo", 10))
            p.drawText(label_w + 8 + bw + 6, y + bar_h - 6, f"{val:.1f}%")

        p.end()


class TimelineOverview(QWidget):
    """Thin strip above the scrubber showing full signal with current window rect.

    Professor feedback 2026-04-14: "w pasku pokazać całą długość i na to nałożyć
    prostokąt-okienko, gdzie my jesteśmy".
    """

    seek_requested = Signal(float)  # time in seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setCursor(Qt.PointingHandCursor)
        self._overview: np.ndarray | None = None
        self._duration: float = 0.0
        self._time_pos: float = 0.0
        self._window: float = 0.0
        self._dragging = False

    def set_signal(self, signal: np.ndarray | None, duration: float):
        """Set the full signal (N,C) and duration; builds a lightweight envelope."""
        if signal is None or signal.size == 0:
            self._overview = None
            self._duration = 0.0
        else:
            # Average across leads for a single envelope line
            avg = signal.mean(axis=1) if signal.ndim == 2 else signal
            bins = 600
            if len(avg) > bins:
                step = len(avg) // bins
                trimmed = avg[: step * bins].reshape(bins, step)
                self._overview = np.stack([trimmed.min(axis=1), trimmed.max(axis=1)], axis=1)
            else:
                self._overview = np.stack([avg, avg], axis=1)
            self._duration = float(duration)
        self.update()

    def set_window(self, time_pos: float, window: float):
        self._time_pos = float(time_pos)
        self._window = float(window)
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        h = self.height()

        # Background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(T.BORDER_LIGHT))
        p.drawRoundedRect(0, 2, w, h - 4, 3, 3)

        # Signal envelope
        if self._overview is not None and self._duration > 0:
            mn = float(self._overview[:, 0].min())
            mx = float(self._overview[:, 1].max())
            rng = mx - mn if mx > mn else 1.0
            bins = self._overview.shape[0]
            p.setPen(QPen(QColor(T.TEXT_DIM), 1))
            p.setBrush(QColor(T.TEXT_DIM))
            pad = 4
            usable = h - 2 * pad
            for i in range(bins):
                x = int(i / bins * w)
                y1 = pad + int((1 - (self._overview[i, 1] - mn) / rng) * usable)
                y2 = pad + int((1 - (self._overview[i, 0] - mn) / rng) * usable)
                if y1 >= y2:
                    y2 = y1 + 1
                p.drawLine(x, y1, x, y2)

        # Window rectangle — snapped to discrete page boundaries so it jumps
        # in sync with the actual graph window instead of sliding continuously.
        if self._duration > 0 and self._window > 0:
            page_idx = int(self._time_pos // self._window)
            snapped = page_idx * self._window
            if snapped + self._window > self._duration:
                snapped = max(0.0, self._duration - self._window)
            x = int(snapped / self._duration * w)
            rect_w = max(4, int(self._window / self._duration * w))
            rect_w = min(rect_w, w - x)
            p.setPen(QPen(QColor(T.ACCENT), 2))
            p.setBrush(QBrush(QColor(T.ACCENT).lighter(180)))
            # semi-transparent fill
            fill = QColor(T.ACCENT)
            fill.setAlpha(70)
            p.setBrush(QBrush(fill))
            p.drawRoundedRect(x, 1, rect_w, h - 2, 3, 3)

        p.end()

    def _x_to_time(self, x: int) -> float:
        if self.width() <= 0 or self._duration <= 0:
            return 0.0
        t = x / self.width() * self._duration
        # Snap click target to the page boundary that contains x
        if self._window > 0:
            page_idx = int(t // self._window)
            snapped = page_idx * self._window
            return max(0.0, min(self._duration - self._window, snapped))
        return max(0.0, min(self._duration - self._window, t - self._window / 2))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._duration > 0:
            self._dragging = True
            self.seek_requested.emit(self._x_to_time(event.position().x()))

    def mouseMoveEvent(self, event):
        if self._dragging and self._duration > 0:
            self.seek_requested.emit(self._x_to_time(event.position().x()))

    def mouseReleaseEvent(self, _event):
        self._dragging = False


def make_logo(font_size=14):
    logo = QLabel()
    logo.setText('<span style="color:#4a9eff;font-weight:700;">Kardio</span>'
                 '<span style="color:white;font-weight:700;">skop</span>')
    logo.setFont(QFont(".AppleSystemUIFont", font_size))
    logo.setTextFormat(Qt.RichText)
    return logo


def make_separator(width=1, height=24):
    sep = QFrame()
    sep.setFixedSize(width, height)
    sep.setStyleSheet(f"background: {T.SEPARATOR};")
    return sep


def section_header(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        font-size: 11px; font-weight: 700; color: {T.TEXT_DIM};
        text-transform: uppercase; letter-spacing: 0.5px;
        padding-bottom: 4px; border-bottom: 1px solid {T.BORDER_LIGHT};
    """)
    return lbl


def _format_val(value, unit):
    return f'{value} <span style="font-size:10px;color:{T.TEXT_DIM};">{unit}</span>' if unit else value


def info_row(label, value, unit=""):
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px;")
    val = QLabel(_format_val(value, unit))
    val.setTextFormat(Qt.RichText)
    val.setStyleSheet("font-weight: 600; font-family: Menlo; font-size: 13px;")
    val.setAlignment(Qt.AlignRight)
    layout.addWidget(lbl)
    layout.addStretch()
    layout.addWidget(val)
    row.value_label = val
    row.unit = unit
    return row


def set_info_row(row, value):
    """Update an info_row in place without recreating widgets."""
    row.value_label.setText(_format_val(value, row.unit))


def make_action_btn(text, primary=False):
    btn = QPushButton(text)
    btn.setCursor(Qt.PointingHandCursor)
    if primary:
        btn.setObjectName("primary")
        btn.setStyleSheet(f"""
            QPushButton {{
                padding: 8px; border-radius: 6px; border: none;
                background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{
                background: {T.GREEN if T.is_dark_mode() else '#3a8eef'};
            }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{
                padding: 8px; border-radius: 6px; border: 1px solid {T.BORDER};
                background: {T.WHITE}; color: {T.TEXT}; font-size: 12px;
            }}
            QPushButton:hover {{
                background: {T.BG_SECONDARY};
            }}
        """)
    return btn
