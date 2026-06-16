"""Shared UI helpers used across pages."""
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFrame

import ui.theme as T


class RangeSlider(QWidget):
    """Horizontal dual-handle slider for the two AI confidence thresholds.

    A single track with two handles: the YELLOW handle = 'do sprawdzenia'
    (borderline) threshold, the RED handle = 'choroba' (illness) threshold.
    The track is tinted in tier colors — grey below the yellow handle, yellow
    between handles, red above the red handle — so the meaning is obvious
    without reading anything. The yellow handle can never pass the red one
    (low <= high is enforced). Values are integer percents (0..100).
    """

    valueChanged = Signal(int, int)  # low_pct, high_pct (live, during drag)

    def __init__(self, low: int = 40, high: int = 70, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(34)
        self.setMinimumWidth(180)
        self._low = int(low)
        self._high = int(high)
        self._drag: str | None = None  # 'low' | 'high'
        self._r = 8  # handle radius
        self.setCursor(Qt.PointingHandCursor)

    def values(self) -> tuple[int, int]:
        return self._low, self._high

    def set_values(self, low: int, high: int) -> None:
        low = max(0, min(100, int(low)))
        high = max(0, min(100, int(high)))
        if low > high:
            low, high = high, low
        if (low, high) != (self._low, self._high):
            self._low, self._high = low, high
            self.update()

    # ── geometry ──
    def _track(self):
        m = self._r + 2
        return m, self.width() - m, self.height() / 2  # x0, x1, y_center

    def _pct_to_x(self, pct: int) -> float:
        x0, x1, _ = self._track()
        if x1 <= x0:
            return x0
        return x0 + (x1 - x0) * pct / 100.0

    def _x_to_pct(self, x: float) -> int:
        x0, x1, _ = self._track()
        if x1 <= x0:
            return 0
        return int(round(max(0.0, min(1.0, (x - x0) / (x1 - x0))) * 100))

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        x0, x1, y = self._track()
        if x1 <= x0:
            return  # too narrow to draw meaningfully
        xl = self._pct_to_x(self._low)
        xh = self._pct_to_x(self._high)
        th = 5.0  # track thickness
        p.setPen(Qt.NoPen)
        # grey: below borderline
        p.setBrush(QColor(T.BORDER))
        p.drawRoundedRect(QRectF(x0, y - th / 2, x1 - x0, th), 2.5, 2.5)
        # yellow: borderline .. illness
        if xh > xl:
            p.setBrush(QColor(T.TIER_YELLOW))
            p.drawRect(QRectF(xl, y - th / 2, xh - xl, th))
        # red: illness .. top
        if x1 > xh:
            p.setBrush(QColor(T.RED))
            p.drawRect(QRectF(xh, y - th / 2, x1 - xh, th))
        # "Niepewne" zone: 0 up to the yellow threshold, capped at 40% (above
        # which the model is confident enough to name an illness). Diagonal
        # hatch marks it as low-reliability ("not healthy, no known illness").
        mark = min(self._low, 40)
        if mark > 0:
            xm = self._pct_to_x(mark)
            zone = QRectF(x0, y - 8, xm - x0, 16)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(T.TEXT_DIM), Qt.FDiagPattern))
            p.drawRect(zone)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(T.TEXT_DIM), 1))
            p.drawRect(zone)
            p.setPen(Qt.NoPen)
        # handles (draw red first so an overlap shows yellow on top)
        for x, col in ((xh, T.RED), (xl, T.TIER_YELLOW)):
            p.setBrush(QColor(T.WHITE))
            p.setPen(QPen(QColor(col), 2.5))
            p.drawEllipse(QPointF(x, y), self._r, self._r)

    def _nearest(self, x: float) -> str:
        xl, xh = self._pct_to_x(self._low), self._pct_to_x(self._high)
        dl, dh = abs(x - xl), abs(x - xh)
        if dl == dh:
            # Handles coincide (or equidistant): pick by drag side so a
            # collapsed pair can always be separated again.
            return "high" if x >= xl else "low"
        return "low" if dl < dh else "high"

    def mousePressEvent(self, e):
        self._drag = self._nearest(e.position().x())
        self._apply(e.position().x())

    def mouseMoveEvent(self, e):
        if self._drag:
            self._apply(e.position().x())

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def _apply(self, x: float):
        pct = self._x_to_pct(x)
        if self._drag == "low":
            self._low = min(pct, self._high)
        elif self._drag == "high":
            self._high = max(pct, self._low)
        else:
            return
        self.update()
        self.valueChanged.emit(self._low, self._high)


class LeadImportanceBar(QWidget):
    """Bar chart of lead importances (XAI). Shows top 3 leads that influenced model.

    Per professor 2026-04-14 + 11mat11: explain which leads drove the AI's decision.
    """

    closed = Signal()  # X button clicked: caller deselects the active illness

    _CLOSE_SZ = 20  # hit-box for the X (drawn directly; native button stays
                    # invisible under the custom paintEvent)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self._importances: dict[str, float] = {}
        self._title = ""
        self._close_rect = QRectF()
        self._close_hover = False

    def _close_box(self) -> QRectF:
        s = self._CLOSE_SZ
        return QRectF(self.width() - s - 6, 4, s, s)

    def mouseMoveEvent(self, event):
        hov = self._close_box().contains(event.position())
        if hov != self._close_hover:
            self._close_hover = hov
            self.setCursor(Qt.PointingHandCursor if hov else Qt.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self._close_box().contains(event.position()):
            self.closed.emit()
            return
        super().mousePressEvent(event)

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

        # Close (X) — drawn so it composites correctly over the painted bg
        box = self._close_box()
        p.setPen(QColor(T.RED if self._close_hover else T.TEXT_SECONDARY))
        p.setFont(QFont(".AppleSystemUIFont", 14, QFont.Bold))
        p.drawText(box, Qt.AlignCenter, "✕")

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
        # list of (t_start, t_end, code) — color codes match TwelveLeadGrid
        # autoscan: 1 = yellow (borderline), 2 = red (illness).
        self._autoscan_regions: list = []

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

    def set_autoscan_regions(self, regions: list):
        """Accept (t_start, t_end, code, *_) tuples; only first three used."""
        self._autoscan_regions = [
            (float(r[0]), float(r[1]), int(r[2])) for r in (regions or [])
        ]
        self.update()

    def clear_autoscan_regions(self):
        self._autoscan_regions = []
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

        # Autoscan colored bands — drawn beneath envelope so signal stays
        # readable. Yellow = borderline, red = illness, healthy gets nothing.
        if self._autoscan_regions and self._duration > 0:
            colors = {
                1: QColor(250, 191, 0, 210),   # yellow — intense amber
                2: QColor(229, 28, 28, 225),    # red — intense
            }
            for t_s, t_e, code in self._autoscan_regions:
                if code <= 0:
                    continue
                x1 = int(t_s / self._duration * w)
                x2 = int(t_e / self._duration * w)
                if x2 <= x1:
                    continue
                p.setBrush(colors.get(min(code, 2), colors[2]))
                p.setPen(Qt.NoPen)
                p.drawRect(x1, 2, x2 - x1, h - 4)

        # Signal envelope
        if self._overview is not None and self._duration > 0:
            mn = float(self._overview[:, 0].min())
            mx = float(self._overview[:, 1].max())
            rng = mx - mn if mx > mn else 1.0
            bins = self._overview.shape[0]
            p.setPen(QPen(QColor(T.TEXT), 1))
            p.setBrush(QColor(T.TEXT))
            pad = 4
            usable = h - 2 * pad
            for i in range(bins):
                x = int(i / bins * w)
                y1 = pad + int((1 - (self._overview[i, 1] - mn) / rng) * usable)
                y2 = pad + int((1 - (self._overview[i, 0] - mn) / rng) * usable)
                if y1 >= y2:
                    y2 = y1 + 1
                p.drawLine(x, y1, x, y2)

        # Window rectangle — track _time_pos continuously so the box stays
        # in sync with the slider/scrubber. Click-seek still snaps via _x_to_time.
        if self._duration > 0 and self._window > 0:
            pos = max(0.0, min(self._time_pos, self._duration - self._window))
            x = int(pos / self._duration * w)
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
