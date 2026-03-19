"""Custom QWidget for rendering EKG signals with paper-grid background."""
import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath, QBrush
from PySide6.QtWidgets import QWidget

from ui.theme import (GRID_MINOR, GRID_MAJOR, SIGNAL_COLOR, ACCENT, WHITE,
                       LEAD_SEEDS, LEAD_AMPS, TEXT, TEXT_DIM, BORDER)


# ── Synthetic EKG generator ────────────────────
def synth_ekg(t: np.ndarray, seed: float = 0.0, amp: float = 1.0) -> np.ndarray:
    """Generate synthetic EKG signal matching the HTML design's waveform."""
    period = 0.833  # ~72 bpm
    phase = ((t % period) / period) * 2 * np.pi
    v = 0.12 * np.exp(-((phase - 0.9) ** 2) / 0.03)        # P wave
    v -= 0.08 * np.exp(-((phase - 1.55) ** 2) / 0.004)      # Q wave
    v += (0.9 + seed * 0.08) * np.exp(-((phase - 1.65) ** 2) / 0.005)  # R wave
    v -= 0.15 * np.exp(-((phase - 1.78) ** 2) / 0.004)      # S wave
    v += 0.2 * np.exp(-((phase - 2.8) ** 2) / 0.06)         # T wave
    return v * amp


def generate_demo_signal(leads: list[str], fs: int = 500, duration: float = 10.0) -> np.ndarray:
    """Generate demo 12-lead signal. Returns (n_samples, n_leads)."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    signal = np.zeros((n, len(leads)))
    for i, lead in enumerate(leads):
        seed = LEAD_SEEDS.get(lead, 0)
        amp = LEAD_AMPS.get(lead, 1.0)
        signal[:, i] = synth_ekg(t, seed, amp)
    return signal


# ── Single-cell EKG canvas ─────────────────────
class EkgCellCanvas(QWidget):
    """Draws a single EKG lead cell with paper grid, calibration pulse, and signal."""

    clicked = Signal(float, float)  # time_s, voltage_mV

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lead_name = ""
        self.signal = None       # 1-D numpy array (mV)
        self.fs = 500
        self.t_start = 0.0
        self.t_end = 2.5
        self.show_cal = True
        self.show_label = True
        self.calipers = []       # list of (t1, t2, color, label)
        self.annotations = []    # list of (t1, t2)
        self._sweep_pos = None   # fraction 0..1 for monitor mode
        self.setMinimumSize(80, 40)

    def set_data(self, lead_name: str, signal: np.ndarray, fs: int,
                 t_start: float = 0.0, t_end: float = 2.5):
        self.lead_name = lead_name
        self.signal = signal
        self.fs = fs
        self.t_start = t_start
        self.t_end = t_end
        self.update()

    def set_sweep(self, fraction: float):
        self._sweep_pos = fraction
        self.update()

    # Inset for the border — paint grid/signal inside this margin
    INSET = 0
    draw_border = False

    def paintEvent(self, event):
        if self.width() < 2 or self.height() < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        full_w, full_h = self.width(), self.height()
        ins = self.INSET

        # ── Background ──
        p.fillRect(0, 0, full_w, full_h, QColor(WHITE))

        # ── Border (drawn by this widget, not stylesheet) ──
        if self.draw_border:
            p.setPen(QPen(QColor(BORDER), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(0.75, 0.75, full_w - 1.5, full_h - 1.5), 4, 4)

        # Clip painting to the inner area
        w = full_w - 2 * ins
        h = full_h - 2 * ins
        if w < 2 or h < 2:
            p.end()
            return
        p.translate(ins, ins)
        p.setClipRect(QRectF(0, 0, w, h))

        # ── Grid ──
        duration = self.t_end - self.t_start
        if duration <= 0:
            duration = 2.5

        # Minor grid: 0.04s horizontal, adaptive vertical
        sq_x = w / (duration / 0.04) if duration > 0 else 10
        sq_y = h / 30.0  # ~30 minor divisions vertically

        # Minor lines
        p.setPen(QPen(QColor(GRID_MINOR), 0.5))
        x = 0.0
        while x < w:
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            x += sq_x
        y = 0.0
        while y < h:
            p.drawLine(QPointF(0, y), QPointF(w, y))
            y += sq_y

        # Major lines (every 5)
        p.setPen(QPen(QColor(GRID_MAJOR), 1.0))
        x = 0.0
        while x < w:
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            x += sq_x * 5
        y = 0.0
        while y < h:
            p.drawLine(QPointF(0, y), QPointF(w, y))
            y += sq_y * 5

        mid_y = h / 2.0
        mv_px = h / 3.0  # pixels per mV

        # ── Calibration pulse ──
        sig_start = 0.0
        if self.show_cal:
            cal_w = w * 0.06
            p.setPen(QPen(QColor(SIGNAL_COLOR), 1.5))
            path = QPainterPath()
            path.moveTo(3, mid_y)
            path.lineTo(5, mid_y)
            path.lineTo(5, mid_y - mv_px)
            path.lineTo(5 + cal_w, mid_y - mv_px)
            path.lineTo(5 + cal_w, mid_y)
            p.drawPath(path)
            sig_start = 5 + cal_w + 4

        # ── Annotation highlights ──
        if self.annotations:
            for a_t1, a_t2 in self.annotations:
                ax1 = sig_start + ((a_t1 - self.t_start) / duration) * (w - sig_start)
                ax2 = sig_start + ((a_t2 - self.t_start) / duration) * (w - sig_start)
                p.fillRect(QRectF(ax1, 0, ax2 - ax1, h), QColor(74, 158, 255, 30))
                p.setPen(QPen(QColor(ACCENT), 1.5, Qt.DashLine))
                p.drawLine(QPointF(ax1, 0), QPointF(ax1, h))
                p.drawLine(QPointF(ax2, 0), QPointF(ax2, h))

        # ── Signal ──
        if self.signal is not None and len(self.signal) > 0:
            n_samples = len(self.signal)
            sig_w = w - sig_start
            if sig_w > 0:
                p.setPen(QPen(QColor(SIGNAL_COLOR), 1.5))
                path = QPainterPath()
                first = True
                for px_i in range(int(sig_w)):
                    frac = px_i / sig_w
                    t = self.t_start + frac * duration
                    sample_idx = int((t - self.t_start) / duration * n_samples)
                    sample_idx = max(0, min(sample_idx, n_samples - 1))
                    v = self.signal[sample_idx]
                    py = mid_y - v * mv_px
                    if first:
                        path.moveTo(sig_start + px_i, py)
                        first = False
                    else:
                        path.lineTo(sig_start + px_i, py)
                p.drawPath(path)

        # ── Calipers ──
        if self.calipers:
            for i, (t1, t2, color, label) in enumerate(self.calipers):
                x1 = sig_start + ((t1 - self.t_start) / duration) * (w - sig_start)
                x2 = sig_start + ((t2 - self.t_start) / duration) * (w - sig_start)
                y_off = 30 + i * 50
                pen = QPen(QColor(color), 1.5, Qt.DashLine)
                p.setPen(pen)
                p.drawLine(QPointF(x1, y_off + 30), QPointF(x1, h - 40))
                p.drawLine(QPointF(x2, y_off + 30), QPointF(x2, h - 40))
                p.setPen(QPen(QColor(color), 1.5))
                p.drawLine(QPointF(x1, y_off + 20), QPointF(x2, y_off + 20))
                # Dots
                p.setBrush(QColor(color))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(x1, y_off + 20), 4, 4)
                p.drawEllipse(QPointF(x2, y_off + 20), 4, 4)
                # Label
                p.setFont(QFont("Menlo", 10, QFont.Bold))
                fm = p.fontMetrics()
                lw = fm.horizontalAdvance(label)
                lx = (x1 + x2) / 2 - lw / 2
                p.fillRect(QRectF(lx - 4, y_off - 2, lw + 8, 18), QColor(WHITE))
                p.setPen(QColor(color))
                p.drawText(QPointF(lx, y_off + 12), label)

        # ── Sweep cursor (monitor mode) ──
        if self._sweep_pos is not None:
            sx = w * self._sweep_pos
            p.setPen(QPen(QColor(ACCENT), 2))
            p.drawLine(QPointF(sx, 0), QPointF(sx, h))
            p.fillRect(QRectF(sx + 2, 0, w - sx - 2, h), QColor(249, 250, 251, 180))

        # ── Lead label ──
        if self.show_label and self.lead_name:
            p.setFont(QFont("Menlo", 11, QFont.Bold))
            p.setPen(QColor(TEXT))
            p.fillRect(QRectF(4, 2, p.fontMetrics().horizontalAdvance(self.lead_name) + 8, 18),
                       QColor(255, 255, 255, 220))
            p.drawText(QPointF(8, 16), self.lead_name)

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.signal is not None:
            w = self.width()
            duration = self.t_end - self.t_start
            sig_start = 5 + w * 0.06 + 4 if self.show_cal else 0
            sig_w = w - sig_start
            if sig_w > 0:
                frac = (event.position().x() - sig_start) / sig_w
                t = self.t_start + frac * duration
                # Approximate voltage at this time
                n = len(self.signal)
                idx = int(frac * n)
                idx = max(0, min(idx, n - 1))
                v = self.signal[idx]
                self.clicked.emit(t, v)
        super().mousePressEvent(event)


# ── 12-Lead Grid ───────────────────────────────
class TwelveLeadGrid(QWidget):
    """4x3 grid of EKG cells + rhythm strip, matching the v2 12-lead design."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame
        self.cells: dict[str, EkgCellCanvas] = {}
        self.setStyleSheet("background: #f9fafb;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        grid_rows = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        for row_leads in grid_rows:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            for lead in row_leads:
                cell = EkgCellCanvas()
                cell.draw_border = True
                self.cells[lead] = cell
                row_layout.addWidget(cell)
            layout.addLayout(row_layout, stretch=1)

        # Rhythm strip — separated by more space
        layout.addSpacing(4)
        self.rhythm = EkgCellCanvas()
        self.rhythm.draw_border = True
        self.rhythm.setFixedHeight(100)
        layout.addWidget(self.rhythm)

    def set_signal(self, signal: np.ndarray, leads: list[str], fs: int):
        """Set real or demo signal data into all cells."""
        grid_rows = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        for r, row_leads in enumerate(grid_rows):
            for c, lead in enumerate(row_leads):
                if lead in self.cells and lead in leads:
                    lead_idx = leads.index(lead)
                    t_start = c * 2.5
                    t_end = t_start + 2.5
                    n_start = int(t_start * fs)
                    n_end = int(t_end * fs)
                    n_end = min(n_end, signal.shape[0])
                    if n_start < n_end:
                        self.cells[lead].set_data(lead, signal[n_start:n_end, lead_idx],
                                                  fs, t_start, t_end)

        # Rhythm strip: lead II, full duration
        if "II" in leads:
            ii_idx = leads.index("II")
            duration = signal.shape[0] / fs
            self.rhythm.set_data("II (rytm)", signal[:, ii_idx], fs, 0, duration)


# ── Single-Lead View ───────────────────────────
class SingleLeadCanvas(EkgCellCanvas):
    """Large single-lead view with rulers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.show_cal = True
        self.show_rulers = True

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.show_rulers:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid_y = h / 2.0
        mv_px = h / 3.0

        # Voltage ruler (left)
        p.setFont(QFont("Menlo", 9))
        p.setPen(QColor(TEXT_DIM))
        for mv in [-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0]:
            y = mid_y - mv * (mv_px * 0.5)
            if 0 <= y <= h:
                p.drawText(QPointF(4, y + 4), f"{mv:.1f}")

        # Time ruler (bottom)
        duration = self.t_end - self.t_start
        for sec in range(int(self.t_start), int(self.t_end) + 1):
            frac = (sec - self.t_start) / duration if duration > 0 else 0
            x = 50 + frac * (w - 50)
            if 0 <= x <= w:
                p.drawText(QPointF(x - 5, h - 4), f"{sec}s")

        p.end()
