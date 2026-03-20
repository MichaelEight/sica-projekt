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
        self.v_min = None        # voltage range override (mV)
        self.v_max = None
        self.show_zero_line = False
        self.setMinimumSize(80, 40)

    def clear(self):
        """Reset all data so the cell draws empty."""
        self.lead_name = ""
        self.signal = None
        self.fs = 500
        self.t_start = 0.0
        self.t_end = 2.5
        self.v_min = None
        self.v_max = None
        self.calipers = []
        self.annotations = []
        self._sweep_pos = None
        self.update()

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
        px_per_sec = w / duration if duration > 0 else 100
        sq_x = px_per_sec * 0.04  # 0.04s per minor division
        sq_y = h / 30.0  # ~30 minor divisions vertically

        # Horizontal offset: align grid lines to absolute time
        time_offset_px = (self.t_start % 0.04) * px_per_sec

        # Minor lines
        p.setPen(QPen(QColor(GRID_MINOR), 0.5))
        x = -time_offset_px
        while x < w:
            if x >= 0:
                p.drawLine(QPointF(x, 0), QPointF(x, h))
            x += sq_x
        y = 0.0
        while y < h:
            p.drawLine(QPointF(0, y), QPointF(w, y))
            y += sq_y

        # Major lines (every 5 minor = 0.2s)
        major_offset_px = (self.t_start % 0.2) * px_per_sec
        p.setPen(QPen(QColor(GRID_MAJOR), 1.0))
        x = -major_offset_px
        while x < w:
            if x >= 0:
                p.drawLine(QPointF(x, 0), QPointF(x, h))
            x += sq_x * 5
        y = 0.0
        while y < h:
            p.drawLine(QPointF(0, y), QPointF(w, y))
            y += sq_y * 5

        # Vertical mapping: voltage -> pixel y
        margin = 0.05 * h  # 5% padding top/bottom
        if self.v_min is not None and self.v_max is not None:
            vmin, vmax = self.v_min, self.v_max
        else:
            vmin, vmax = -1.5, 1.5
        v_range = vmax - vmin if vmax != vmin else 3.0
        mv_px = (h - 2 * margin) / v_range  # pixels per mV

        def v_to_y(v):
            return margin + (vmax - v) * mv_px

        mid_y = v_to_y(0)

        # ── Calibration pulse ──
        sig_start = 0.0
        if self.show_cal:
            cal_w = w * 0.06
            cal_base_y = v_to_y(0)
            cal_top_y = v_to_y(1.0)
            p.setPen(QPen(QColor(SIGNAL_COLOR), 1.5))
            path = QPainterPath()
            path.moveTo(3, cal_base_y)
            path.lineTo(5, cal_base_y)
            path.lineTo(5, cal_top_y)
            path.lineTo(5 + cal_w, cal_top_y)
            path.lineTo(5 + cal_w, cal_base_y)
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

        # ── Zero line (drawn under the signal) ──
        if self.show_zero_line:
            zero_y = v_to_y(0)
            if 0 <= zero_y <= h:
                p.setPen(QPen(QColor("#9ca3af"), 2.0))
                p.drawLine(QPointF(0, zero_y), QPointF(w, zero_y))

        # ── Signal ──
        if self.signal is not None and len(self.signal) > 0:
            n_samples = len(self.signal)
            total_duration = n_samples / self.fs if self.fs > 0 else duration
            sig_w = w - sig_start
            if sig_w > 0:
                p.setPen(QPen(QColor(SIGNAL_COLOR), 1.5))
                path = QPainterPath()
                first = True
                for px_i in range(int(sig_w)):
                    frac = px_i / sig_w
                    t = self.t_start + frac * duration
                    sample_idx = int(t * self.fs)
                    sample_idx = max(0, min(sample_idx, n_samples - 1))
                    v = self.signal[sample_idx]
                    py = v_to_y(v)
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
            if sig_w > 0 and duration > 0:
                frac = (event.position().x() - sig_start) / sig_w
                t = self.t_start + frac * duration
                idx = int(t * self.fs)
                idx = max(0, min(idx, len(self.signal) - 1))
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        grid_rows = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        for ri, row_leads in enumerate(grid_rows):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(0)
            for ci, lead in enumerate(row_leads):
                cell = EkgCellCanvas()
                cell.draw_border = False
                cell.INSET = 1
                self.cells[lead] = cell
                row_layout.addWidget(cell)
                if ci < len(row_leads) - 1:
                    vsep = QFrame()
                    vsep.setFixedWidth(3)
                    vsep.setStyleSheet("background: #6b7280;")
                    row_layout.addWidget(vsep)
            layout.addLayout(row_layout, stretch=1)
            if ri < len(grid_rows) - 1:
                hsep = QFrame()
                hsep.setFixedHeight(3)
                hsep.setStyleSheet("background: #6b7280;")
                layout.addWidget(hsep)

        # Rhythm strip — separated visually
        hsep = QFrame()
        hsep.setFixedHeight(3)
        hsep.setStyleSheet(f"background: {ACCENT};")
        layout.addWidget(hsep)
        self.rhythm = EkgCellCanvas()
        self.rhythm.draw_border = False
        self.rhythm.INSET = 1
        self.rhythm.setFixedHeight(100)
        layout.addWidget(self.rhythm)

    def clear(self):
        """Clear all cells."""
        for cell in self.cells.values():
            cell.clear()
        self.rhythm.clear()

    def set_signal(self, signal: np.ndarray, leads: list[str], fs: int,
                   time_pos: float = 0.0, window: float = 2.5,
                   v_min: float = None, v_max: float = None):
        """Set real or demo signal data into all cells."""
        grid_rows = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        duration = signal.shape[0] / fs
        t_start = max(0.0, time_pos)
        t_end = min(duration, t_start + window)
        if t_end - t_start < window:
            t_start = max(0.0, t_end - window)

        # Voltage range: use provided or compute from signal
        if v_min is None or v_max is None:
            global_min = float(signal.min())
            global_max = float(signal.max())
            pad = max((global_max - global_min) * 0.15, 0.2)
            v_min = global_min - pad
            v_max = global_max + pad

        # Clear cells for leads not in this file
        for lead, cell in self.cells.items():
            if lead not in leads:
                cell.clear()
                cell.lead_name = lead  # keep label to show it's empty

        for r, row_leads in enumerate(grid_rows):
            for c, lead in enumerate(row_leads):
                if lead in self.cells and lead in leads:
                    lead_idx = leads.index(lead)
                    self.cells[lead].v_min = v_min
                    self.cells[lead].v_max = v_max
                    self.cells[lead].set_data(lead, signal[:, lead_idx],
                                              fs, t_start, t_end)

        # Rhythm strip: lead II, full duration
        if "II" in leads:
            ii_idx = leads.index("II")
            self.rhythm.v_min = v_min
            self.rhythm.v_max = v_max
            self.rhythm.set_data("II (rytm)", signal[:, ii_idx], fs, t_start, t_end)
        else:
            self.rhythm.clear()


# ── Single-Lead View ───────────────────────────
class SingleLeadCanvas(EkgCellCanvas):
    """Large single-lead view with rulers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.show_cal = True
        self.show_rulers = True
        self.show_zero_line = True

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.show_rulers:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Use same vertical mapping as parent
        margin = 0.05 * h
        if self.v_min is not None and self.v_max is not None:
            vmin, vmax = self.v_min, self.v_max
        else:
            vmin, vmax = -1.5, 1.5
        v_range = vmax - vmin if vmax != vmin else 3.0
        mv_px = (h - 2 * margin) / v_range

        def v_to_y(v):
            return margin + (vmax - v) * mv_px

        # Voltage ruler (left) — generate ticks at 0.5 mV steps
        p.setFont(QFont("Menlo", 9))
        p.setPen(QColor(TEXT_DIM))
        step = 0.5
        mv = math.ceil(vmin / step) * step
        while mv <= vmax:
            y = v_to_y(mv)
            if 0 <= y <= h:
                p.drawText(QPointF(4, y + 4), f"{mv:.1f}")
            mv += step

        # Min/max markers on left edge
        if self.signal is not None and len(self.signal) > 0:
            sig_min = float(self.signal.min())
            sig_max = float(self.signal.max())
            p.setPen(QPen(QColor(ACCENT), 2.0))
            p.setFont(QFont("Menlo", 8, QFont.Bold))
            for val, label in [(sig_max, f"{sig_max:.2f}"), (sig_min, f"{sig_min:.2f}")]:
                y = v_to_y(val)
                if 0 <= y <= h:
                    # Small triangle marker
                    tri = QPainterPath()
                    tri.moveTo(0, y)
                    tri.lineTo(6, y - 4)
                    tri.lineTo(6, y + 4)
                    tri.closeSubpath()
                    p.setBrush(QColor(ACCENT))
                    p.drawPath(tri)
                    p.setBrush(Qt.NoBrush)

        # Time ruler (bottom)
        duration = self.t_end - self.t_start
        if duration > 0:
            px_per_sec = w / duration
            for sec in range(int(self.t_start), int(self.t_end) + 2):
                x = (sec - self.t_start) * px_per_sec
                if 0 <= x <= w:
                    p.drawText(QPointF(x - 5, h - 4), f"{sec}s")

        p.end()
