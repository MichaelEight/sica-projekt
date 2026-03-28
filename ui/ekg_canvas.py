"""Custom QWidget for rendering EKG signals with paper-grid background."""
import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath, QBrush
from PySide6.QtWidgets import QWidget

import ui.theme as T
from ui.theme import LEAD_SEEDS, LEAD_AMPS


# Synthetic EKG generator
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
        self.annotations = []    # list of (t1, t2) — saved annotations
        self.hovered_annotation = None  # int index into self.annotations or None
        self.selected_annotation = None  # int index into self.annotations or None
        self.pending_marker = None   # float time_s — first click in annotation mode
        self.annotation_preview = None  # (t1, t2) — completed but unsaved region
        self.annot_mode = False  # whether annotation tool is active
        self._sweep_pos = None   # fraction 0..1 for monitor mode
        self._old_signal = None  # previous page signal data (1-D)
        self._old_t_start = 0.0
        self._old_t_end = 2.5
        self.v_min = None
        self.v_max = None
        self.show_zero_line = False
        self.analysis_region = None       # (t_start, t_end) or None
        self.analysis_clickable_end = None  # max clickable time or None
        self.autoscan_regions = []        # list of (t_start, t_end, color_code, label_lines)
        self.show_autoscan_labels = False
        self.gt_annotations = []          # list of (t_start, t_end, label_text)
        self.setMinimumSize(80, 40)
        self.setMouseTracking(True)
        self._hover_x = None

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
        self.hovered_annotation = None
        self.selected_annotation = None
        self.pending_marker = None
        self.annotation_preview = None
        self.annot_mode = False
        self._sweep_pos = None
        self._old_signal = None
        self._old_t_start = 0.0
        self._old_t_end = 2.5
        self.analysis_region = None
        self.analysis_clickable_end = None
        self.autoscan_regions = []
        self.show_autoscan_labels = False
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

    INSET = 0
    draw_border = False

    def paintEvent(self, event):
        if self.width() < 2 or self.height() < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        full_w, full_h = self.width(), self.height()
        ins = self.INSET

        # Background
        painter.fillRect(0, 0, full_w, full_h, QColor(T.WHITE))

        # Border
        if self.draw_border:
            painter.setPen(QPen(QColor(T.BORDER), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(0.75, 0.75, full_w - 1.5, full_h - 1.5), 4, 4)

        w = full_w - 2 * ins
        h = full_h - 2 * ins
        if w < 2 or h < 2:
            painter.end()
            return
        painter.translate(ins, ins)
        painter.setClipRect(QRectF(0, 0, w, h))

        # Grid
        duration = self.t_end - self.t_start
        if duration <= 0:
            duration = 2.5

        px_per_sec = w / duration if duration > 0 else 100
        sq_x = px_per_sec * 0.04
        sq_y = h / 30.0

        time_offset_px = (self.t_start % 0.04) * px_per_sec

        # Minor lines
        painter.setPen(QPen(QColor(T.GRID_MINOR), 0.5))
        x = -time_offset_px
        while x < w:
            if x >= 0:
                painter.drawLine(QPointF(x, 0), QPointF(x, h))
            x += sq_x
        y = 0.0
        while y < h:
            painter.drawLine(QPointF(0, y), QPointF(w, y))
            y += sq_y

        # Major lines (every 5 minor = 0.2s)
        major_offset_px = (self.t_start % 0.2) * px_per_sec
        painter.setPen(QPen(QColor(T.GRID_MAJOR), 1.0))
        x = -major_offset_px
        while x < w:
            if x >= 0:
                painter.drawLine(QPointF(x, 0), QPointF(x, h))
            x += sq_x * 5
        y = 0.0
        while y < h:
            painter.drawLine(QPointF(0, y), QPointF(w, y))
            y += sq_y * 5

        margin = 0.05 * h
        if self.v_min is not None and self.v_max is not None:
            vmin, vmax = self.v_min, self.v_max
        else:
            vmin, vmax = -1.5, 1.5
        v_range = vmax - vmin if vmax != vmin else 3.0
        mv_px = (h - 2 * margin) / v_range

        def v_to_y(v):
            return margin + (vmax - v) * mv_px

        mid_y = v_to_y(0)

        # Calibration pulse
        sig_start = 0.0
        if self.show_cal:
            cal_w = w * 0.06
            cal_base_y = v_to_y(0)
            cal_top_y = v_to_y(1.0)
            painter.setPen(QPen(QColor(T.SIGNAL_COLOR), 1.5))
            path = QPainterPath()
            path.moveTo(3, cal_base_y)
            path.lineTo(5, cal_base_y)
            path.lineTo(5, cal_top_y)
            path.lineTo(5 + cal_w, cal_top_y)
            path.lineTo(5 + cal_w, cal_base_y)
            painter.drawPath(path)
            sig_start = 5 + cal_w + 4

        # Saved annotation highlights — normal / hovered / selected
        if self.annotations:
            for ai, (a_t1, a_t2) in enumerate(self.annotations):
                ax1 = sig_start + ((a_t1 - self.t_start) / duration) * (w - sig_start)
                ax2 = sig_start + ((a_t2 - self.t_start) / duration) * (w - sig_start)
                is_selected = (ai == self.selected_annotation)
                is_hovered = (ai == self.hovered_annotation) and not is_selected

                if is_selected:
                    # Selected: strong blue fill, thick solid border
                    painter.fillRect(QRectF(ax1, 0, ax2 - ax1, h), QColor(74, 158, 255, 60))
                    painter.setPen(QPen(QColor(T.ACCENT), 2.5))
                elif is_hovered:
                    # Hovered: medium fill, dashed border
                    painter.fillRect(QRectF(ax1, 0, ax2 - ax1, h), QColor(74, 158, 255, 45))
                    painter.setPen(QPen(QColor(T.ACCENT), 1.5, Qt.DashLine))
                else:
                    # Normal: light fill, thin solid border
                    painter.fillRect(QRectF(ax1, 0, ax2 - ax1, h), QColor(74, 158, 255, 25))
                    painter.setPen(QPen(QColor(T.ACCENT), 1.0))
                painter.drawLine(QPointF(ax1, 0), QPointF(ax1, h))
                painter.drawLine(QPointF(ax2, 0), QPointF(ax2, h))

        # Annotation preview (completed but unsaved — dashed purple)
        if self.annotation_preview:
            ap_t1, ap_t2 = self.annotation_preview
            apx1 = sig_start + ((ap_t1 - self.t_start) / duration) * (w - sig_start)
            apx2 = sig_start + ((ap_t2 - self.t_start) / duration) * (w - sig_start)
            painter.fillRect(QRectF(apx1, 0, apx2 - apx1, h), QColor(139, 92, 246, 35))
            painter.setPen(QPen(QColor(139, 92, 246), 2.0, Qt.DashLine))
            painter.drawLine(QPointF(apx1, 0), QPointF(apx1, h))
            painter.drawLine(QPointF(apx2, 0), QPointF(apx2, h))

        # Pending marker + live preview (first click placed, following mouse)
        if self.pending_marker is not None and self.annot_mode:
            mk_x = sig_start + ((self.pending_marker - self.t_start) / duration) * (w - sig_start)
            painter.setPen(QPen(QColor(139, 92, 246), 2.0))
            painter.drawLine(QPointF(mk_x, 0), QPointF(mk_x, h))
            # Live fill to hover position
            if self._hover_x is not None:
                hx = self._hover_x
                left_x = min(mk_x, hx)
                right_x = max(mk_x, hx)
                painter.fillRect(QRectF(left_x, 0, right_x - left_x, h), QColor(139, 92, 246, 18))
                painter.setPen(QPen(QColor(139, 92, 246, 120), 1.0, Qt.DotLine))
                painter.drawLine(QPointF(hx, 0), QPointF(hx, h))

        # Zero line
        if self.show_zero_line:
            zero_y = v_to_y(0)
            if 0 <= zero_y <= h:
                painter.setPen(QPen(QColor(T.TEXT_DIM), 2.0))
                painter.drawLine(QPointF(0, zero_y), QPointF(w, zero_y))

        # Signal
        if self.signal is not None and len(self.signal) > 0:
            n_samples = len(self.signal)
            sig_w = w - sig_start
            if sig_w > 0:
                if self._sweep_pos is not None:
                    draw_end = int(sig_w * self._sweep_pos)
                else:
                    draw_end = int(sig_w)

                painter.setPen(QPen(QColor(T.SIGNAL_COLOR), 1.5))
                path = QPainterPath()
                first = True
                for px_i in range(draw_end):
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
                painter.drawPath(path)

        # Calipers
        if self.calipers:
            for i, (t1, t2, color, label) in enumerate(self.calipers):
                x1 = sig_start + ((t1 - self.t_start) / duration) * (w - sig_start)
                x2 = sig_start + ((t2 - self.t_start) / duration) * (w - sig_start)
                y_off = 30 + i * 50
                pen = QPen(QColor(color), 1.5, Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(x1, y_off + 30), QPointF(x1, h - 40))
                painter.drawLine(QPointF(x2, y_off + 30), QPointF(x2, h - 40))
                painter.setPen(QPen(QColor(color), 1.5))
                painter.drawLine(QPointF(x1, y_off + 20), QPointF(x2, y_off + 20))
                painter.setBrush(QColor(color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(x1, y_off + 20), 4, 4)
                painter.drawEllipse(QPointF(x2, y_off + 20), 4, 4)
                painter.setFont(QFont("Menlo", 10, QFont.Bold))
                fm = painter.fontMetrics()
                lw = fm.horizontalAdvance(label)
                lx = (x1 + x2) / 2 - lw / 2
                painter.fillRect(QRectF(lx - 4, y_off - 2, lw + 8, 18), QColor(T.WHITE))
                painter.setPen(QColor(color))
                painter.drawText(QPointF(lx, y_off + 12), label)

        # Sweep cursor + old data (monitor mode)
        if self._sweep_pos is not None:
            sx = sig_start + (w - sig_start) * self._sweep_pos

            if self._old_signal is not None and len(self._old_signal) > 0:
                old_n = len(self._old_signal)
                old_dur = self._old_t_end - self._old_t_start
                if old_dur > 0 and (w - sig_start) > 0:
                    painter.setPen(QPen(QColor(T.SIGNAL_COLOR).lighter(170), 1.0))
                    path_old = QPainterPath()
                    first_old = True
                    gap_end = int(sx - sig_start) + 10
                    for px_i in range(gap_end, int(w - sig_start)):
                        frac = px_i / (w - sig_start)
                        t = self._old_t_start + frac * old_dur
                        si = int(t * self.fs)
                        si = max(0, min(si, old_n - 1))
                        py = v_to_y(self._old_signal[si])
                        if first_old:
                            path_old.moveTo(sig_start + px_i, py)
                            first_old = False
                        else:
                            path_old.lineTo(sig_start + px_i, py)
                    painter.drawPath(path_old)

            gap = 10
            painter.fillRect(QRectF(sx - gap, 0, gap * 2 + 2, h), QColor(T.WHITE))

            painter.setPen(QPen(QColor(T.ACCENT), 2))
            painter.drawLine(QPointF(sx, 0), QPointF(sx, h))

        # Autoscan colored regions
        if self.autoscan_regions:
            cal_w_a = w * 0.06 if self.show_cal else 0
            sig_s_a = (5 + cal_w_a + 4) if self.show_cal else 0
            sig_w_a = w - sig_s_a
            if sig_w_a > 0 and duration > 0:
                _AUTOSCAN_COLORS = {
                    1: QColor(250, 204, 21, 45),   # yellow (borderline)
                    2: QColor(239, 68, 68, 40),     # red (illness)
                }
                for region in self.autoscan_regions:
                    ar_s, ar_e, code = region[0], region[1], region[2]
                    label_lines = region[3] if len(region) > 3 else None
                    if code == 0:
                        continue  # healthy = no overlay
                    ax1 = sig_s_a + ((ar_s - self.t_start) / duration) * sig_w_a
                    ax2 = sig_s_a + ((ar_e - self.t_start) / duration) * sig_w_a
                    ax1 = max(sig_s_a, ax1)
                    ax2 = min(float(w), ax2)
                    if ax2 > ax1:
                        painter.fillRect(QRectF(ax1, 0, ax2 - ax1, h),
                                         _AUTOSCAN_COLORS[min(code, 2)])

                # Draw autoscan labels — second pass after all regions
                if self.show_autoscan_labels:
                    painter.setFont(QFont("Menlo", 14, QFont.Bold))
                    fm = painter.fontMetrics()
                    line_h = fm.height()
                    label_idx = 0
                    for region in self.autoscan_regions:
                        ar_s, ar_e, code = region[0], region[1], region[2]
                        label_lines = region[3] if len(region) > 3 else None
                        if code == 0 or not label_lines:
                            label_idx += 1
                            continue

                        total_h = line_h * len(label_lines) + 6
                        max_tw = max(fm.horizontalAdvance(ln) for ln in label_lines)
                        pill_w = max_tw + 16

                        # Anchor in the non-overlapping first half of the region
                        # (step is half the window, so first half is unique to this region)
                        t_anchor = ar_s + (ar_e - ar_s) * 0.25
                        cx = sig_s_a + ((t_anchor - self.t_start) / duration) * sig_w_a
                        pill_x = cx - pill_w / 2

                        # Always at bottom
                        by = h - total_h - 6

                        label_idx += 1

                        if pill_x + pill_w > sig_s_a and pill_x < w:
                            bg = QColor(T.WHITE)
                            bg.setAlpha(220)
                            painter.fillRect(
                                QRectF(pill_x, by - 3, pill_w, total_h + 6), bg)
                            painter.setPen(QColor(T.TEXT))
                            for li, line in enumerate(label_lines):
                                tw = fm.horizontalAdvance(line)
                                tx = pill_x + (pill_w - tw) / 2
                                ty = by + (li + 1) * line_h
                                painter.drawText(QPointF(tx, ty), line)

        # Ground truth annotation brackets
        if self.gt_annotations and self.show_autoscan_labels:
            cal_w_g = w * 0.06 if self.show_cal else 0
            sig_s_g = (5 + cal_w_g + 4) if self.show_cal else 0
            sig_w_g = w - sig_s_g
            if sig_w_g > 0 and duration > 0:
                gt_color = QColor(T.GREEN)
                painter.setFont(QFont("Menlo", 10, QFont.Bold))
                fm_gt = painter.fontMetrics()
                tick_h = 8
                bracket_y = 28  # below lead label
                for gt_s, gt_e, gt_label in self.gt_annotations:
                    gx1 = sig_s_g + ((gt_s - self.t_start) / duration) * sig_w_g
                    gx2 = sig_s_g + ((gt_e - self.t_start) / duration) * sig_w_g
                    gx1 = max(sig_s_g, gx1)
                    gx2 = min(float(w), gx2)
                    if gx2 <= gx1:
                        continue
                    # Bracket: vertical ticks + horizontal line
                    painter.setPen(QPen(gt_color, 1.5))
                    painter.drawLine(QPointF(gx1, bracket_y), QPointF(gx1, bracket_y + tick_h))
                    painter.drawLine(QPointF(gx2, bracket_y), QPointF(gx2, bracket_y + tick_h))
                    line_y = bracket_y + tick_h / 2
                    painter.drawLine(QPointF(gx1, line_y), QPointF(gx2, line_y))
                    # Label centered above the line
                    tw = fm_gt.horizontalAdvance(gt_label)
                    tx = (gx1 + gx2) / 2 - tw / 2
                    ty = bracket_y - 3
                    # Background pill for readability
                    pill_bg = QColor(T.WHITE)
                    pill_bg.setAlpha(220)
                    painter.fillRect(QRectF(tx - 4, ty - fm_gt.ascent(), tw + 8, fm_gt.height() + 2), pill_bg)
                    painter.setPen(gt_color)
                    painter.drawText(QPointF(tx, ty), gt_label)

        # Analysis overlay
        if self.analysis_clickable_end is not None or self.analysis_region is not None:
            cal_w = w * 0.06 if self.show_cal else 0
            sig_s = (5 + cal_w + 4) if self.show_cal else 0
            sig_w = w - sig_s
            if sig_w > 0 and duration > 0:
                def t_to_x(t_val):
                    return sig_s + ((t_val - self.t_start) / duration) * sig_w

                # Grayed-out unclickable zone
                if self.analysis_clickable_end is not None:
                    ce = self.analysis_clickable_end
                    if ce < self.t_end:
                        gx = t_to_x(ce)
                        gray = QColor(128, 128, 128, 60)
                        painter.fillRect(QRectF(gx, 0, w - gx, h), gray)

                # Selected 10s region
                if self.analysis_region is not None:
                    ar_start, ar_end = self.analysis_region
                    ax1 = max(sig_s, t_to_x(ar_start))
                    ax2 = min(float(w), t_to_x(ar_end))
                    if ax2 > ax1:
                        sel = QColor(74, 158, 255, 35)
                        painter.fillRect(QRectF(ax1, 0, ax2 - ax1, h), sel)
                        painter.setPen(QPen(QColor(T.ACCENT), 1.5, Qt.DashLine))
                        painter.drawLine(QPointF(ax1, 0), QPointF(ax1, h))
                        painter.drawLine(QPointF(ax2, 0), QPointF(ax2, h))
                        # Label
                        label = f"{ar_start:.1f} – {ar_end:.1f} s"
                        painter.setFont(QFont("Menlo", 8))
                        painter.setPen(QColor(T.ACCENT))
                        lbl_bg = QColor(T.WHITE)
                        lbl_bg.setAlpha(200)
                        lw = painter.fontMetrics().horizontalAdvance(label) + 6
                        lx = (ax1 + ax2) / 2 - lw / 2
                        painter.fillRect(QRectF(lx, 2, lw, 14), lbl_bg)
                        painter.drawText(QPointF(lx + 3, 12), label)

        # Lead label
        if self.show_label and self.lead_name:
            painter.setFont(QFont("Menlo", 11, QFont.Bold))
            painter.setPen(QColor(T.TEXT))
            bg = QColor(T.WHITE)
            bg.setAlpha(220)
            painter.fillRect(QRectF(4, 2, painter.fontMetrics().horizontalAdvance(self.lead_name) + 8, 18), bg)
            painter.drawText(QPointF(8, 16), self.lead_name)

        # Hover vertical cursor line
        if self._hover_x is not None:
            hx = self._hover_x - ins  # adjust for inset translation
            if 0 <= hx <= w:
                if T.is_dark_mode():
                    hover_color = QColor(255, 255, 255, 128)
                else:
                    hover_color = QColor(0, 0, 0, 77)
                hover_pen = QPen(hover_color, 1.0, Qt.DashLine)
                painter.setPen(hover_pen)
                painter.drawLine(QPointF(hx, 0), QPointF(hx, h))

        painter.end()

    annot_canceled = Signal()  # emitted on RMB cancel in annotation mode

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self.annot_mode:
            self.pending_marker = None
            self.annotation_preview = None
            self.annot_canceled.emit()
            self.update()
            return
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

    def mouseMoveEvent(self, event):
        self._hover_x = event.position().x()
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_x = None
        self.update()
        super().leaveEvent(event)


# 12-Lead Grid
class TwelveLeadGrid(QWidget):
    """4x3 grid of EKG cells + rhythm strip, matching the v2 12-lead design."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame
        self.cells: dict[str, EkgCellCanvas] = {}
        self.setStyleSheet(f"background: {T.BG_SECONDARY};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        grid_rows = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        self._separators = []
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
                    vsep.setStyleSheet(f"background: {T.SEPARATOR};")
                    self._separators.append(vsep)
                    row_layout.addWidget(vsep)
            layout.addLayout(row_layout, stretch=1)
            if ri < len(grid_rows) - 1:
                hsep = QFrame()
                hsep.setFixedHeight(3)
                hsep.setStyleSheet(f"background: {T.SEPARATOR};")
                self._separators.append(hsep)
                layout.addWidget(hsep)

        self._rhythm_sep = QFrame()
        self._rhythm_sep.setFixedHeight(3)
        self._rhythm_sep.setStyleSheet(f"background: {T.ACCENT};")
        layout.addWidget(self._rhythm_sep)
        self.rhythm = EkgCellCanvas()
        self.rhythm.draw_border = False
        self.rhythm.INSET = 1
        self.rhythm.setFixedHeight(100)
        layout.addWidget(self.rhythm)

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.BG_SECONDARY};")
        for sep in self._separators:
            sep.setStyleSheet(f"background: {T.SEPARATOR};")
        self._rhythm_sep.setStyleSheet(f"background: {T.ACCENT};")
        for cell in self.cells.values():
            cell.update()
        self.rhythm.update()

    def set_analysis_overlay(self, region: tuple | None, clickable_end: float | None):
        """Set analysis overlay on all cells."""
        for cell in self.cells.values():
            cell.analysis_region = region
            cell.analysis_clickable_end = clickable_end
            cell.update()
        self.rhythm.analysis_region = region
        self.rhythm.analysis_clickable_end = clickable_end
        self.rhythm.update()

    def clear_analysis_overlay(self):
        self.set_analysis_overlay(None, None)

    def set_autoscan_regions(self, regions: list):
        """Set colored autoscan regions on all cells. Labels only on rhythm strip."""
        for cell in self.cells.values():
            cell.autoscan_regions = regions
            cell.show_autoscan_labels = False
            cell.update()
        self.rhythm.autoscan_regions = regions
        self.rhythm.show_autoscan_labels = True
        self.rhythm.update()

    def set_gt_annotations(self, gt_lines: list):
        """Set ground truth annotation brackets on rhythm strip only."""
        self.rhythm.gt_annotations = gt_lines
        self.rhythm.update()

    def clear_autoscan_regions(self):
        self.set_autoscan_regions([])
        self.rhythm.gt_annotations = []

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

        if v_min is None or v_max is None:
            global_min = float(signal.min())
            global_max = float(signal.max())
            pad = max((global_max - global_min) * 0.15, 0.2)
            v_min = global_min - pad
            v_max = global_max + pad

        for lead, cell in self.cells.items():
            if lead not in leads:
                cell.clear()
                cell.lead_name = lead

        for r, row_leads in enumerate(grid_rows):
            for c, lead in enumerate(row_leads):
                if lead in self.cells and lead in leads:
                    lead_idx = leads.index(lead)
                    self.cells[lead].v_min = v_min
                    self.cells[lead].v_max = v_max
                    self.cells[lead].set_data(lead, signal[:, lead_idx],
                                              fs, t_start, t_end)

        # Rhythm strip: lead II
        if "II" in leads:
            ii_idx = leads.index("II")
            self.rhythm.v_min = v_min
            self.rhythm.v_max = v_max
            self.rhythm.set_data("II (rytm)", signal[:, ii_idx], fs, t_start, t_end)
        else:
            self.rhythm.clear()


# Single-Lead View
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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        margin = 0.05 * h
        if self.v_min is not None and self.v_max is not None:
            vmin, vmax = self.v_min, self.v_max
        else:
            vmin, vmax = -1.5, 1.5
        v_range = vmax - vmin if vmax != vmin else 3.0
        mv_px = (h - 2 * margin) / v_range

        def v_to_y(v):
            return margin + (vmax - v) * mv_px

        # Voltage ruler (left)
        painter.setFont(QFont("Menlo", 9))
        painter.setPen(QColor(T.TEXT_DIM))
        step = 0.5
        mv = math.ceil(vmin / step) * step
        while mv <= vmax:
            y = v_to_y(mv)
            if 0 <= y <= h:
                painter.drawText(QPointF(4, y + 4), f"{mv:.1f}")
            mv += step

        if self.signal is not None and len(self.signal) > 0:
            sig_min = float(self.signal.min())
            sig_max = float(self.signal.max())
            painter.setPen(QPen(QColor(T.ACCENT), 2.0))
            painter.setFont(QFont("Menlo", 8, QFont.Bold))
            for val, label in [(sig_max, f"{sig_max:.2f}"), (sig_min, f"{sig_min:.2f}")]:
                y = v_to_y(val)
                if 0 <= y <= h:
                    tri = QPainterPath()
                    tri.moveTo(0, y)
                    tri.lineTo(6, y - 4)
                    tri.lineTo(6, y + 4)
                    tri.closeSubpath()
                    painter.setBrush(QColor(T.ACCENT))
                    painter.drawPath(tri)
                    painter.setBrush(Qt.NoBrush)

        # Time ruler (bottom)
        duration = self.t_end - self.t_start
        if duration > 0:
            px_per_sec = w / duration
            for sec in range(int(self.t_start), int(self.t_end) + 2):
                x = (sec - self.t_start) * px_per_sec
                if 0 <= x <= w:
                    painter.drawText(QPointF(x - 5, h - 4), f"{sec}s")

        painter.end()
