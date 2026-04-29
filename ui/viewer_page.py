"""Main EKG viewer page with toolbar, view modes, panels, navigation bar."""
import glob
import hashlib
import json
import os
import time
import sys

import threading
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property, QPoint
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QStackedWidget, QSlider,
                                QSizePolicy, QApplication, QDialog,
                                QGraphicsOpacityEffect)

import ui.theme as T
from ui.theme import STANDARD_LEADS, TARGET_CLASSES
from ui.widgets import make_logo, make_separator, TimelineOverview
from ui.ekg_canvas import (EkgCellCanvas, TwelveLeadGrid, SingleLeadCanvas,
                            generate_demo_signal, synth_ekg, LEAD_SEEDS, LEAD_AMPS)
from ui.panels import InfoPanel, MonitorSidebar
from marking_store import MarkingStore, Marking, MARKING_STYLES
from ui.context_menu import SelectionContextMenu
from ui.markings_panel import MarkingsPanel
from PySide6.QtGui import QCursor
from ecg_measurements import compute_measurements


class _CheckmarkWidget(QWidget):
    """Animated green circle with white checkmark."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self._progress = 0.0  # 0..1 controls checkmark draw length

    def _get_progress(self):
        return self._progress

    def _set_progress(self, v):
        self._progress = v
        self.update()

    progress = Property(float, _get_progress, _set_progress)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # Green circle
        p.setBrush(QBrush(QColor("#22c55e")))
        p.setPen(Qt.NoPen)
        p.drawEllipse(2, 2, 68, 68)
        # White checkmark
        if self._progress > 0:
            pen = QPen(QColor("#ffffff"), 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            # Checkmark: short stroke (20,38)->(32,50), long stroke (32,50)->(52,26)
            seg1_len = 0.35  # first segment takes 35% of animation
            if self._progress <= seg1_len:
                frac = self._progress / seg1_len
                x = 20 + (32 - 20) * frac
                y = 38 + (50 - 38) * frac
                p.drawLine(20, 38, int(x), int(y))
            else:
                p.drawLine(20, 38, 32, 50)
                frac = (self._progress - seg1_len) / (1.0 - seg1_len)
                x = 32 + (52 - 32) * frac
                y = 50 + (26 - 50) * frac
                p.drawLine(32, 50, int(x), int(y))
        p.end()


class AutoscanOverlay(QDialog):
    """Modal overlay shown during autoscan processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setFixedSize(320, 200)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        # Spinner / checkmark area
        self._spinner_label = QLabel("⏳")
        self._spinner_label.setAlignment(Qt.AlignCenter)
        self._spinner_label.setStyleSheet("font-size: 48px; background: transparent;")
        layout.addWidget(self._spinner_label, alignment=Qt.AlignCenter)

        self._checkmark = _CheckmarkWidget()
        self._checkmark.hide()
        layout.addWidget(self._checkmark, alignment=Qt.AlignCenter)

        self._text = QLabel("Trwa analiza...")
        self._text.setAlignment(Qt.AlignCenter)
        self._text.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #1e293b; background: transparent;"
        )
        layout.addWidget(self._text)

        self._subtext = QLabel("")
        self._subtext.setAlignment(Qt.AlignCenter)
        self._subtext.setStyleSheet(
            "font-size: 12px; color: #64748b; background: transparent;"
        )
        layout.addWidget(self._subtext)

        # Pulse animation on the spinner
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(500)
        self._pulse_frames = ["⏳", "⌛"]
        self._pulse_idx = 0
        self._pulse_timer.timeout.connect(self._pulse_tick)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(QColor(255, 255, 255, 240))
        p.setPen(QPen(QColor("#e2e8f0"), 1))
        p.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 16, 16)
        p.end()

    def show_loading(self, progress_text=""):
        self._spinner_label.show()
        self._checkmark.hide()
        self._text.setText("Trwa analiza...")
        self._subtext.setText(progress_text)
        self._pulse_timer.start()
        self._center_on_parent()
        self.show()

    def update_progress(self, text):
        self._subtext.setText(text)

    def show_done(self):
        self._pulse_timer.stop()
        self._spinner_label.hide()
        self._checkmark.show()
        self._checkmark._progress = 0.0
        self._text.setText("Analiza wykonana")
        self._subtext.setText("")

        # Animate the checkmark
        self._check_anim = QPropertyAnimation(self._checkmark, b"progress")
        self._check_anim.setDuration(400)
        self._check_anim.setStartValue(0.0)
        self._check_anim.setEndValue(1.0)
        self._check_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._check_anim.start()

        # Auto-close after 1.2 seconds
        QTimer.singleShot(1200, self.accept)

    def _pulse_tick(self):
        self._pulse_idx = (self._pulse_idx + 1) % len(self._pulse_frames)
        self._spinner_label.setText(self._pulse_frames[self._pulse_idx])

    def _center_on_parent(self):
        parent = self.parent()
        if parent:
            pr = parent.rect()
            x = pr.center().x() - self.width() // 2
            y = pr.center().y() - self.height() // 2
            self.move(parent.mapToGlobal(QPoint(x, y)))


def get_resource_path(relative_path):
    """ Get the absolute path to a resource. Works for dev and for PyInstaller. """
    try:
        # PyInstaller creates a temp folder and stores its path in sys._MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # If not running as an exe, use the current absolute directory
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def discover_models():
    """Scan known directories for .pt checkpoint files."""
    search_dirs = ["model/annotations", "models"]
    found = []
    for d in search_dirs:
        abs_dir = get_resource_path(d)
        if os.path.isdir(abs_dir):
            found.extend(glob.glob(os.path.join(abs_dir, "*.pt")))
    found.sort(key=lambda p: (0 if "model-sota" in os.path.basename(p) else 1, p))
    return found


# Segmented Control
class SegmentedControl(QWidget):
    """Toggle button group like the HTML seg-btn design."""

    changed = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.buttons: list[QPushButton] = []
        for i, label in enumerate(labels):
            btn = QPushButton(label)
            btn.setFont(QFont(".AppleSystemUIFont", 11, QFont.Medium))
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self.set_active(idx))
            self.buttons.append(btn)
            layout.addWidget(btn)
        self._active = -1
        self.set_active(0)

    def set_active(self, idx: int):
        if idx == self._active:
            return
        self._active = idx
        self._apply_styles()
        self.changed.emit(idx)

    def _apply_styles(self):
        for i, btn in enumerate(self.buttons):
            if i == self._active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {T.ACCENT}; color: {T.ACCENT_TEXT}; border: none;
                        padding: 8px 22px; font-size: 13px; font-weight: 600;
                    }}
                    QPushButton:hover {{ background: {T.ACCENT}; opacity: 0.85; }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {T.BTN_DARK}; color: {T.TEXT_MUTED}; border: none;
                        padding: 8px 22px; font-size: 13px; font-weight: 600;
                    }}
                    QPushButton:hover {{ background: {T.SEPARATOR}; color: {T.BTN_TEXT}; }}
                """)

    def active(self) -> int:
        return self._active


# ── Toolbar Button ─────────────────────────────
class ToolbarBtn(QPushButton):
    def __init__(self, text: str, active: bool = False, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont(".AppleSystemUIFont", 11))
        self.setCursor(Qt.PointingHandCursor)
        self.set_active(active)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self.setStyleSheet(f"""
                QPushButton {{ background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;padding:6px 10px;border-radius:5px; }}
                QPushButton:hover {{ background:{T.ACCENT}; opacity:0.85; }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{ background:{T.BTN_DARK};color:{T.BTN_TEXT};border:none;padding:6px 10px;border-radius:5px; }}
                QPushButton:hover {{ background:{T.SEPARATOR};color:{T.ACCENT_TEXT}; }}
            """)

    def is_active(self) -> bool:
        return self._active


# ── Unified Left Panel ─────────────────────────
class LayoutSwitcher(QWidget):
    """Wide unified left panel: patient + measurements + layout + leads + live/speed.

    All sections inline (no subpanels/popups). Replaces SegmentedControl +
    LeadSidebar + MonitorSidebar + InfoPanel-as-drawer.
    """

    layout_changed = Signal(str)
    live_toggled = Signal(bool)
    speed_changed = Signal(float)
    focus_lead_changed = Signal(str)
    visible_leads_changed = Signal(list)

    _LAYOUTS = [
        ("grid_3x4", "3×4"),
        ("grid_2x6", "2×6"),
        ("stack_1xN", "Paski"),
        ("focus_1L", "Pojedyncze"),
    ]
    _SPEEDS = [0.5, 1.0, 2.0]

    def __init__(self, info_panel, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")

        from PySide6.QtWidgets import QScrollArea, QGridLayout
        from ui.widgets import section_header
        from ui.ekg_canvas import ALL_LEADS_ORDER

        self._layout_id = "grid_3x4"
        self._live = False
        self._speed_idx = 1
        self._focus_lead = "II"
        self._visible_leads: set[str] = set(ALL_LEADS_ORDER)
        self._info_panel = info_panel

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none;")
        outer.addWidget(scroll)

        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignTop)

        # ── Section 1: Patient + measurements (inline InfoPanel) ──
        info_panel.setFixedWidth(240)
        layout.addWidget(info_panel)

        # ── Section 2: Layout buttons ──
        layout.addWidget(section_header("Układ"))
        layout_grid = QGridLayout()
        layout_grid.setSpacing(4)
        layout_grid.setContentsMargins(0, 0, 0, 0)
        self._layout_btns: dict[str, QPushButton] = {}
        for i, (lid, label) in enumerate(self._LAYOUTS):
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFont(QFont(".AppleSystemUIFont", 10, QFont.DemiBold))
            btn.clicked.connect(lambda checked=False, l=lid: self._set_layout(l))
            self._layout_btns[lid] = btn
            layout_grid.addWidget(btn, i // 2, i % 2)
        layout.addLayout(layout_grid)

        # ── Section 3: Lead toggle grid (12 buttons, 3 cols) ──
        layout.addWidget(section_header("Odprowadzenia"))
        lead_grid = QGridLayout()
        lead_grid.setSpacing(3)
        lead_grid.setContentsMargins(0, 0, 0, 0)
        self._lead_btns: dict[str, QPushButton] = {}
        for i, lead in enumerate(ALL_LEADS_ORDER):
            btn = QPushButton(lead)
            btn.setFixedSize(72, 28)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFont(QFont("Menlo", 11, QFont.DemiBold))
            btn.clicked.connect(lambda checked=False, l=lead: self._on_lead_toggle(l))
            lead_grid.addWidget(btn, i // 3, i % 3)
            self._lead_btns[lead] = btn
        layout.addLayout(lead_grid)

        self._lead_all_btn = QPushButton("Pokaż wszystkie")
        self._lead_all_btn.setFixedHeight(22)
        self._lead_all_btn.setCursor(Qt.PointingHandCursor)
        self._lead_all_btn.setFont(QFont(".AppleSystemUIFont", 10))
        self._lead_all_btn.clicked.connect(self._select_all_leads)
        layout.addWidget(self._lead_all_btn)

        # ── Section 4: Monitor controls ──
        layout.addWidget(section_header("Podgląd ciągły"))
        self._live_btn = QPushButton("▶ Włącz")
        self._live_btn.setFixedHeight(32)
        self._live_btn.setCursor(Qt.PointingHandCursor)
        self._live_btn.setFont(QFont(".AppleSystemUIFont", 11, QFont.DemiBold))
        self._live_btn.clicked.connect(self._toggle_live)
        layout.addWidget(self._live_btn)

        speed_lbl = QLabel("Prędkość")
        speed_lbl.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED};")
        layout.addWidget(speed_lbl)
        speed_row = QHBoxLayout()
        speed_row.setSpacing(4)
        self._speed_pills: list[QPushButton] = []
        for i, spd in enumerate(self._SPEEDS):
            btn = QPushButton(f"{spd:g}×")
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFont(QFont(".AppleSystemUIFont", 11, QFont.DemiBold))
            btn.clicked.connect(lambda checked=False, idx=i: self._set_speed(idx))
            speed_row.addWidget(btn, 1)
            self._speed_pills.append(btn)
        layout.addLayout(speed_row)

        layout.addStretch()
        self._refresh_styles()

    # ── Public API ──
    def set_layout(self, layout_id: str):
        if layout_id == self._layout_id:
            return
        if layout_id not in {lid for lid, _ in self._LAYOUTS}:
            return
        self._layout_id = layout_id
        self._refresh_styles()

    def set_live(self, live: bool):
        if live == self._live:
            return
        self._live = live
        self._refresh_styles()

    def set_speed(self, speed: float):
        try:
            self._speed_idx = self._SPEEDS.index(speed)
        except ValueError:
            return
        self._refresh_styles()

    def set_focus_lead(self, lead: str):
        self._focus_lead = lead

    def set_visible_leads(self, leads: list[str]):
        self._visible_leads = set(leads)
        for lead, btn in self._lead_btns.items():
            btn.blockSignals(True)
            btn.setChecked(lead in self._visible_leads)
            btn.blockSignals(False)
        self._refresh_styles()

    def active_layout(self) -> str:
        return self._layout_id

    def is_live(self) -> bool:
        return self._live

    def speed(self) -> float:
        return self._SPEEDS[self._speed_idx]

    def visible_leads(self) -> list[str]:
        from ui.ekg_canvas import ALL_LEADS_ORDER
        return [l for l in ALL_LEADS_ORDER if l in self._visible_leads]

    # ── Internals ──
    def _set_layout(self, layout_id: str):
        if layout_id == self._layout_id:
            return
        self._layout_id = layout_id
        self._refresh_styles()
        self.layout_changed.emit(layout_id)

    def _toggle_live(self):
        self._live = not self._live
        self._refresh_styles()
        self.live_toggled.emit(self._live)

    def _set_speed(self, idx: int):
        self._speed_idx = idx
        self._refresh_styles()
        self.speed_changed.emit(self._SPEEDS[idx])

    def _on_lead_toggle(self, lead: str):
        if self._lead_btns[lead].isChecked():
            self._visible_leads.add(lead)
        else:
            self._visible_leads.discard(lead)
            if not self._visible_leads:
                self._visible_leads.add(lead)
                self._lead_btns[lead].setChecked(True)
                return
        self._refresh_styles()
        self.visible_leads_changed.emit(self.visible_leads())

    def _select_all_leads(self):
        from ui.ekg_canvas import ALL_LEADS_ORDER
        self._visible_leads = set(ALL_LEADS_ORDER)
        for lead, btn in self._lead_btns.items():
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
        self._refresh_styles()
        self.visible_leads_changed.emit(self.visible_leads())

    def _refresh_styles(self):
        active_bg = T.ACCENT
        active_fg = T.ACCENT_TEXT
        idle_bg = T.WHITE
        idle_fg = T.TEXT_MUTED
        for lid, btn in self._layout_btns.items():
            on = (lid == self._layout_id)
            btn.setStyleSheet(self._btn_style(on, active_bg, active_fg, idle_bg, idle_fg))
        for lead, btn in self._lead_btns.items():
            on = btn.isChecked()
            btn.setStyleSheet(self._btn_style(on, active_bg, active_fg, idle_bg, idle_fg))
        self._lead_all_btn.setStyleSheet(self._btn_style(False, active_bg, active_fg, idle_bg, idle_fg))
        live_on = self._live
        live_bg = T.GREEN if live_on else idle_bg
        live_fg = T.ACCENT_TEXT if live_on else idle_fg
        self._live_btn.setText("⏸ Pauza" if live_on else "▶ Włącz")
        self._live_btn.setStyleSheet(self._btn_style(live_on, live_bg, live_fg, idle_bg, idle_fg))
        for i, btn in enumerate(self._speed_pills):
            on = (i == self._speed_idx)
            btn.setStyleSheet(self._btn_style(on, active_bg, active_fg, idle_bg, idle_fg))
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")

    @staticmethod
    def _btn_style(on, on_bg, on_fg, off_bg, off_fg):
        return (
            f"QPushButton {{ background: {on_bg if on else off_bg};"
            f"  color: {on_fg if on else off_fg};"
            f"  border: 1px solid {T.BORDER if not on else on_bg};"
            f"  border-radius: 4px; font-size: 11px; }}"
            f"QPushButton:hover {{ background: {T.BG_SECONDARY if not on else on_bg}; }}"
        )

    def apply_theme(self):
        if self._info_panel is not None:
            self._info_panel.apply_theme()
        self._refresh_styles()


# ── Lead Sidebar (for 1-lead mode) ─────────────
class LeadSidebar(QWidget):
    """Sidebar with lead selection buttons."""

    lead_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(58)
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignTop)

        self.buttons: dict[str, QPushButton] = {}
        groups = [["I", "II", "III"], ["aVR", "aVL", "aVF"],
                  ["V1", "V2", "V3", "V4", "V5", "V6"]]
        for gi, group in enumerate(groups):
            if gi > 0:
                sep = QFrame()
                sep.setFixedSize(36, 1)
                sep.setStyleSheet(f"background: {T.BORDER};")
                layout.addWidget(sep, alignment=Qt.AlignCenter)
            for lead in group:
                btn = QPushButton(lead)
                btn.setFixedSize(46, 32)
                btn.setFont(QFont("Menlo", 11, QFont.DemiBold))
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda checked, l=lead: self._select(l))
                self.buttons[lead] = btn
                layout.addWidget(btn, alignment=Qt.AlignCenter)

        layout.addStretch()
        self._active = "II"
        self._update_styles()

    def _select(self, lead: str):
        self._active = lead
        self._update_styles()
        self.lead_selected.emit(lead)

    def _update_styles(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")
        for lead, btn in self.buttons.items():
            if lead == self._active:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {T.ACCENT}; color: {T.ACCENT_TEXT}; border: 1px solid {T.ACCENT};
                        border-radius: 4px; font-family: Menlo;
                    }}
                    QPushButton:hover {{ background: {T.ACCENT}; opacity: 0.85; }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {T.WHITE}; color: {T.TEXT_MUTED}; border: 1px solid {T.BORDER};
                        border-radius: 4px; font-family: Menlo;
                    }}
                    QPushButton:hover {{ background: {T.BG_SECONDARY}; color: {T.TEXT}; }}
                """)

    def active_lead(self) -> str:
        return self._active


# ── Separator ──────────────────────────────────
def _sep():
    sep = QFrame()
    sep.setFixedSize(1, 24)
    sep.setStyleSheet(f"background: {T.SEPARATOR};")
    return sep


# ── Viewer Page ────────────────────────────────
class ViewerPage(QWidget):
    """Main EKG viewer with toolbar, view modes, panels, navigation."""

    open_file = Signal()         # request to open a new file
    show_report = Signal()       # request to show report page
    toggle_dark = Signal()       # request to toggle dark mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal = None
        self.leads = STANDARD_LEADS
        self.fs = 500
        self.filename = ""
        self.duration = 0.0
        self.time_pos = 0.0
        self._scrubber_max = 0.0
        self._window_12 = 2.5
        self._window_1 = 3.0
        self._v_min = -1.5
        self._v_max = 1.5
        # Unified screen state — replaces _view_mode
        self._layout_id = "grid_3x4"  # grid_3x4 | grid_2x6 | stack_1xN | focus_1L
        self._live = False
        self._focus_lead = "II"
        self._visible_leads: list[str] = list(STANDARD_LEADS)
        self._last_clicked_lead = "II"
        self._marking_store = MarkingStore()
        self._context_menu = None
        self._pending_selection = None
        self._model = None
        self._model_device = None
        self._model_path = None
        self._last_results = None
        self._autoscan_active = False
        self._autoscan_results = None
        self._autoscan_file_path = None
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(50)
        self._monitor_timer.timeout.connect(self._monitor_tick)
        self._monitor_playing = False
        self._monitor_t = 0.0
        self._monitor_speed = 1.0
        self._monitor_window = 3.0
        self._monitor_page_start = 0.0

        self._build_ui()

    # ── Build UI ──
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Single-row top bar ──
        self.toolbar = QWidget()
        self.toolbar.setFixedHeight(40)
        self.toolbar.setStyleSheet(f"background: {T.TOPBAR};")
        tb = QHBoxLayout(self.toolbar)
        tb.setContentsMargins(10, 0, 10, 0)
        tb.setSpacing(8)

        logo = make_logo(12)
        tb.addWidget(logo)
        tb.addWidget(_sep())

        self.file_label = QLabel("")
        self.file_label.setStyleSheet(f"font-size:11px; color:{T.BTN_TEXT}; font-family:Menlo;")
        tb.addWidget(self.file_label)

        self.analysis_badge = QLabel("Analiza zakończona")
        self.analysis_badge.setFont(QFont(".AppleSystemUIFont", 11))
        self.analysis_badge.setFixedHeight(24)
        self.analysis_badge.setStyleSheet(f"""
            font-size: 11px; background: {T.BADGE_NORM_BG}; color: {T.BADGE_NORM_TEXT};
            padding: 0px 8px; border-radius: 3px; font-weight: 600;
        """)
        self.analysis_badge.hide()
        tb.addWidget(self.analysis_badge)

        tb.addStretch()

        from ui.theme import is_dark_mode as _idm
        _ah = '#047857' if _idm() else '#3a8eef'
        self.btn_full_analysis = QPushButton("Pełna analiza")
        self.btn_full_analysis.setObjectName("primary")
        self.btn_full_analysis.setCursor(Qt.PointingHandCursor)
        self.btn_full_analysis.setStyleSheet(f"""
            QPushButton {{ background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;
                padding:5px 14px;border-radius:5px;font-weight:600;font-size:12px; }}
            QPushButton:hover {{ background:{_ah}; }}
        """)
        self.btn_full_analysis.clicked.connect(self._run_full_analysis)
        tb.addWidget(self.btn_full_analysis)

        self.btn_report = QPushButton("Raport")
        self.btn_report.setStyleSheet(f"""
            background:{T.BTN_DARK};color:{T.BTN_TEXT};font-size:12px;padding:6px 14px;
            border-radius:5px;border:none;
        """)
        self.btn_report.setCursor(Qt.PointingHandCursor)
        self.btn_report.clicked.connect(self.show_report.emit)
        tb.addWidget(self.btn_report)

        tb.addWidget(make_separator())

        btn_load = QPushButton("Wczytaj plik")
        btn_load.setObjectName("secondary")
        btn_load.setCursor(Qt.PointingHandCursor)
        btn_load.clicked.connect(self.open_file.emit)
        tb.addWidget(btn_load)

        self.btn_dark = QPushButton("Tryb ciemny")
        self.btn_dark.setObjectName("secondary")
        self.btn_dark.setCursor(Qt.PointingHandCursor)
        self.btn_dark.clicked.connect(self.toggle_dark.emit)
        tb.addWidget(self.btn_dark)

        outer.addWidget(self.toolbar)

        # ── Content area ──
        self.content = QHBoxLayout()
        self.content.setContentsMargins(0, 0, 0, 0)
        self.content.setSpacing(0)

        # InfoPanel constructed first (embedded inside LayoutSwitcher inline)
        self.info_panel = InfoPanel()
        self.info_panel.patient_changed.connect(self._save_ann)

        # Wide unified left panel
        self.layout_switcher = LayoutSwitcher(self.info_panel)
        self.layout_switcher.layout_changed.connect(self._on_layout_changed)
        self.layout_switcher.live_toggled.connect(self._on_live_toggled)
        self.layout_switcher.speed_changed.connect(self._on_monitor_speed)
        self.layout_switcher.visible_leads_changed.connect(self._on_visible_leads_changed)
        self.content.addWidget(self.layout_switcher)

        # Single grid view shared by all layouts. Cells are reused across layouts.
        self.grid_view = TwelveLeadGrid()
        self.grid_12 = self.grid_view  # backward-compat alias
        if hasattr(self.grid_view, 'cell_double_clicked'):
            self.grid_view.cell_double_clicked.connect(self._on_cell_double_click)

        # Wire selection / zoom / pan / right-click for every cell + rhythm strip
        all_cells = list(self.grid_view.cells.items()) + [("II", self.grid_view.rhythm)]
        for lead, cell in all_cells:
            cell.selection_mode = True
            if hasattr(cell, 'selection_completed'):
                cell.selection_completed.connect(
                    lambda t1, t2, ln=lead: self._on_cell_selection_completed(ln, t1, t2))
            if hasattr(cell, 'selection_live'):
                cell.selection_live.connect(
                    lambda t1, t2, ln=lead: self._on_cell_selection_live(ln, t1, t2))
            if hasattr(cell, 'right_clicked'):
                cell.right_clicked.connect(self._on_canvas_right_click)
            if hasattr(cell, 'zoom_requested'):
                cell.zoom_requested.connect(self._on_pinch_zoom)
            if hasattr(cell, 'scroll_pan'):
                cell.scroll_pan.connect(self._on_scroll_pan)
            if hasattr(cell, 'clicked'):
                cell.clicked.connect(lambda t, v, ln=lead: self._on_cell_clicked(ln))

        # Compatibility shim: code that referenced single_lead now points at the
        # focus cell. Updated dynamically when focus_lead changes.
        self.single_lead = self.grid_view.cells["II"]

        self.content.addWidget(self.grid_view, stretch=1)

        # Markings panel (always visible)
        self.markings_panel = MarkingsPanel()
        self.markings_panel.marking_hovered.connect(self._on_marking_hovered)
        self.markings_panel.marking_unhovered.connect(self._on_marking_unhovered)
        self.markings_panel.marking_selected.connect(self._on_marking_selected)
        self.markings_panel.marking_deleted.connect(self._on_marking_deleted)
        self.markings_panel.marking_edited.connect(self._on_marking_edited)
        self.markings_panel.marking_focus.connect(self._on_marking_focus)
        self.markings_panel.annotation_created.connect(self._on_annotation_created)
        self.markings_panel.undo_requested.connect(self._undo)
        self.markings_panel.redo_requested.connect(self._redo)
        self.content.addWidget(self.markings_panel)

        content_widget = QWidget()
        content_widget.setLayout(self.content)
        outer.addWidget(content_widget, stretch=1)

        # Bottom bar container (navbar + selection indicator)
        self.bottom_bar = QWidget()
        self.bottom_bar.setStyleSheet(f"background: {T.WHITE}; border-top: 1px solid {T.BORDER};")
        bottom_layout = QVBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        # Row 1: Navigation bar (always visible, fixed layout)
        self.navbar = QWidget()
        self.navbar.setFixedHeight(42)
        nav = QHBoxLayout(self.navbar)
        nav.setContentsMargins(12, 0, 12, 0)
        nav.setSpacing(6)

        # Navigation buttons
        back_buttons = [
            ("\u23ee", self._nav_start),
            ("\u25c0\u25c0", lambda: self._nav_step(-1.0)),
            ("\u25c0", lambda: self._nav_step(-0.2)),
        ]
        fwd_buttons = [
            ("\u25b6", lambda: self._nav_step(0.2)),
            ("\u25b6\u25b6", lambda: self._nav_step(1.0)),
            ("\u23ed", self._nav_end),
        ]

        self._nav_btns = []
        self._hold_timer = QTimer(self)
        self._hold_timer.setInterval(80)
        self._hold_handler = None
        self._hold_timer.timeout.connect(self._on_hold_tick)

        def _make_nav(label, handler):
            btn = QPushButton(label)
            btn.setObjectName("nav")
            btn.setCursor(Qt.PointingHandCursor)
            btn.pressed.connect(lambda h=handler: self._start_hold(h))
            btn.released.connect(self._stop_hold)
            self._nav_btns.append(btn)
            nav.addWidget(btn)

        for label, handler in back_buttons:
            _make_nav(label, handler)

        self.pause_btn = QPushButton("\u275a\u275a")
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self._apply_pause_btn_style()
        self.pause_btn.clicked.connect(self._on_navbar_pause)
        self.pause_btn.setEnabled(False)  # enabled when live mode active
        nav.addWidget(self.pause_btn)

        for label, handler in fwd_buttons:
            _make_nav(label, handler)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.setValue(350)
        self.scrubber.setStyleSheet(self._scrubber_style())
        self.scrubber.valueChanged.connect(self._on_scrubber)
        nav.addWidget(self.scrubber, stretch=1)

        self.time_label = QLabel()
        self.time_label.setFixedWidth(180)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setStyleSheet(f"font-size:11px; font-family:Menlo; color:{T.TEXT_SECONDARY};")
        nav.addWidget(self.time_label)

        nav.addWidget(make_separator())

        # Zoom controls — fixed width group
        self._zoom_out_btn = QPushButton("\u2212")
        self._zoom_out_btn.setObjectName("nav")
        self._zoom_out_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        nav.addWidget(self._zoom_out_btn)

        self._zoom_label = QLabel("3.0 s")
        self._zoom_label.setFixedWidth(42)
        self._zoom_label.setStyleSheet(f"font-size:11px; font-family:Menlo; color:{T.TEXT_MUTED};")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self._zoom_label)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setObjectName("nav")
        self._zoom_in_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        nav.addWidget(self._zoom_in_btn)

        self._zoom_reset_btn = QPushButton("Reset")
        self._zoom_reset_btn.setStyleSheet(self._zoom_reset_btn_style())
        self._zoom_reset_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_reset_btn.clicked.connect(self._reset_zoom)
        nav.addWidget(self._zoom_reset_btn)

        bottom_layout.addWidget(self.navbar)

        # Row 1b: Timeline overview — full signal with window rectangle
        self.timeline_overview = TimelineOverview()
        self.timeline_overview.seek_requested.connect(self._on_overview_seek)
        self._overview_row = QWidget()
        self._overview_row.setStyleSheet(f"background: {T.WHITE};")
        orl = QHBoxLayout(self._overview_row)
        orl.setContentsMargins(12, 2, 12, 6)
        orl.setSpacing(0)
        orl.addWidget(self.timeline_overview)
        bottom_layout.addWidget(self._overview_row)

        # Row 2: Selection indicator bar — always present, fixed height
        self._sel_indicator = QLabel()
        self._sel_indicator.setAlignment(Qt.AlignCenter)
        self._sel_indicator.setFixedHeight(24)
        self._sel_indicator.setStyleSheet(f"background: {T.WHITE}; border: none; color: transparent;")
        bottom_layout.addWidget(self._sel_indicator)

        # Row 3: Persistent status strip — always visible across modes
        self.status_strip = QWidget()
        self.status_strip.setFixedHeight(26)
        self.status_strip.setStyleSheet(
            f"background: {T.BG_SECONDARY}; border-top: 1px solid {T.BORDER};"
        )
        ss = QHBoxLayout(self.status_strip)
        ss.setContentsMargins(14, 0, 14, 0)
        ss.setSpacing(16)
        self.status_file = QLabel("Brak wczytanego pliku")
        self.status_file.setStyleSheet(f"font-size:11px; color:{T.TEXT_SECONDARY}; font-family:Menlo;")
        self.status_meta = QLabel("")
        self.status_meta.setStyleSheet(f"font-size:11px; color:{T.TEXT_DIM}; font-family:Menlo;")
        self.status_pos = QLabel("")
        self.status_pos.setStyleSheet(f"font-size:11px; color:{T.TEXT_DIM}; font-family:Menlo;")
        ss.addWidget(self.status_file)
        ss.addWidget(self.status_meta)
        ss.addStretch()
        ss.addWidget(self.status_pos)
        bottom_layout.addWidget(self.status_strip)

        outer.addWidget(self.bottom_bar)
        self._update_time_display()

    def apply_theme(self):
        """Re-apply all styles after theme change."""
        # Toolbar
        self.toolbar.setStyleSheet(f"background: {T.TOPBAR};")
        self.file_label.setStyleSheet(f"font-size:11px; color:{T.BTN_TEXT}; font-family:Menlo;")
        self.analysis_badge.setStyleSheet(f"""
            font-size: 11px; background: {T.BADGE_NORM_BG}; color: {T.BADGE_NORM_TEXT};
            padding: 0px 8px; border-radius: 3px; font-weight: 600;
        """)
        from ui.theme import is_dark_mode as _idm
        _ah = '#047857' if _idm() else '#3a8eef'
        self.btn_full_analysis.setStyleSheet(f"""
            QPushButton {{ background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;
                padding:5px 14px;border-radius:5px;font-weight:600;font-size:12px; }}
            QPushButton:hover {{ background:{_ah}; }}
        """)
        from ui.theme import is_dark_mode
        self.btn_dark.setText("Tryb jasny" if is_dark_mode() else "Tryb ciemny")
        # Layout switcher (re-themes embedded info_panel too)
        self.layout_switcher.apply_theme()
        # Grid view
        self.grid_view.setStyleSheet(f"background: {T.BG_SECONDARY};")
        self.markings_panel.apply_theme()

        # Bottom controls panel
        self.bottom_bar.setStyleSheet(f"background: {T.WHITE}; border-top: 1px solid {T.BORDER};")
        self.scrubber.setStyleSheet(self._scrubber_style())
        self.time_label.setStyleSheet(f"font-size:11px; font-family:Menlo; color:{T.TEXT_SECONDARY};")
        self._zoom_label.setStyleSheet(f"font-size:11px; font-family:Menlo; color:{T.TEXT_MUTED};")
        self._zoom_reset_btn.setStyleSheet(self._zoom_reset_btn_style())
        self._sel_indicator.setStyleSheet(f"background: {T.WHITE}; border: none; color: transparent;")
        self.status_strip.setStyleSheet(
            f"background: {T.BG_SECONDARY}; border-top: 1px solid {T.BORDER};"
        )
        self.status_file.setStyleSheet(f"font-size:11px; color:{T.TEXT_SECONDARY}; font-family:Menlo;")
        self.status_meta.setStyleSheet(f"font-size:11px; color:{T.TEXT_DIM}; font-family:Menlo;")
        self.status_pos.setStyleSheet(f"font-size:11px; color:{T.TEXT_DIM}; font-family:Menlo;")
        self.timeline_overview.update()
        self._apply_pause_btn_style()

        self.grid_12.apply_theme()

    def set_signal(self, signal: np.ndarray, leads: list[str], fs: int, filename: str = "",
                   ground_truth: dict | None = None, patient_info: dict | None = None,
                   base_path: str = ""):
        """Load new signal data into the viewer."""
        self._ground_truth = ground_truth
        self._base_path = base_path
        self._marking_store.clear()
        self.grid_view.clear()

        self.signal = signal
        self.leads = leads
        self.fs = fs
        self.filename = filename
        self.duration = signal.shape[0] / fs
        self.time_pos = 0.0
        self._monitor_t = 0.0
        self._monitor_playing = False
        # Reset to default 3x4 layout, live off
        self._layout_id = "grid_3x4"
        self._live = False
        self._visible_leads = list(leads) if leads else list(STANDARD_LEADS)
        self.layout_switcher.set_layout("grid_3x4")
        self.layout_switcher.set_live(False)
        self.layout_switcher.set_visible_leads(self._visible_leads)
        self._monitor_timer.stop()
        self.pause_btn.setEnabled(False)

        # ── Analyze signal ranges once ──
        self._global_min = float(signal.min())
        self._global_max = float(signal.max())
        v_range = self._global_max - self._global_min
        pad = max(v_range * 0.15, 0.2)  # at least 0.2 mV padding
        self._v_min = self._global_min - pad
        self._v_max = self._global_max + pad

        # Adaptive time windows based on signal duration
        if self.duration <= 3.0:
            self._window_12 = self.duration
            self._window_1 = self.duration
        elif self.duration <= 10.0:
            self._window_12 = min(2.5, self.duration)
            self._window_1 = min(3.0, self.duration)
        else:
            self._window_12 = 2.5
            self._window_1 = 3.0
        self._update_zoom_label()

        self.file_label.setText(
            f"{filename} | {self.duration:.1f} s"
        )
        self.analysis_badge.hide()

        # Reset autoscan
        self._autoscan_active = False
        self._autoscan_results = None
        self._autoscan_file_path = None
        self.grid_12.clear_autoscan_regions()
        self.single_lead.autoscan_regions = []

        # Enable/disable full analysis based on duration
        min_samples = int(10.0 * self.fs)
        if self.signal.shape[0] < min_samples:
            self.btn_full_analysis.setEnabled(False)
            self.btn_full_analysis.setToolTip("")
        else:
            self.btn_full_analysis.setEnabled(True)
            self.btn_full_analysis.setToolTip("")

        max_window = max(self._window_12, self._window_1)
        self._scrubber_max = max(0.0, self.duration - max_window)
        self.scrubber.setRange(0, int(self._scrubber_max * 100))
        self.scrubber.setValue(0)

        # Timeline overview for window-rect visualization
        self.timeline_overview.set_signal(signal, self.duration)

        # Persistent status strip
        self.status_file.setText(f"Plik: {filename}" if filename else "Brak wczytanego pliku")
        self.status_meta.setText(f"{self.duration:.1f} s")

        # Load patient info
        if patient_info:
            self.info_panel.set_patient(
                patient_id=patient_info.get("id", ""),
                age=patient_info.get("age", ""),
                sex=patient_info.get("sex", ""),
                date=patient_info.get("date", ""),
            )
        else:
            self.info_panel.set_patient()

        # Compute ECG measurements
        self._measurements = {}
        try:
            pat_sex = patient_info.get("sex") if patient_info else None
            meas = compute_measurements(self.signal, self.fs, sex=pat_sex)
            self._measurements = meas
            self.info_panel.set_measurements(
                hr=meas["hr"],
                pr=meas["pr"],
                qrs=meas["qrs"],
                qt_val=meas["qt"],
                qtc=meas["qtc"],
                axis=meas["axis"],
            )
        except Exception:
            pass

        # Load .ann file (overrides patient info, loads markings)
        self._load_ann()
        # Prefer the raw window-level autoscan cache (full per-window probs)
        # over rebuilding from merged markings — markings collapse into 1 region.
        cached_raw = self._load_autoscan_cache()
        if cached_raw:
            self._autoscan_results = cached_raw
            self._apply_autoscan_overlay()
            self.analysis_badge.show()
        else:
            self._rebuild_autoscan_from_markings()

        self._refresh_views()
        self._refresh_markings()
        self._update_time_display()


    def _refresh_views(self):
        if self.signal is None:
            return
        # Apply current layout (rebuilds row layouts from spec)
        self.grid_view.set_layout(self._layout_id, self._visible_leads, self._focus_lead)
        # Refresh focus alias to point at current focused lead's cell
        if self._focus_lead in self.grid_view.cells:
            self.single_lead = self.grid_view.cells[self._focus_lead]
        # Push signal into all cells (window adapts to layout)
        win = self._current_window()
        if self._live:
            self.grid_view.set_signal(self.signal, self.leads, self.fs,
                                      self._monitor_page_start, self._monitor_window,
                                      self._v_min, self._v_max)
        else:
            self.grid_view.set_signal(self.signal, self.leads, self.fs,
                                      self.time_pos, win,
                                      self._v_min, self._v_max)
        # Selection mode: enabled on every cell when not live
        for cell in self.grid_view.cells.values():
            cell.selection_mode = (not self._live)
        self.grid_view.rhythm.selection_mode = (not self._live)

    def _refresh_single_lead(self):
        """Backward-compat: refresh the focused single-lead cell only."""
        if self.signal is None or self._focus_lead not in self.leads:
            return
        idx = self.leads.index(self._focus_lead)
        window = self._window_1
        t_start = max(0.0, self.time_pos)
        t_end = min(self.duration, t_start + window)
        if t_end - t_start < window:
            t_start = max(0.0, t_end - window)
        cell = self.grid_view.cells.get(self._focus_lead)
        if cell is None:
            return
        cell.v_min = self._v_min
        cell.v_max = self._v_max
        cell.set_data(self._focus_lead, self.signal[:, idx], self.fs, t_start, t_end)
        lead_markings = [m for m in self._marking_store.get_all() if m.lead == self._focus_lead]
        cell.markings = [
            {"id": m.id, "t1": m.t1, "t2": m.t2, "type": m.type, "label": m.label}
            for m in lead_markings
        ]
        cell.update()
        self.single_lead = cell

    def _refresh_monitor(self):
        """Live tick: drive sweep cursor across every visible cell."""
        if self.signal is None:
            return
        sweep_frac = (self._monitor_t - self._monitor_page_start) / self._monitor_window \
            if self._monitor_window > 0 else 0
        sweep_frac = max(0.0, min(1.0, sweep_frac))
        # Cells participating in live: those visible per current layout
        visible = self.grid_view.visible_cell_leads()
        for lead_name in visible:
            cell = self.grid_view.cells.get(lead_name)
            if cell is None or lead_name not in self.leads:
                continue
            idx = self.leads.index(lead_name)
            cell.v_min = self._v_min
            cell.v_max = self._v_max
            cell.set_data(lead_name, self.signal[:, idx], self.fs,
                          self._monitor_page_start,
                          self._monitor_page_start + self._monitor_window)
            cell.set_sweep(sweep_frac)


    # ── Layout / live switching ──
    def _apply_layout(self):
        """Rebuild grid_view for current layout_id + refresh signal data."""
        self.grid_view.set_layout(self._layout_id, self._visible_leads, self._focus_lead)
        if self._focus_lead in self.grid_view.cells:
            self.single_lead = self.grid_view.cells[self._focus_lead]
        if self.signal is not None:
            self._refresh_views()
            self._refresh_markings()
        self._update_zoom_label()
        self._update_time_display()

    def _on_layout_changed(self, layout_id: str):
        self._layout_id = layout_id
        self._clear_selection_preview()
        self._apply_layout()

    def _on_live_toggled(self, live: bool):
        self._live = live
        self.pause_btn.setEnabled(live)
        if live:
            self._start_monitor()
        else:
            self._monitor_timer.stop()
            self._monitor_playing = False
            # Clear sweep state on every cell
            for cell in self.grid_view.cells.values():
                cell._sweep_pos = None
                cell._old_signal = None
            self.grid_view.rhythm._sweep_pos = None
            self.grid_view.rhythm._old_signal = None
            self._restore_scrubber_range()
            self._refresh_views()

    def _on_visible_leads_changed(self, leads: list):
        self._visible_leads = list(leads)
        if self._layout_id in {"grid_3x4", "grid_2x6", "stack_1xN"}:
            self._apply_layout()

    def _on_cell_clicked(self, lead: str):
        self._last_clicked_lead = lead
        self._focus_lead = lead
        self.markings_panel.set_current_lead(lead)
        self.layout_switcher.set_focus_lead(lead)

    def _on_cell_selection_completed(self, lead: str, t1: float, t2: float):
        self._on_cell_clicked(lead)
        self._on_selection_completed(t1, t2)

    def _on_cell_selection_live(self, lead: str, t1: float, t2: float):
        self._on_cell_clicked(lead)
        self._on_selection_live(t1, t2)

    # ── Lead selection ──
    def _on_lead_selected(self, lead: str):
        self._focus_lead = lead
        self._on_cell_clicked(lead)
        self._refresh_single_lead()
        self._refresh_markings()

    def _default_model_path(self) -> str:
        """First discovered checkpoint (model-sota prioritised by discover_models)."""
        models = discover_models()
        return models[0] if models else ""

    def _ensure_model_loaded(self):
        """Load model if not cached. Returns (model, device) or raises."""
        from model.inference_api import load_checkpoint_model
        path = self._default_model_path()
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono żadnego modelu w katalogach model/annotations lub models/")
        if self._model is not None and self._model_path == path:
            return self._model, self._model_device
        model, device = load_checkpoint_model(path, num_classes=len(TARGET_CLASSES))
        self._model = model
        self._model_device = device
        self._model_path = path
        return model, device

    # ── Selection / Marking flow ──────────────────────────────

    def _on_selection_completed(self, t1, t2):
        """User finished selecting a region on the canvas -- show context menu."""
        lead = self._last_clicked_lead or self._focus_lead
        self._pending_selection = (lead, t1, t2)

        # Show context menu at cursor position
        selection_seconds = t2 - t1
        self._context_menu = SelectionContextMenu(self)
        self._context_menu.action_selected.connect(self._on_context_action)
        global_pos = QCursor.pos()
        self._context_menu.show_at(global_pos, selection_seconds)

    def _on_context_action(self, action: str):
        """Handle an action chosen from the selection context menu."""
        if self._pending_selection is None:
            return
        lead, t1, t2 = self._pending_selection
        dt_ms = (t2 - t1) * 1000

        if action == "annotate":
            self._show_annotation_form(lead, t1, t2)
        elif action.startswith("mark_"):
            mark_type = action.replace("mark_", "")
            if mark_type == "custom":
                from PySide6.QtWidgets import QInputDialog
                label, ok = QInputDialog.getText(self, "Oznaczenie", "Etykieta:")
                if not ok or not label:
                    self._clear_selection_preview()
                    return
                marking = Marking(type="custom", lead=lead, t1=t1, t2=t2, label=label, value_ms=dt_ms)
            else:
                marking = Marking(type=mark_type, lead=lead, t1=t1, t2=t2, value_ms=dt_ms)
            self._marking_store.add(marking)
        elif action == "scan":
            self._run_window_scan(lead, t1, t2)
        elif action == "zoom":
            self._zoom_to_region(t1, t2)
        elif action == "export_png":
            self._export_region_png(t1, t2)

        self._clear_selection_preview()
        self._refresh_markings()
        self._save_ann()

    def _clear_selection_preview(self):
        """Clear selection preview on every cell."""
        for cell in list(self.grid_view.cells.values()) + [self.grid_view.rhythm]:
            cell.pending_marker = None
            cell.selection_preview = None
            cell._drag_active = False
            cell._drag_start_px = None
            cell.update()
        self._sel_indicator.setText("")
        self._sel_indicator.setStyleSheet(f"background: {T.WHITE}; border: none; color: transparent;")

    def _show_annotation_form(self, lead, t1, t2):
        """Open the annotation creation form in the markings panel."""
        self.markings_panel.show_annotation_form(lead, t1, t2)

    def _refresh_markings(self):
        """Sync marking store to canvas and panel."""
        lead = self._last_clicked_lead or self._focus_lead
        # Distribute all markings to grid cells (each cell shows its own lead's markings)
        all_markings = [
            {"id": m.id, "t1": m.t1, "t2": m.t2, "type": m.type, "label": m.label, "lead": m.lead}
            for m in self._marking_store.get_all()
        ]
        self.grid_view.set_markings(all_markings)

        # Panel: show all markings + set current lead for filtering
        self.markings_panel.set_current_lead(lead)
        self.markings_panel.set_markings(self._marking_store.get_all())
        self.markings_panel.set_undo_enabled(self._marking_store.can_undo)
        self.markings_panel.set_redo_enabled(self._marking_store.can_redo)

    def _on_marking_hovered(self, marking_id: str):
        for cell in list(self.grid_view.cells.values()) + [self.grid_view.rhythm]:
            cell.hovered_marking = marking_id
            cell.update()

    def _on_marking_unhovered(self):
        for cell in list(self.grid_view.cells.values()) + [self.grid_view.rhythm]:
            cell.hovered_marking = None
            cell.update()

    def _on_marking_selected(self, marking_id: str):
        for cell in list(self.grid_view.cells.values()) + [self.grid_view.rhythm]:
            cell.selected_marking = marking_id
            cell.update()
        # Compute lead importance on demand for scan markings without cached XAI
        m = self._marking_store.get_by_id(marking_id)
        if m and m.type == "scan":
            if not m.lead_importance and self.signal is not None:
                self._compute_lead_importance_for_marking(m)
            else:
                self._apply_explain_heatmap(m)
        else:
            self._apply_explain_heatmap(None)

    def _apply_explain_heatmap(self, marking):
        """Set/clear attention overlay on all canvases."""
        ov = None
        if marking is not None and marking.time_heatmap:
            ov = (marking.t1, marking.t2, marking.time_heatmap)
        for cell in list(self.grid_view.cells.values()) + [self.grid_view.rhythm]:
            cell.attention_overlay = ov
            cell.update()

    def _compute_lead_importance_for_marking(self, marking):
        """Run explain_prediction on a scan marking's window and update panel."""
        try:
            model, device = self._ensure_model_loaded()
        except Exception:
            return
        from model.inference_api import explain_prediction
        s = max(0, int(marking.t1 * self.fs))
        e = min(self.signal.shape[0], int(marking.t2 * self.fs))
        if e - s < 100:
            return
        window = self.signal[s:e]
        # Pick non-healthy top class to explain
        probs = marking.probs or {}
        non_healthy = [(c, p) for c, p in probs.items() if c != "class_healthy"]
        if not non_healthy:
            return
        target_cls = max(non_healthy, key=lambda x: x[1])[0]
        try:
            exp = explain_prediction(
                model=model, data=window,
                target_classes=[target_cls],
                class_names=TARGET_CLASSES, device=device,
            )
            if exp.get("explanations"):
                imp = exp["explanations"][0].get("lead_importance_percent")
                heat = exp["explanations"][0].get("time_heatmap")
                if imp:
                    self._marking_store.edit(
                        marking.id, lead_importance=imp, time_heatmap=heat,
                    )
                    title = f"{marking.t1:.1f}\u2013{marking.t2:.1f} s \u00b7 {marking.label}"
                    self.markings_panel._lead_importance_panel.set_data(imp, title=title)
                    self._apply_explain_heatmap(marking)
                    self._save_ann()
        except Exception:
            pass

    def _on_marking_deleted(self, marking_id: str):
        self._marking_store.delete(marking_id)
        self._refresh_markings()
        self._save_ann()

    def _on_marking_focus(self, marking_id: str):
        """Scroll canvas to show the marking, switching to focus_1L if needed."""
        m = self._marking_store.get_by_id(marking_id)
        if not m:
            return
        # Switch to focus on this lead
        self._focus_lead = m.lead
        self._last_clicked_lead = m.lead
        self._layout_id = "focus_1L"
        self.layout_switcher.set_layout("focus_1L")
        self.layout_switcher.set_focus_lead(m.lead)
        # Center the view on the marking
        mid = (m.t1 + m.t2) / 2
        self.time_pos = max(0.0, mid - self._window_1 / 2)
        self.time_pos = min(self.time_pos, max(0.0, self.duration - self._window_1))
        self._restore_scrubber_range()
        self._apply_layout()
        # Highlight it
        cell = self.grid_view.cells.get(m.lead)
        if cell is not None:
            cell.selected_marking = marking_id
            cell.update()

    def _on_marking_edited(self, marking_id: str, field: str, value: str):
        self._marking_store.edit(marking_id, **{field: value})
        # Regenerate label for annotations when category changes
        m = self._marking_store.get_by_id(marking_id)
        if m and m.type == "annotation" and field == "category":
            self._marking_store.edit(marking_id, label=value)
        self._refresh_markings()
        self._save_ann()

    def _on_annotation_created(self, lead: str, category: str, note: str, t1: float, t2: float):
        marking = Marking(type="annotation", lead=lead, t1=t1, t2=t2,
                          category=category, note=note, source="user")
        self._marking_store.add(marking)
        self._refresh_markings()
        self._save_ann()
        self.markings_panel.set_selected(marking.id)

    def _undo(self):
        if self._marking_store.undo():
            self._refresh_markings()
            self._save_ann()

    def _redo(self):
        if self._marking_store.redo():
            self._refresh_markings()
            self._save_ann()

    def _on_cell_double_click(self, lead: str, time: float):
        """Double-click on cell — jump to focus single-lead view of that lead."""
        self._focus_lead = lead
        self._last_clicked_lead = lead
        self._layout_id = "focus_1L"
        self.layout_switcher.set_layout("focus_1L")
        self.layout_switcher.set_focus_lead(lead)
        self.time_pos = max(0, time - self._window_1 / 2)
        self._restore_scrubber_range()
        self._apply_layout()

    def _on_canvas_right_click(self, gx, gy):
        """Show a right-click context menu on the canvas (e.g. reset zoom)."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background: {T.WHITE}; color: {T.TEXT};
                border: 1px solid {T.BORDER}; border-radius: 6px;
                padding: 4px; font-size: 13px;
            }}
            QMenu::item {{
                padding: 6px 16px; border-radius: 4px;
            }}
            QMenu::item:selected {{
                background: {T.BG_SECONDARY};
            }}
        """)
        reset_action = menu.addAction("Resetuj powiększenie")
        reset_action.triggered.connect(self._reset_zoom)
        menu.exec(QPoint(int(gx), int(gy)))

    _ZOOM_STEPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    def _is_focus(self) -> bool:
        return self._layout_id == "focus_1L"

    def _current_window(self) -> float:
        if self._live:
            return self._monitor_window
        return self._window_1 if self._is_focus() else self._window_12

    def _set_current_window(self, value: float):
        if self._live:
            self._monitor_window = value
        elif self._is_focus():
            self._window_1 = value
        else:
            self._window_12 = value

    def _zoom_in(self):
        """Decrease the time window — snap to nearest smaller step."""
        current = self._current_window()
        target = self._ZOOM_STEPS[0]
        for step in self._ZOOM_STEPS:
            if step < current - 0.01:
                target = step
            else:
                break
        self._set_current_window(max(target, self._ZOOM_STEPS[0]))
        self._apply_zoom()

    def _zoom_out(self):
        """Increase the time window — snap to nearest larger step."""
        current = self._current_window()
        max_window = min(self.duration, self._ZOOM_STEPS[-1])
        target = max_window
        for step in self._ZOOM_STEPS:
            if step > current + 0.01:
                target = min(step, max_window)
                break
        self._set_current_window(target)
        self._apply_zoom()

    def _reset_zoom(self):
        """Reset current view to default window size."""
        if self._live:
            self._monitor_window = min(3.0, self.duration)
        elif self._is_focus():
            self._window_1 = min(3.0, self.duration)
        else:
            self._window_12 = min(2.5, self.duration)
        self._apply_zoom()

    def _on_pinch_zoom(self, direction: int):
        """Handle pinch zoom from any cell. +1=in, -1=out."""
        if self._live:
            return
        if direction > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _on_scroll_pan(self, dt: float):
        """Handle two-finger swipe pan from any cell."""
        if self._live:
            return
        new_pos = self.time_pos + dt
        new_pos = max(0.0, min(self._scrubber_max, new_pos))
        self.time_pos = new_pos
        self.scrubber.setValue(int(new_pos * 100))

    def _on_grid_scroll_pan(self, dt: float):
        """Backward-compat alias."""
        self._on_scroll_pan(dt)

    def _restore_scrubber_range(self):
        """Recompute scrubber range from current signal duration and zoom."""
        if self.signal is None:
            return
        max_window = max(self._window_12, self._window_1)
        self._scrubber_max = max(0.0, self.duration - max_window)
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, int(self._scrubber_max * 100))
        self.scrubber.setValue(int(self.time_pos * 100))
        self.scrubber.blockSignals(False)

    def _apply_zoom(self):
        """Common zoom update: label, scrubber, view, time display."""
        win = self._current_window()
        if self.time_pos + win > self.duration:
            self.time_pos = max(0, self.duration - win)
        self._update_zoom_label()
        self._restore_scrubber_range()
        if self.signal is not None and not self._live:
            self.grid_view.set_signal(self.signal, self.leads, self.fs,
                                      self.time_pos, win,
                                      self._v_min, self._v_max)
        self._update_time_display()

    def _update_zoom_label(self):
        if hasattr(self, '_zoom_label'):
            self._zoom_label.setText(f"{self._current_window():.1f} s")

    def _on_selection_live(self, t1, t2):
        """Update the selection indicator in the status bar during live selection."""
        dt = t2 - t1
        dt_ms = dt * 1000

        # Build indicator text
        time_text = f"{dt:.2f} s ({dt_ms:.0f} ms)"

        if dt >= 10.0 - 0.01:  # small epsilon for floating-point pixel conversion
            # AI scan available
            indicator = f"<b>{time_text}</b>  \u2714 Skan AI dostępny"
            self._sel_indicator.setStyleSheet(f"""
                font-size: 11px; color: {T.GREEN}; font-family: Menlo;
                background: {T.GREEN_BG}; border: 1px solid {T.GREEN_BORDER};
                border-radius: 4px; padding: 2px 10px;
            """)
        else:
            # AI scan NOT available — show how much more is needed
            remaining = 10.0 - dt
            indicator = f"<b>{time_text}</b>  \u2718 Brakuje {remaining:.1f} s do skanu AI"
            self._sel_indicator.setStyleSheet(f"""
                font-size: 11px; color: {T.AMBER_TEXT}; font-family: Menlo;
                background: {T.AMBER_BG}; border: 1px solid {T.AMBER_BORDER};
                border-radius: 4px; padding: 2px 10px;
            """)

        self._sel_indicator.setText(indicator)

    def _zoom_to_region(self, t1, t2):
        """Zoom the 1-lead view to a specific time region."""
        pad = (t2 - t1) * 0.1
        self.time_pos = max(0, t1 - pad)
        self._window_1 = (t2 - t1) + 2 * pad
        self._apply_zoom()

    def _export_region_png(self, t1, t2):
        """Export the current canvas view as a PNG file."""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Eksportuj fragment", "fragment.png", "PNG (*.png)")
        if path:
            pixmap = self.single_lead.grab()
            pixmap.save(path, "PNG")

    # ── Autoscan ────────────────────────────────────────────────

    def _autoscan_cache_path(self) -> str:
        file_key = self._base_path or self.filename or ""
        model_key = self._model_path or self._default_model_path() or ""
        key = f"{file_key}:{model_key}"
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(".cache", "autoscan", f"{h}.json")

    def _load_autoscan_cache(self) -> list | None:
        path = self._autoscan_cache_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("windows")
        except Exception:
            return None

    def _save_autoscan_cache(self, results: list):
        path = self._autoscan_cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"windows": results}, f)

    # ── .ann file I/O ──────────────────────────────────────

    def _ann_path(self) -> str:
        if self._base_path:
            return self._base_path + ".ann"
        return ""

    def _get_patient_dict(self) -> dict:
        """Read patient fields from InfoPanel into a dict."""
        pf = self.info_panel._patient_fields
        return {
            "name": pf["name"].text() if "name" in pf else "",
            "id": pf["id"].text() if "id" in pf else "",
            "age": pf["age"].text() if "age" in pf else "",
            "sex": pf["sex"].text() if "sex" in pf else "",
            "date": pf["date"].text() if "date" in pf else "",
        }

    def _load_ann(self):
        """Load markings and patient data from .ann file."""
        path = self._ann_path()
        if not path or not os.path.exists(path):
            return
        success, patient = self._marking_store.load_ann(path)
        if success:
            self._refresh_markings()
            if patient:
                self.info_panel.set_patient(
                    patient_id=patient.get("id", ""),
                    age=patient.get("age", ""),
                    sex=patient.get("sex", ""),
                    date=patient.get("date", ""),
                    name=patient.get("name", ""),
                )

    def _save_ann(self):
        """Save current markings and patient data to .ann file."""
        path = self._ann_path()
        if not path:
            return
        self._marking_store.save_ann(path, patient=self._get_patient_dict())

    def _run_full_analysis(self):
        """Run full analysis: sliding window scan across the entire signal."""
        if self.signal is None:
            return

        # Try cache first (keyed by full base_path + selected model)
        self._autoscan_file_path = self.filename
        cached = self._load_autoscan_cache()
        if cached:
            self._autoscan_results = cached
            self._log_autoscan_results(cached)
            self._apply_autoscan_results()
            self._apply_autoscan_overlay()
        else:
            self._run_autoscan()

    def _run_autoscan(self):
        """Slide a 10s window across the signal and classify each segment."""
        try:
            model, device = self._ensure_model_loaded()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Błąd", f"Nie udało się załadować modelu:\n{e}")
            self._autoscan_active = False
            return

        from model.inference_api import predict_with_model

        window_sec = 10.0
        step_sec = 5.0
        window_samples = int(window_sec * self.fs)
        step_samples = int(step_sec * self.fs)
        total = self.signal.shape[0]

        starts = list(range(0, total - window_samples + 1, step_samples))
        last_start = total - window_samples
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

        n_windows = len(starts)

        # Show loading overlay
        self._autoscan_overlay = AutoscanOverlay(self)
        self._autoscan_overlay.show_loading(f"0/{n_windows} okien")

        # Progress polling timer
        self._autoscan_poll = QTimer(self)
        self._autoscan_poll.setInterval(100)
        self._autoscan_thread_results = None
        self._autoscan_thread_progress = [0, n_windows]

        def _worker():
            """Run inference in background thread."""
            from model.inference_api import explain_prediction
            results = []
            for i, s in enumerate(starts):
                self._autoscan_thread_progress[0] = i + 1
                window = self.signal[s:s + window_samples]
                t_start = s / self.fs
                t_end = (s + window_samples) / self.fs

                try:
                    res = predict_with_model(
                        model=model, data=window, threshold=0.5,
                        class_names=TARGET_CLASSES, device=device,
                    )
                    probs = res["probabilities"][0]
                    prob_dict = {cls: float(probs[j]) for j, cls in enumerate(TARGET_CLASSES)}
                except Exception:
                    prob_dict = {cls: 0.0 for cls in TARGET_CLASSES}

                # Thresholds (per user 2026-04-19):
                #   top illness >= 0.7              → red (illness label)
                #   top illness in [0.4, 0.7)       → yellow (illness label)
                #   true top is healthy and < 0.5   → yellow (Niepewne label)
                #   else                            → drop
                ILLNESS_HIGH = 0.7
                ILLNESS_FLOOR = 0.4
                HEALTHY_FLOOR = 0.5

                p_hlt = prob_dict.get("class_healthy", 0.0)
                non_healthy = [(c, p) for c, p in prob_dict.items() if c != "class_healthy"]
                top_ill_cls, p_ill = max(non_healthy, key=lambda x: x[1]) if non_healthy else ("", 0.0)
                true_top_cls = max(prob_dict, key=prob_dict.get) if prob_dict else ""

                if p_ill >= ILLNESS_HIGH:
                    color = 2
                elif p_ill >= ILLNESS_FLOOR:
                    color = 1
                elif true_top_cls == "class_healthy" and p_hlt < HEALTHY_FLOOR:
                    color = 1  # model thinks healthy but uncertain → flag as Niepewne
                else:
                    color = 0

                # XAI: lead importance + time heatmap. Only run for actionable windows.
                lead_importance = None
                time_heatmap = None
                non_healthy_top = max(
                    ((c, p) for c, p in prob_dict.items() if c != "class_healthy"),
                    key=lambda x: x[1], default=(None, 0.0),
                )
                if color != 0 and non_healthy_top[0]:
                    try:
                        exp = explain_prediction(
                            model=model, data=window,
                            target_classes=[non_healthy_top[0]],
                            class_names=TARGET_CLASSES, device=device,
                        )
                        if exp.get("explanations"):
                            lead_importance = exp["explanations"][0].get("lead_importance_percent")
                            time_heatmap = exp["explanations"][0].get("time_heatmap")
                    except Exception:
                        lead_importance = None
                        time_heatmap = None

                results.append({
                    "t_start": t_start, "t_end": t_end,
                    "color": color, "probs": prob_dict,
                    "lead_importance": lead_importance,
                    "time_heatmap": time_heatmap,
                })
            self._autoscan_thread_results = results

        def _poll():
            """Check thread progress from main thread."""
            done, total_w = self._autoscan_thread_progress
            if hasattr(self, '_autoscan_overlay') and self._autoscan_overlay:
                self._autoscan_overlay.update_progress(f"{done}/{total_w} okien")
            if self._autoscan_thread_results is not None:
                self._autoscan_poll.stop()
                self._autoscan_results = self._autoscan_thread_results
                self._save_autoscan_cache(self._autoscan_thread_results)
                self._log_autoscan_results(self._autoscan_thread_results)
                self._autoscan_thread_results = None
                # Show done animation
                if hasattr(self, '_autoscan_overlay') and self._autoscan_overlay:
                    self._autoscan_overlay.show_done()
                    QTimer.singleShot(1300, self._autoscan_finish)
                else:
                    self._autoscan_finish()

        self._autoscan_poll.timeout.connect(_poll)
        self._autoscan_poll.start()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _log_autoscan_results(self, results: list):
        """Append full per-window probability dump to logs/autoscan.log."""
        try:
            from datetime import datetime
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", "autoscan.log")
            with open(path, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"----- [{ts}] -----\n")
                f.write(f"file: {self.filename}  duration: {self.duration:.1f}s  fs: {self.fs}Hz\n")
                f.write(f"model: {self._model_path or '(unknown)'}\n")
                f.write(f"windows: {len(results)}\n\n")
                # Header listing class order for clarity
                cls_names = list(TARGET_CLASSES)
                f.write("window".ljust(18))
                for c in cls_names:
                    short = c.replace("class_", "")
                    f.write(f"  {short[:22]:>22}")
                f.write("\n")
                for r in results:
                    rng = f"{r['t_start']:6.1f}-{r['t_end']:6.1f}s"
                    f.write(rng.ljust(18))
                    probs = r.get("probs") or {}
                    for c in cls_names:
                        v = probs.get(c, 0.0) * 100
                        f.write(f"  {v:21.2f}%")
                    f.write("\n")
                f.write("\n")
        except Exception:
            pass

    def _autoscan_finish(self):
        """Apply results after overlay closes."""
        if hasattr(self, '_autoscan_overlay') and self._autoscan_overlay:
            self._autoscan_overlay.close()
            self._autoscan_overlay = None
        self._apply_autoscan_results()
        self._apply_autoscan_overlay()

        # Build _last_results for the report page from the first window
        if self._autoscan_results:
            first = self._autoscan_results[0]
            probs = first.get("probs", {})
            model_name = os.path.basename(self._model_path) if self._model_path else ""
            self._last_results = {
                "probabilities": probs,
                "model_name": model_name,
                "elapsed": 0.0,
            }
            self.analysis_badge.show()



    def _rebuild_autoscan_from_markings(self):
        """If the marking store contains scan markings (from a .ann sidecar),
        reconstruct _autoscan_results from them so the colored overlay and
        analysis badge render on file load without re-running the model.
        """
        scan_markings = [m for m in self._marking_store.get_all() if m.type == "scan"]
        if not scan_markings:
            return
        synthetic = []
        for m in scan_markings:
            synthetic.append({
                "t_start": m.t1,
                "t_end": m.t2,
                "color": int(m.color_code or 0),
                "probs": m.probs or {},
                "lead_importance": m.lead_importance,
                "time_heatmap": m.time_heatmap,
            })
        # Sort by start time for predictable merge output.
        synthetic.sort(key=lambda r: r["t_start"])
        self._autoscan_results = synthetic
        self._apply_autoscan_overlay()
        self.analysis_badge.show()

    def _merged_autoscan_regions(self) -> list[dict]:
        """Split overlapping scan windows into atomic segments, keep the
        highest-priority prediction (red > yellow > none) per segment, then
        merge neighbors that share color and top class. Healthy segments
        are dropped.
        """
        raw = self._autoscan_results or []
        if not raw:
            return []

        # Atomic boundaries from every window edge
        bounds = sorted({r["t_start"] for r in raw} | {r["t_end"] for r in raw})
        atoms: list[dict] = []
        prob_tol = 0.10  # merge when top probability differs by <= 10pp

        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            if b - a < 1e-6:
                continue
            mid = 0.5 * (a + b)
            covering = [r for r in raw if r["t_start"] <= mid < r["t_end"]]
            if not covering:
                continue
            max_color = max(r.get("color", 0) for r in covering)
            if max_color == 0:
                continue
            candidates = [r for r in covering if r.get("color", 0) == max_color]

            def _non_healthy_top(r):
                probs = r.get("probs") or {}
                items = [(c, p) for c, p in probs.items() if c != "class_healthy"]
                if not items:
                    return ("", 0.0)
                return max(items, key=lambda x: x[1])

            rep = max(candidates, key=lambda r: _non_healthy_top(r)[1])
            top_cls, top_prob = _non_healthy_top(rep)
            atoms.append({
                "t_start": a,
                "t_end": b,
                "color": max_color,
                "top_cls": top_cls,
                "top_prob": top_prob,
                "probs": rep.get("probs") or {},
                "lead_importance": rep.get("lead_importance"),
                "time_heatmap": rep.get("time_heatmap"),
            })

        # Drop "Niepewne" atoms whose top class matches a touching strong atom
        # (avoids Y-R-R-Y noise where Y predicts same illness weakly).
        cleaned: list[dict] = []
        for i, seg in enumerate(atoms):
            if seg["color"] == 1 and seg["top_cls"]:
                neighbours = []
                if i > 0:
                    neighbours.append(atoms[i - 1])
                if i + 1 < len(atoms):
                    neighbours.append(atoms[i + 1])
                if any(
                    n["color"] == 2 and n["top_cls"] == seg["top_cls"]
                    and abs(n["t_end"] - seg["t_start"]) < 1e-6
                    or n["color"] == 2 and n["top_cls"] == seg["top_cls"]
                    and abs(seg["t_end"] - n["t_start"]) < 1e-6
                    for n in neighbours
                ):
                    continue
            cleaned.append(seg)

        merged: list[dict] = []
        for seg in cleaned:
            if merged:
                prev = merged[-1]
                can_merge = (
                    abs(prev["t_end"] - seg["t_start"]) < 1e-6
                    and prev["color"] == seg["color"]
                    and prev["top_cls"] == seg["top_cls"]
                    and abs(prev["top_prob"] - seg["top_prob"]) <= prob_tol
                )
                if can_merge:
                    prev["t_end"] = seg["t_end"]
                    if seg["top_prob"] > prev["top_prob"]:
                        prev["top_prob"] = seg["top_prob"]
                        prev["probs"] = seg["probs"]
                        prev["lead_importance"] = seg.get("lead_importance")
                        prev["time_heatmap"] = seg.get("time_heatmap")
                    continue
            merged.append(dict(seg))
        return merged

    def _apply_autoscan_results(self):
        """Create Marking objects from merged autoscan regions and add to the store."""
        if not self._autoscan_results:
            return
        # Remove old scan markings before adding new ones
        old_scans = [m for m in self._marking_store.get_all() if m.type == "scan"]
        for m in old_scans:
            self._marking_store.delete(m.id)

        for seg in self._merged_autoscan_regions():
            marking = Marking(
                type="scan",
                lead="all",
                t1=seg["t_start"],
                t2=seg["t_end"],
                probs=seg.get("probs"),
                color_code=seg.get("color", 0),
                source="ai",
                lead_importance=seg.get("lead_importance"),
                time_heatmap=seg.get("time_heatmap"),
            )
            self._marking_store.add(marking)
        # Scan bulk operations are not undoable
        self._marking_store.clear_history()
        self._refresh_markings()
        self._save_ann()

    def _apply_autoscan_overlay(self):
        if not self._autoscan_results:
            return
        from ui.theme import CLASS_NAMES_PL

        # Build ground truth annotation lines
        gt_lines = self._build_gt_lines(CLASS_NAMES_PL)

        regions = []
        for seg in self._merged_autoscan_regions():
            code = seg["color"]
            top_cls = seg.get("top_cls") or ""
            top_prob = seg.get("top_prob") or 0.0
            if top_prob * 100 < 40:
                label = "Niepewne"
            else:
                name = CLASS_NAMES_PL.get(top_cls, top_cls)
                if len(name) > 22:
                    name = name[:20] + "."
                label = f"{name} {top_prob * 100:.0f}%"
            if self._is_gt_mismatch(seg["t_start"], seg["t_end"]):
                label = "\u26a0 " + label
            regions.append((seg["t_start"], seg["t_end"], code, [label]))

        self.grid_12.set_autoscan_regions(regions)
        self.grid_12.set_gt_annotations(gt_lines)
        self.single_lead.autoscan_regions = regions
        self.single_lead.show_autoscan_labels = True
        self.single_lead.gt_annotations = gt_lines
        self.single_lead.update()

    def _build_gt_lines(self, class_names_pl: dict) -> list:
        """Build ground truth annotation bracket data from _ground_truth."""
        gt = self._ground_truth
        if gt is None:
            return []
        gt_lines = []
        if isinstance(gt, list):
            # Windowed GT from .annotations.json
            for win in gt:
                truth = win.get("ground_truth", {})
                # Find top non-healthy class with value 1.0
                top_cls = None
                for cls, val in truth.items():
                    if cls != "class_healthy" and val >= 1.0:
                        top_cls = cls
                        break
                if top_cls:
                    label = class_names_pl.get(top_cls, top_cls)
                    gt_lines.append((win["start"], win["end"], label))
        elif isinstance(gt, dict):
            # Whole-file GT (PTB-XL)
            top_cls = None
            for cls, val in gt.items():
                if cls != "class_healthy" and val >= 1.0:
                    top_cls = cls
                    break
            if top_cls:
                label = class_names_pl.get(top_cls, top_cls)
                gt_lines.append((0.0, self.duration, label))
        return gt_lines

    def _is_gt_mismatch(self, t_start: float, t_end: float) -> bool:
        """Check if ground truth disagrees with a non-healthy model prediction.

        Returns True if GT says healthy or no GT exists for this time range.
        """
        gt = self._ground_truth
        if gt is None:
            return True  # No GT at all → mismatch
        if isinstance(gt, dict):
            # Whole-file: check if healthy
            return gt.get("class_healthy", 0.0) >= 1.0
        if isinstance(gt, list):
            # Find best-overlapping window
            best_overlap = 0.0
            best_gt = None
            for win in gt:
                ws, we = win["start"], win["end"]
                overlap = max(0.0, min(t_end, we) - max(t_start, ws))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gt = win.get("ground_truth", {})
            if best_gt is None:
                return True  # No overlapping GT window
            return best_gt.get("class_healthy", 0.0) >= 1.0
        return True

    def _clear_autoscan_overlay(self):
        self.grid_12.clear_autoscan_regions()
        self.single_lead.autoscan_regions = []
        self.single_lead.gt_annotations = []
        self.single_lead.update()

    def _run_window_scan(self, lead: str, t1: float, t2: float):
        """Run AI scan on the selected region and store result as a scan marking."""
        if self.signal is None:
            return
        window_samples = int((t2 - t1) * self.fs)
        if window_samples < int(10.0 * self.fs):
            return  # Need at least 10s

        try:
            model, device = self._ensure_model_loaded()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Błąd", f"Nie udało się załadować modelu:\n{e}")
            return

        from model.inference_api import predict_with_model

        start_sample = int(t1 * self.fs)
        end_sample = int(t2 * self.fs)
        window_signal = self.signal[start_sample:end_sample]

        try:
            res = predict_with_model(
                model=model, data=window_signal, threshold=0.5,
                class_names=TARGET_CLASSES, device=device,
            )
            probs = res["probabilities"][0]
            prob_dict = {cls: float(probs[j]) for j, cls in enumerate(TARGET_CLASSES)}
        except Exception:
            prob_dict = {cls: 0.0 for cls in TARGET_CLASSES}

        top_cls = max(prob_dict, key=prob_dict.get)
        top_prob = prob_dict[top_cls]
        color_code = 0
        if top_cls == "class_healthy" and top_prob >= 0.5:
            color_code = 0
        elif top_cls != "class_healthy" and top_prob >= 0.5:
            color_code = 2
        else:
            color_code = 1

        marking = Marking(
            type="scan", lead=lead, t1=t1, t2=t2,
            probs=prob_dict, color_code=color_code, source="ai",
        )
        self._marking_store.add(marking)
        self._refresh_markings()
        self._save_ann()

    def _resolve_ground_truth(self, t_start: float, t_end: float) -> dict | None:
        """Pick the correct ground truth for a given analysis window.

        Handles both:
        - dict: whole-file GT (PTB-XL) — returned as-is
        - list: windowed GT from .annotations.json — picks the best-overlapping window
        """
        gt = self._ground_truth
        if gt is None:
            return None
        if isinstance(gt, dict):
            return gt
        if isinstance(gt, list):
            # Find the window with the most overlap
            best = None
            best_overlap = 0.0
            for w in gt:
                ws, we = w["start"], w["end"]
                overlap = max(0.0, min(t_end, we) - max(t_start, ws))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = w.get("ground_truth")
            return best
        return None

    def _get_analysis_window(self):
        """Extract a 10-second signal window for analysis.

        Returns (window_signal, t_start, t_end) where window_signal has shape (N, 12).
        """
        target_samples = int(10.0 * self.fs)
        total_samples = self.signal.shape[0]

        start_sample = int(self.time_pos * self.fs)

        start_sample = max(0, min(start_sample, total_samples - target_samples))
        end_sample = min(start_sample + target_samples, total_samples)

        window = self.signal[start_sample:end_sample]
        t_start = start_sample / self.fs
        t_end = end_sample / self.fs
        return window, t_start, t_end

    def _on_analyze(self):
        """Run single-window analysis (used by Ctrl+Return shortcut)."""
        if self.signal is None:
            return
        # Use _run_full_analysis for the new flow
        self._run_full_analysis()

    def _scrubber_style(self) -> str:
        return f"""
            QSlider::groove:horizontal {{
                height: 6px; background: {T.BORDER}; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 14px; height: 14px; margin: -4px 0;
                background: {T.ACCENT}; border: 2px solid {T.WHITE};
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {T.ACCENT}; border-radius: 3px;
            }}
        """

    def _zoom_reset_btn_style(self) -> str:
        return f"""
            QPushButton {{
                font-size: 10px; padding: 0 8px; height: 28px;
                border: 1px solid {T.BORDER}; border-radius: 6px;
                background: {T.WHITE}; color: {T.TEXT_MUTED};
            }}
            QPushButton:hover {{ background: {T.BG_SECONDARY}; color: {T.TEXT}; }}
        """

    def _apply_pause_btn_style(self):
        from ui.theme import is_dark_mode
        hover = '#047857' if is_dark_mode() else '#3a8eef'
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                height: 32px; padding: 0 14px; border: none;
                border-radius: 16px; background: {T.ACCENT};
                color: {T.ACCENT_TEXT}; font-size: 14px; font-weight: 700;
            }}
            QPushButton:hover {{ background: {hover}; }}
        """)

    # ── Hold-to-scroll ─────────────────────────────────────
    def _start_hold(self, handler):
        self._hold_handler = handler
        handler()
        self._hold_timer.start()

    def _stop_hold(self):
        self._hold_timer.stop()
        self._hold_handler = None

    def _on_hold_tick(self):
        if self._hold_handler:
            self._hold_handler()

    # ── Navbar pause button ──────────────────────────────
    def _on_navbar_pause(self):
        if not self._live:
            return
        self._monitor_playing = not self._monitor_playing
        if self._monitor_playing:
            self._monitor_timer.start()
            self.pause_btn.setText("\u275a\u275a")
        else:
            self._monitor_timer.stop()
            self.pause_btn.setText("\u25b6")

    def _nav_start(self):
        self.time_pos = 0
        self.scrubber.setValue(0)

    def _nav_end(self):
        self.time_pos = self._scrubber_max
        self.scrubber.setValue(self.scrubber.maximum())

    def wheelEvent(self, event):
        """Handle trackpad two-finger swipe (horizontal scroll) to navigate the signal."""
        dx = event.pixelDelta().x() if not event.pixelDelta().isNull() else event.angleDelta().x()
        if dx != 0 and self.signal is not None:
            # Convert pixel delta to time: ~200px swipe = 1 second
            dt = -dx / 200.0
            self._nav_step(dt)
            event.accept()
        else:
            super().wheelEvent(event)

    def _nav_step(self, dt: float):
        self.time_pos = max(0, min(self._scrubber_max, self.time_pos + dt))
        self.scrubber.setValue(int(self.time_pos * 100))

    def _on_scrubber(self, value: int):
        self.time_pos = value / 100.0
        self._update_time_display()
        if self.signal is None:
            return
        if self._live:
            # Seek live sweep to scrubber position
            self._monitor_t = self.time_pos
            self._monitor_page_start = (self._monitor_t // self._monitor_window) * self._monitor_window
            for cell in list(self.grid_view.cells.values()) + [self.grid_view.rhythm]:
                cell._old_signal = None
                cell._sweep_pos = None
            self._refresh_monitor()
            return
        # Static: redraw current window
        win = self._current_window()
        self.grid_view.set_signal(self.signal, self.leads, self.fs,
                                  self.time_pos, win,
                                  self._v_min, self._v_max)
        # Update AI indicator if a cell has a pending selection marker
        focus = self.grid_view.cells.get(self._focus_lead)
        if focus is not None and focus.pending_marker is not None and focus._hover_x is not None:
            t = focus._px_to_time(focus._hover_x)
            if t is not None:
                t1 = min(focus.pending_marker, t)
                t2 = max(focus.pending_marker, t)
                self._on_selection_live(t1, t2)

    def _update_time_display(self):
        window = self._current_window()
        t_end = min(self.duration, self.time_pos + window)
        self.time_label.setText(f"{self.time_pos:.2f} – {t_end:.2f} s / {self.duration:.2f} s")
        self.timeline_overview.set_window(self.time_pos, window)
        layout_names = {
            "grid_3x4": "Siatka 3×4",
            "grid_2x6": "Siatka 2×6",
            "stack_1xN": "Wybrane jako paski",
            "focus_1L": "Pojedyncze odprowadzenie",
        }
        mode_name = layout_names.get(self._layout_id, self._layout_id)
        if self._live:
            mode_name = f"Podgląd ciągły · {mode_name}"
        self.status_pos.setText(f"{mode_name} · pozycja: {self.time_pos:.2f} s")

    def _on_overview_seek(self, t: float):
        if self.signal is None:
            return
        # Clamp and update scrubber (triggers refresh via _on_scrubber)
        t = max(0.0, min(t, self.scrubber.maximum() / 100.0))
        self.scrubber.setValue(int(t * 100))

    # ── Live preview ("podgląd ciągły") ──
    def _start_monitor(self):
        if self.signal is None:
            return
        self._monitor_t = self.time_pos
        if self._monitor_window <= 0:
            self._monitor_window = 3.0
        self._monitor_page_start = (self._monitor_t // self._monitor_window) * self._monitor_window
        self._monitor_playing = True
        self._monitor_timer.setInterval(int(50 / max(0.1, self._monitor_speed)))
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, int(self.duration * 100))
        self.scrubber.setValue(int(self._monitor_t * 100))
        self.scrubber.blockSignals(False)
        self.pause_btn.setText("\u275a\u275a")
        for cell in list(self.grid_view.cells.values()) + [self.grid_view.rhythm]:
            cell._old_signal = None
            cell._sweep_pos = None
            cell.selection_mode = False
        self._monitor_timer.start()
        self._refresh_monitor()

    def _monitor_tick(self):
        if not self._monitor_playing or self.signal is None:
            return
        self._monitor_t += 0.05
        page_end = self._monitor_page_start + self._monitor_window

        if self._monitor_t >= page_end:
            visible = self.grid_view.visible_cell_leads()
            for lead_name in visible:
                cell = self.grid_view.cells.get(lead_name)
                if cell is None or lead_name not in self.leads:
                    continue
                idx = self.leads.index(lead_name)
                cell._old_signal = self.signal[:, idx]
                cell._old_t_start = self._monitor_page_start
                cell._old_t_end = page_end
            self._monitor_page_start = page_end
            if self._monitor_page_start >= self.duration:
                self._monitor_page_start = 0.0
                self._monitor_t = 0.0

        self._refresh_monitor()
        self.time_pos = self._monitor_t
        self._update_time_display()
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(int(self._monitor_t * 100))
        self.scrubber.blockSignals(False)

    def _on_monitor_pause(self, paused: bool):
        self._monitor_playing = not paused
        if self._monitor_playing:
            self._monitor_timer.start()
            self.pause_btn.setText("\u275a\u275a")
        else:
            self._monitor_timer.stop()
            self.pause_btn.setText("\u25b6")

    def _on_monitor_speed(self, speed: float):
        self._monitor_speed = speed
        self._monitor_timer.setInterval(int(50 / max(0.1, speed)))

