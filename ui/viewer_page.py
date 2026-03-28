"""Main EKG viewer page with toolbar, view modes, panels, navigation bar."""
import glob
import hashlib
import json
import os
import time

import threading
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QStackedWidget, QSlider,
                                QSizePolicy, QComboBox, QApplication, QDialog,
                                QGraphicsOpacityEffect)

import ui.theme as T
from ui.theme import STANDARD_LEADS, TARGET_CLASSES
from ui.widgets import make_logo, make_separator
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
        if self.parent():
            pr = self.parent().rect()
            x = pr.center().x() - self.width() // 2
            y = pr.center().y() - self.height() // 2
            self.move(self.parent().mapToGlobal(pr.topLeft()).x() + x - self.parent().mapToGlobal(pr.topLeft()).x() + x,
                      self.parent().mapToGlobal(pr.topLeft()).y() + y - self.parent().mapToGlobal(pr.topLeft()).y() + y)
            # Simpler: just position relative to parent widget
            self.move(x, y)


def discover_models():
    """Scan known directories for .pt checkpoint files."""
    search_dirs = ["model/annotations", "models"]
    found = []
    for d in search_dirs:
        if os.path.isdir(d):
            found.extend(glob.glob(os.path.join(d, "*.pt")))
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
                        padding: 6px 12px; font-size: 12px; font-weight: 500;
                    }}
                    QPushButton:hover {{ background: {T.ACCENT}; opacity: 0.85; }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {T.BTN_DARK}; color: {T.TEXT_MUTED}; border: none;
                        padding: 6px 12px; font-size: 12px; font-weight: 500;
                    }}
                    QPushButton:hover {{ background: {T.SEPARATOR}; color: {T.BTN_TEXT}; }}
                """)

    def active(self) -> int:
        return self._active


# Toolbar Button
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


# Lead Sidebar (for 1-lead mode)
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


class ViewerPage(QWidget):
    """Main EKG viewer with toolbar, view modes, panels, navigation."""

    open_file = Signal()
    show_report = Signal()
    toggle_dark = Signal()

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
        self._view_mode = 0       # 0=12lead, 1=1lead, 2=monitor
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

    def _scrubber_style(self):
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

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Row 1: Navigation & view ──
        self.toolbar = QWidget()
        self.toolbar.setFixedHeight(36)
        self.toolbar.setStyleSheet(f"background: {T.TOPBAR};")
        tb = QHBoxLayout(self.toolbar)
        tb.setContentsMargins(10, 0, 10, 0)
        tb.setSpacing(6)

        logo = make_logo(12)
        tb.addWidget(logo)
        tb.addWidget(make_separator())

        self.file_label = QLabel("")
        self.file_label.setStyleSheet(f"font-size:11px; color:{T.BTN_TEXT}; font-family:Menlo;")
        tb.addWidget(self.file_label)
        tb.addStretch()

        self.analysis_badge = QLabel("Analiza zakończona")
        self.analysis_badge.setFont(QFont(".AppleSystemUIFont", 11))
        self.analysis_badge.setFixedHeight(24)
        self.analysis_badge.setStyleSheet(f"""
            font-size: 11px; background: {T.BADGE_NORM_BG}; color: {T.BADGE_NORM_TEXT};
            padding: 0px 8px; border-radius: 3px; font-weight: 600;
        """)
        self.analysis_badge.hide()
        tb.addWidget(self.analysis_badge)

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

        # ── Row 2: Tools & analysis ──
        self.toolbar2 = QWidget()
        self.toolbar2.setFixedHeight(36)
        self.toolbar2.setStyleSheet(f"background: {T.TOPBAR};")
        tb2 = QHBoxLayout(self.toolbar2)
        tb2.setContentsMargins(10, 0, 10, 0)
        tb2.setSpacing(6)

        self.view_seg = SegmentedControl(["12-Lead", "1-Lead", "Monitor"])
        self.view_seg.changed.connect(self._on_view_mode)
        tb2.addWidget(self.view_seg)
        tb2.addWidget(make_separator())
        tb2.addStretch()

        # Model selector
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(140)
        self.model_combo.setStyleSheet(f"""
            QComboBox {{
                color: {T.BTN_TEXT}; background: {T.BTN_DARK};
                border: 1px solid {T.SEPARATOR}; border-radius: 5px;
                padding: 3px 6px; font-size: 11px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                color: {T.TEXT}; background: {T.WHITE};
                selection-background-color: {T.ACCENT};
                selection-color: {T.ACCENT_TEXT};
            }}
        """)
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        tb2.addWidget(self.model_combo)

        # Full analysis button (replaces Autoskan + Zaznacz do analizy + Analizuj)
        from ui.theme import is_dark_mode as _idm
        _ah = '#00c864' if _idm() else '#3a8eef'
        self.btn_full_analysis = QPushButton("Pelna Analiza")
        self.btn_full_analysis.setObjectName("primary")
        self.btn_full_analysis.setCursor(Qt.PointingHandCursor)
        self.btn_full_analysis.setStyleSheet(f"""
            QPushButton {{ background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;
                padding:5px 14px;border-radius:5px;font-weight:600;font-size:12px; }}
            QPushButton:hover {{ background:{_ah}; }}
        """)
        self.btn_full_analysis.clicked.connect(self._run_full_analysis)
        tb2.addWidget(self.btn_full_analysis)

        tb2.addWidget(make_separator())

        self.btn_report = QPushButton("Raport")
        self.btn_report.setObjectName("secondary")
        self.btn_report.setCursor(Qt.PointingHandCursor)
        self.btn_report.clicked.connect(self.show_report.emit)
        tb2.addWidget(self.btn_report)

        outer.addWidget(self.toolbar2)

        # Content area
        self.content = QHBoxLayout()
        self.content.setContentsMargins(0, 0, 0, 0)
        self.content.setSpacing(0)

        self.info_panel = InfoPanel()
        self.info_panel.patient_changed.connect(self._save_ann)
        self.content.addWidget(self.info_panel)

        self.monitor_sidebar = MonitorSidebar()
        self.monitor_sidebar.pause_toggled.connect(self._on_monitor_pause)
        self.monitor_sidebar.speed_changed.connect(self._on_monitor_speed)
        self.monitor_sidebar.leads_changed.connect(self._on_monitor_leads)
        self.monitor_sidebar.hide()
        self.content.addWidget(self.monitor_sidebar)

        self.lead_sidebar = LeadSidebar()
        self.lead_sidebar.lead_selected.connect(self._on_lead_selected)
        self.lead_sidebar.hide()
        self.content.addWidget(self.lead_sidebar)

        # Stacked views
        self.view_stack = QStackedWidget()
        self.view_stack.setStyleSheet(f"background: {T.BG_SECONDARY};")

        self.grid_12 = TwelveLeadGrid()
        self.view_stack.addWidget(self.grid_12)
        # Connect grid double-click for jump to 1-lead
        if hasattr(self.grid_12, 'cell_double_clicked'):
            self.grid_12.cell_double_clicked.connect(self._on_cell_double_click)

        self.single_lead = SingleLeadCanvas()
        self.single_lead.draw_border = True
        # Connect selection_completed for the new marking workflow
        if hasattr(self.single_lead, 'selection_completed'):
            self.single_lead.selection_completed.connect(self._on_selection_completed)
        if hasattr(self.single_lead, 'right_clicked'):
            self.single_lead.right_clicked.connect(self._on_canvas_right_click)
        if hasattr(self.single_lead, 'selection_live'):
            self.single_lead.selection_live.connect(self._on_selection_live)
        self.view_stack.addWidget(self.single_lead)

        self.monitor_area = QWidget()
        self.monitor_area.setStyleSheet(f"background: {T.BG_SECONDARY};")
        self._monitor_strips: list[EkgCellCanvas] = []
        self._build_monitor_area()
        self.view_stack.addWidget(self.monitor_area)

        self.content.addWidget(self.view_stack, stretch=1)

        # Markings panel (replaces caliper, annotation, results panels)
        self.markings_panel = MarkingsPanel()
        self.markings_panel.hide()
        self.markings_panel.marking_hovered.connect(self._on_marking_hovered)
        self.markings_panel.marking_unhovered.connect(self._on_marking_unhovered)
        self.markings_panel.marking_selected.connect(self._on_marking_selected)
        self.markings_panel.marking_deleted.connect(self._on_marking_deleted)
        self.markings_panel.undo_requested.connect(self._undo)
        self.markings_panel.redo_requested.connect(self._redo)
        self.content.addWidget(self.markings_panel)

        content_widget = QWidget()
        content_widget.setLayout(self.content)
        outer.addWidget(content_widget, stretch=1)

        # Navigation bar
        self.navbar = QWidget()
        self.navbar.setFixedHeight(48)
        self.navbar.setStyleSheet(f"background:{T.WHITE}; border-top:1px solid {T.BORDER};")
        nav = QHBoxLayout(self.navbar)
        nav.setContentsMargins(16, 0, 16, 0)
        nav.setSpacing(8)

        # Navigation buttons — standard playback icons
        # ▮◀  ◀◀  ◀  [❚❚/▶]  ▶  ▶▶  ▶▮
        # Using U+25AE (▮ thick bar) for skip endpoints for visual balance
        _TRI_L = "\u25c0"  # ◀ BLACK LEFT-POINTING TRIANGLE
        _TRI_R = "\u25b6"  # ▶ BLACK RIGHT-POINTING TRIANGLE
        _BAR = "\u25ae"    # ▮ BLACK VERTICAL RECTANGLE (thick, matches triangle weight)
        back_buttons = [
            (_BAR + _TRI_L, self._nav_start),
            (_TRI_L + _TRI_L, lambda: self._nav_step(-1.0)),
            (_TRI_L, lambda: self._nav_step(-0.2)),
        ]
        fwd_buttons = [
            (_TRI_R, lambda: self._nav_step(0.2)),
            (_TRI_R + _TRI_R, lambda: self._nav_step(1.0)),
            (_TRI_R + _BAR, self._nav_end),
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

        # Pause/resume button — accent-colored to distinguish from nav step buttons
        self.pause_btn = QPushButton("\u275a\u275a")  # ❚❚ pause
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self._apply_pause_btn_style()
        self.pause_btn.clicked.connect(self._on_navbar_pause)
        self.pause_btn.hide()
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
        self.time_label.setStyleSheet(f"font-size:12px; font-family:Menlo; color:{T.TEXT_SECONDARY};")
        nav.addWidget(self.time_label)

        # Zoom controls
        nav.addWidget(make_separator())

        self._zoom_out_btn = QPushButton("\u2212")  # −
        self._zoom_out_btn.setObjectName("nav")
        self._zoom_out_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_out_btn.setToolTip("Oddal (pokaż więcej)")
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        nav.addWidget(self._zoom_out_btn)

        self._zoom_label = QLabel("3.0 s")
        self._zoom_label.setStyleSheet(f"font-size:11px; font-family:Menlo; color:{T.TEXT_MUTED}; min-width:40px;")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self._zoom_label)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setObjectName("nav")
        self._zoom_in_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_in_btn.setToolTip("Przybliż (pokaż mniej)")
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        nav.addWidget(self._zoom_in_btn)

        self._zoom_reset_btn = QPushButton("Reset")
        self._zoom_reset_btn.setObjectName("nav")
        self._zoom_reset_btn.setCursor(Qt.PointingHandCursor)
        self._zoom_reset_btn.setToolTip("Resetuj powiększenie")
        self._zoom_reset_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 10px; padding: 0 8px; height: 28px;
                border: 1px solid {T.BORDER}; border-radius: 6px;
                background: {T.WHITE}; color: {T.TEXT_MUTED};
            }}
            QPushButton:hover {{ background: {T.BG_SECONDARY}; color: {T.TEXT}; }}
        """)
        self._zoom_reset_btn.clicked.connect(self._reset_zoom)
        nav.addWidget(self._zoom_reset_btn)

        self.speed_label = QLabel("25 mm/s")
        self.speed_label.setStyleSheet(f"font-size:11px; color:{T.TEXT_DIM};")
        nav.addWidget(self.speed_label)

        outer.addWidget(self.navbar)

        # Status bar
        self.statusbar = QWidget()
        self.statusbar.setFixedHeight(28)
        self.statusbar.setStyleSheet(f"background:{T.WHITE}; border-top:1px solid {T.BORDER};")
        sb = QHBoxLayout(self.statusbar)
        sb.setContentsMargins(14, 0, 14, 0)
        sb.setSpacing(16)

        self.st_left = QLabel()
        self.st_left.setStyleSheet(f"font-size:11px; color:{T.TEXT_MUTED}; font-family:Menlo;")
        sb.addWidget(self.st_left)

        self.st_center = QLabel()
        self.st_center.setStyleSheet(f"font-size:10px; color:{T.TEXT_DIM};")
        self.st_center.setAlignment(Qt.AlignCenter)
        sb.addWidget(self.st_center, stretch=1)

        self.st_right = QLabel()
        self.st_right.setStyleSheet(f"font-size:11px; color:{T.TEXT_MUTED}; font-family:Menlo;")
        sb.addWidget(self.st_right)

        # Selection indicator (shows during active selection)
        self._sel_indicator = QLabel()
        self._sel_indicator.setAlignment(Qt.AlignCenter)
        self._sel_indicator.setFixedHeight(22)
        self._sel_indicator.hide()
        sb.addWidget(self._sel_indicator)

        outer.addWidget(self.statusbar)
        self._update_time_display()
        self._update_statusbar()

    def apply_theme(self):
        """Re-apply all styles after theme change."""
        self.toolbar.setStyleSheet(f"background: {T.TOPBAR};")
        self.toolbar2.setStyleSheet(f"background: {T.TOPBAR};")
        self.file_label.setStyleSheet(f"font-size:11px; color:{T.BTN_TEXT}; font-family:Menlo;")
        self.analysis_badge.setStyleSheet(f"""
            font-size: 11px; background: {T.BADGE_NORM_BG}; color: {T.BADGE_NORM_TEXT};
            padding: 0px 8px; border-radius: 3px; font-weight: 600;
        """)
        self.view_seg._apply_styles()

        from ui.theme import is_dark_mode as _idm
        _ah = '#00c864' if _idm() else '#3a8eef'
        self.btn_full_analysis.setStyleSheet(f"""
            QPushButton {{ background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;
                padding:5px 14px;border-radius:5px;font-weight:600;font-size:12px; }}
            QPushButton:hover {{ background:{_ah}; }}
        """)
        self.model_combo.setStyleSheet(f"""
            QComboBox {{
                color: {T.BTN_TEXT}; background: {T.BTN_DARK};
                border: 1px solid {T.SEPARATOR}; border-radius: 5px;
                padding: 4px 8px; font-size: 11px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                color: {T.TEXT}; background: {T.WHITE};
                selection-background-color: {T.ACCENT};
                selection-color: {T.ACCENT_TEXT};
            }}
        """)

        from ui.theme import is_dark_mode
        self.btn_dark.setText("Tryb jasny" if is_dark_mode() else "Tryb ciemny")

        self.lead_sidebar._update_styles()
        self.view_stack.setStyleSheet(f"background: {T.BG_SECONDARY};")
        self.monitor_area.setStyleSheet(f"background: {T.BG_SECONDARY};")

        self.info_panel.apply_theme()
        self.monitor_sidebar.apply_theme()
        self.markings_panel.apply_theme()

        self.navbar.setStyleSheet(f"background:{T.WHITE}; border-top:1px solid {T.BORDER};")
        self.scrubber.setStyleSheet(self._scrubber_style())
        self.time_label.setStyleSheet(f"font-size:12px; font-family:Menlo; color:{T.TEXT_SECONDARY};")
        self.speed_label.setStyleSheet(f"font-size:11px; color:{T.TEXT_DIM};")

        self.statusbar.setStyleSheet(f"background:{T.WHITE}; border-top:1px solid {T.BORDER};")
        self.st_left.setStyleSheet(f"font-size:11px; color:{T.TEXT_MUTED}; font-family:Menlo;")
        self.st_center.setStyleSheet(f"font-size:10px; color:{T.TEXT_DIM};")
        self.st_right.setStyleSheet(f"font-size:11px; color:{T.TEXT_MUTED}; font-family:Menlo;")
        self.grid_12.apply_theme()

    def _build_monitor_area(self):
        layout = QVBoxLayout(self.monitor_area)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)
        self._monitor_strips = []
        for lead_name in ["II", "V1", "V5"]:
            strip = EkgCellCanvas()
            strip.draw_border = True
            strip.show_cal = False
            self._monitor_strips.append((lead_name, strip))
            layout.addWidget(strip, stretch=1)

    def set_signal(self, signal: np.ndarray, leads: list[str], fs: int, filename: str = "",
                   ground_truth: dict | None = None, patient_info: dict | None = None,
                   base_path: str = ""):
        """Load new signal data into the viewer."""
        self._ground_truth = ground_truth
        self._base_path = base_path
        self._marking_store.clear()
        self.grid_12.clear()
        self.single_lead.clear()
        for _, strip in self._monitor_strips:
            strip.clear()

        self.signal = signal
        self.leads = leads
        self.fs = fs
        self.filename = filename
        self.duration = signal.shape[0] / fs
        self.time_pos = 0.0
        self._monitor_t = 0.0
        self._monitor_playing = False
        self._monitor_timer.stop()

        self._global_min = float(signal.min())
        self._global_max = float(signal.max())
        v_range = self._global_max - self._global_min
        pad = max(v_range * 0.15, 0.2)
        self._v_min = self._global_min - pad
        self._v_max = self._global_max + pad

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
            self.btn_full_analysis.setToolTip("Analiza wymaga co najmniej 10s nagrania")
        else:
            self.btn_full_analysis.setEnabled(True)
            self.btn_full_analysis.setToolTip("")

        max_window = max(self._window_12, self._window_1)
        self._scrubber_max = max(0.0, self.duration - max_window)
        self.scrubber.setRange(0, int(self._scrubber_max * 100))
        self.scrubber.setValue(0)

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

        self._refresh_views()
        self._update_time_display()
        self._update_statusbar()

    def _refresh_views(self):
        if self.signal is None:
            return
        self.grid_12.set_signal(self.signal, self.leads, self.fs, self.time_pos,
                                self._window_12, self._v_min, self._v_max)
        self._refresh_single_lead()
        self._refresh_monitor()

    def _refresh_single_lead(self):
        if self.signal is None:
            return
        lead = self.lead_sidebar.active_lead()
        if lead in self.leads:
            idx = self.leads.index(lead)
            window = self._window_1
            t_start = max(0.0, self.time_pos)
            t_end = min(self.duration, t_start + window)
            if t_end - t_start < window:
                t_start = max(0.0, t_end - window)
            self.single_lead.v_min = self._v_min
            self.single_lead.v_max = self._v_max
            self.single_lead.set_data(lead, self.signal[:, idx], self.fs, t_start, t_end)

            # Set markings for this lead from the store
            lead_markings = [m for m in self._marking_store.get_all() if m.lead == lead]
            self.single_lead.markings = [
                {"id": m.id, "t1": m.t1, "t2": m.t2, "type": m.type, "label": m.label}
                for m in lead_markings
            ]

            self.single_lead.update()

    def _refresh_monitor(self):
        if self.signal is None:
            return
        sweep_frac = (self._monitor_t - self._monitor_page_start) / self._monitor_window \
            if self._monitor_window > 0 else 0
        sweep_frac = max(0.0, min(1.0, sweep_frac))
        for lead_name, strip in self._monitor_strips:
            if lead_name in self.leads:
                idx = self.leads.index(lead_name)
                strip.v_min = self._v_min
                strip.v_max = self._v_max
                strip.set_data(lead_name, self.signal[:, idx], self.fs,
                               self._monitor_page_start,
                               self._monitor_page_start + self._monitor_window)
                strip.set_sweep(sweep_frac)

    def _on_view_mode(self, idx: int):
        self._view_mode = idx
        if idx != 2:
            self._monitor_timer.stop()
            self._monitor_playing = False

        # Show pause button only in monitor mode
        self.pause_btn.setVisible(idx == 2)

        self.info_panel.setVisible(idx == 0)
        self.lead_sidebar.setVisible(idx == 1)
        self.monitor_sidebar.setVisible(idx == 2)
        self.markings_panel.setVisible(idx == 1)

        if idx == 0:
            self.info_panel.show()
            self.navbar.show()
            self._restore_scrubber_range()
        elif idx == 1:
            self.info_panel.hide()
            self.markings_panel.show()
            self._refresh_markings()
            self.navbar.show()
            self._restore_scrubber_range()
        elif idx == 2:
            self.info_panel.hide()
            self.markings_panel.hide()
            self.navbar.show()
            self._start_monitor()

        self.view_stack.setCurrentIndex(idx)
        self._update_statusbar()

    def _on_lead_selected(self, lead: str):
        self._refresh_single_lead()
        self._refresh_markings()

    def _populate_model_combo(self):
        models = discover_models()
        self.model_combo.clear()
        if models:
            for path in models:
                self.model_combo.addItem(os.path.basename(path), path)
        else:
            self.model_combo.addItem("(brak modeli)", "")

    def _on_model_changed(self, idx: int):
        new_path = self.model_combo.currentData()
        if new_path != self._model_path:
            self._model = None
            self._model_device = None
            self._model_path = None

    def _ensure_model_loaded(self):
        """Load model if not cached. Returns (model, device) or raises."""
        from model.inference_api import load_checkpoint_model
        path = self.model_combo.currentData()
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Nie znaleziono modelu: {path}")
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
        lead = self.lead_sidebar.active_lead()
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
        """Clear selection preview on canvas."""
        if hasattr(self.single_lead, 'pending_marker'):
            self.single_lead.pending_marker = None
        if hasattr(self.single_lead, 'selection_preview'):
            self.single_lead.selection_preview = None
        self.single_lead.update()
        self._sel_indicator.hide()

    def _show_annotation_form(self, lead, t1, t2):
        """Create an annotation marking with default category."""
        marking = Marking(type="annotation", lead=lead, t1=t1, t2=t2,
                          category="Patologia", source="user")
        self._marking_store.add(marking)
        self._refresh_markings()
        self.markings_panel.set_selected(marking.id)

    def _refresh_markings(self):
        """Sync marking store to canvas and panel."""
        lead = self.lead_sidebar.active_lead()
        # Canvas: set markings for current lead
        lead_markings = [m for m in self._marking_store.get_all() if m.lead == lead]
        self.single_lead.markings = [
            {"id": m.id, "t1": m.t1, "t2": m.t2, "type": m.type, "label": m.label}
            for m in lead_markings
        ]
        self.single_lead.update()

        # Panel: show all markings
        self.markings_panel.set_markings(self._marking_store.get_all())
        self.markings_panel.set_undo_enabled(self._marking_store.can_undo)
        self.markings_panel.set_redo_enabled(self._marking_store.can_redo)

    def _on_marking_hovered(self, marking_id: str):
        if hasattr(self.single_lead, 'hovered_marking'):
            self.single_lead.hovered_marking = marking_id
            self.single_lead.update()

    def _on_marking_unhovered(self):
        if hasattr(self.single_lead, 'hovered_marking'):
            self.single_lead.hovered_marking = None
            self.single_lead.update()

    def _on_marking_selected(self, marking_id: str):
        if hasattr(self.single_lead, 'selected_marking'):
            self.single_lead.selected_marking = marking_id
            self.single_lead.update()

    def _on_marking_deleted(self, marking_id: str):
        self._marking_store.delete(marking_id)
        self._refresh_markings()
        self._save_ann()

    def _undo(self):
        if self._marking_store.undo():
            self._refresh_markings()
            self._save_ann()

    def _redo(self):
        if self._marking_store.redo():
            self._refresh_markings()
            self._save_ann()

    def _on_cell_double_click(self, lead: str, time: float):
        """Double-click on 12-lead grid cell -- jump to 1-lead view."""
        self.view_seg.set_active(1)
        self.lead_sidebar._select(lead)
        self.time_pos = max(0, time - self._window_1 / 2)
        self._restore_scrubber_range()

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

    def _zoom_in(self):
        """Decrease the time window (show less time, more detail)."""
        current = self._window_1
        for step in self._ZOOM_STEPS:
            if step < current - 0.01:
                new_window = step
            else:
                break
        else:
            new_window = self._ZOOM_STEPS[0]
        self._window_1 = max(new_window, self._ZOOM_STEPS[0])
        self._update_zoom_label()
        self._restore_scrubber_range()
        self._refresh_single_lead()

    def _zoom_out(self):
        """Increase the time window (show more time, less detail)."""
        current = self._window_1
        max_window = min(self.duration, self._ZOOM_STEPS[-1])
        for step in self._ZOOM_STEPS:
            if step > current + 0.01:
                self._window_1 = min(step, max_window)
                break
        else:
            self._window_1 = max_window
        self._update_zoom_label()
        self._restore_scrubber_range()
        self._refresh_single_lead()

    def _reset_zoom(self):
        """Reset 1-lead view to default 3-second window."""
        self._window_1 = min(3.0, self.duration)
        self._update_zoom_label()
        self._restore_scrubber_range()
        self._refresh_single_lead()

    def _update_zoom_label(self):
        if hasattr(self, '_zoom_label'):
            self._zoom_label.setText(f"{self._window_1:.1f} s")

    def _on_selection_live(self, t1, t2):
        """Update the selection indicator in the status bar during live selection."""
        dt = t2 - t1
        dt_ms = dt * 1000

        # Build indicator text
        time_text = f"{dt:.2f} s ({dt_ms:.0f} ms)"

        if dt >= 10.0:
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
        self._sel_indicator.show()

    def _zoom_to_region(self, t1, t2):
        """Zoom the 1-lead view to a specific time region."""
        pad = (t2 - t1) * 0.1
        self.time_pos = max(0, t1 - pad)
        self._window_1 = (t2 - t1) + 2 * pad
        self._restore_scrubber_range()
        self._refresh_single_lead()

    def _export_region_png(self, t1, t2):
        """Export the current canvas view as a PNG file."""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Eksportuj fragment", "fragment.png", "PNG (*.png)")
        if path:
            pixmap = self.single_lead.grab()
            pixmap.save(path, "PNG")

    # ── Autoscan ────────────────────────────────────────────────

    def _autoscan_cache_path(self) -> str:
        key = f"{self._autoscan_file_path}:{self._model_path}"
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

    def _load_ann(self):
        """Load markings from .ann file."""
        path = self._ann_path()
        if not path or not os.path.exists(path):
            return
        self._marking_store.load_ann(path)
        self._refresh_markings()

    def _save_ann(self):
        """Save current markings to .ann file."""
        path = self._ann_path()
        if not path:
            return
        self._marking_store.save_ann(path)

    def _run_full_analysis(self):
        """Run full analysis: sliding window scan across the entire signal."""
        if self.signal is None:
            return

        # Try cache first
        self._autoscan_file_path = self.filename
        cached = self._load_autoscan_cache() if self._model_path else None
        if cached:
            self._autoscan_results = cached
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

                top_cls = max(prob_dict, key=prob_dict.get)
                top_prob = prob_dict[top_cls]
                if top_cls == "class_healthy" and top_prob >= 0.5:
                    color = 0
                elif top_cls != "class_healthy" and top_prob >= 0.5:
                    color = 2
                else:
                    color = 1

                results.append({
                    "t_start": t_start, "t_end": t_end,
                    "color": color, "probs": prob_dict,
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

        self._update_statusbar()

    def _apply_autoscan_results(self):
        """Create Marking objects from autoscan results and add to the store."""
        if not self._autoscan_results:
            return
        # Remove old scan markings before adding new ones
        old_scans = [m for m in self._marking_store.get_all() if m.type == "scan"]
        for m in old_scans:
            self._marking_store.delete(m.id)
        # Clear undo/redo for scan bulk operations
        self._marking_store._undo_stack.clear()
        self._marking_store._redo_stack.clear()

        for r in self._autoscan_results:
            marking = Marking(
                type="scan",
                lead="all",
                t1=r["t_start"],
                t2=r["t_end"],
                probs=r.get("probs"),
                color_code=r.get("color", 0),
                source="ai",
            )
            self._marking_store.add(marking)
        # Clear undo/redo again (bulk add should not be undoable)
        self._marking_store._undo_stack.clear()
        self._marking_store._redo_stack.clear()
        self._refresh_markings()
        self._save_ann()

    def _apply_autoscan_overlay(self):
        if not self._autoscan_results:
            return
        from ui.theme import CLASS_NAMES_PL

        # Build ground truth annotation lines
        gt_lines = self._build_gt_lines(CLASS_NAMES_PL)

        regions = []
        for r in self._autoscan_results:
            code = r["color"]
            label_lines = None
            if code != 0 and r.get("probs"):
                sorted_probs = sorted(r["probs"].items(), key=lambda x: x[1], reverse=True)
                lines = []
                for cls, prob in sorted_probs[:2]:
                    name = CLASS_NAMES_PL.get(cls, cls)
                    if len(name) > 22:
                        name = name[:20] + "."
                    lines.append(f"{name} {prob * 100:.0f}%")
                # Mismatch detection: add ⚠ if GT says healthy or no GT
                if lines and self._is_gt_mismatch(r["t_start"], r["t_end"]):
                    lines[0] = "\u26a0 " + lines[0]
                label_lines = lines
            regions.append((r["t_start"], r["t_end"], code, label_lines))

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
            QMessageBox.warning(self, "Blad", f"Nie udalo sie zaladowac modelu:\n{e}")
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

    def _apply_pause_btn_style(self):
        from ui.theme import is_dark_mode
        hover = '#00c864' if is_dark_mode() else '#3a8eef'
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
        self._monitor_playing = not self._monitor_playing
        if self._monitor_playing:
            self._monitor_timer.start()
            self.pause_btn.setText("\u275a\u275a")  # ❚❚ pause
        else:
            self._monitor_timer.stop()
            self.pause_btn.setText("\u25b6")  # ▶ play

    def _nav_start(self):
        if self._view_mode == 2:
            self._monitor_seek(0.0)
        else:
            self.time_pos = 0
            self.scrubber.setValue(0)

    def _nav_end(self):
        if self._view_mode == 2:
            self._monitor_seek(self.duration - 0.05)
        else:
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
        if self._view_mode == 2:
            new_t = max(0.0, min(self.duration - 0.05, self._monitor_t + dt))
            self._monitor_seek(new_t)
        else:
            self.time_pos = max(0, min(self._scrubber_max, self.time_pos + dt))
            self.scrubber.setValue(int(self.time_pos * 100))

    def _on_scrubber(self, value: int):
        if self._view_mode == 2:
            new_t = value / 100.0
            self._monitor_seek(new_t)
            return
        self.time_pos = value / 100.0
        self._update_time_display()
        if self._view_mode == 0:
            self.grid_12.set_signal(self.signal, self.leads, self.fs, self.time_pos,
                                    self._window_12, self._v_min, self._v_max)
        elif self._view_mode == 1:
            self._refresh_single_lead()

    def _restore_scrubber_range(self):
        """Restore scrubber range for 12-lead/1-lead modes."""
        max_window = max(self._window_12, self._window_1)
        self._scrubber_max = max(0.0, self.duration - max_window)
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, int(self._scrubber_max * 100))
        self.scrubber.setValue(int(self.time_pos * 100))
        self.scrubber.blockSignals(False)

    def _monitor_seek(self, t: float):
        """Jump monitor playback to a specific time."""
        t = max(0.0, min(self.duration - 0.05, t))
        self._monitor_t = t
        self._monitor_page_start = max(0.0, (t // self._monitor_window) * self._monitor_window)
        # Clamp last page
        if self._monitor_page_start + self._monitor_window > self.duration:
            self._monitor_page_start = max(0.0, self.duration - self._monitor_window)
        for _, strip in self._monitor_strips:
            strip._old_signal = None
        self._refresh_monitor()
        self._update_monitor_time_display()
        # Sync scrubber without re-triggering _on_scrubber
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(int(t * 100))
        self.scrubber.blockSignals(False)

    def _update_time_display(self):
        window = self._window_1 if self._view_mode == 1 else self._window_12
        t_end = min(self.duration, self.time_pos + window)
        self.time_label.setText(f"{self.time_pos:.2f} – {t_end:.2f} s / {self.duration:.2f} s")

    def _update_monitor_time_display(self):
        page_end = min(self._monitor_page_start + self._monitor_window, self.duration)
        self.time_label.setText(
            f"{self._monitor_page_start:.2f} – {page_end:.2f} s | "
            f"t = {self._monitor_t:.2f} s / {self.duration:.2f} s"
        )
        # Sync scrubber position
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(int(self._monitor_t * 100))
        self.scrubber.blockSignals(False)

    def _start_monitor(self):
        self._monitor_t = 0.0
        self._monitor_window = min(3.0, self.duration)
        self._monitor_page_start = 0.0
        self._monitor_playing = True
        self._monitor_speed = 1.0
        self._monitor_timer.setInterval(50)
        # Set scrubber range to full signal duration
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, int(self.duration * 100))
        self.scrubber.setValue(0)
        self.scrubber.blockSignals(False)
        # Reset pause state
        self.pause_btn.setText("\u275a\u275a")  # ❚❚ pause
        for _, strip in self._monitor_strips:
            strip._old_signal = None
            strip._sweep_pos = None
        self._monitor_timer.start()
        self._refresh_monitor()
        self._update_monitor_time_display()

    def _monitor_tick(self):
        if not self._monitor_playing:
            return
        self._monitor_t += 0.05

        # Loop back when we reach the end of the signal
        if self._monitor_t >= self.duration:
            self._monitor_t = 0.0
            self._monitor_page_start = 0.0
            for _, strip in self._monitor_strips:
                strip._old_signal = None
                strip._sweep_pos = None
            self._refresh_monitor()
            self._update_monitor_time_display()
            return

        page_end = self._monitor_page_start + self._monitor_window
        if self._monitor_t >= page_end:
            for lead_name, strip in self._monitor_strips:
                if lead_name in self.leads:
                    idx = self.leads.index(lead_name)
                    strip._old_signal = self.signal[:, idx]
                    strip._old_t_start = self._monitor_page_start
                    strip._old_t_end = page_end
            self._monitor_page_start = page_end
            # Clamp last page so it doesn't extend past signal duration
            if self._monitor_page_start + self._monitor_window > self.duration:
                self._monitor_page_start = max(0.0, self.duration - self._monitor_window)

        self._refresh_monitor()
        self._update_monitor_time_display()

    def _on_monitor_pause(self, paused: bool):
        self._monitor_playing = not paused
        if self._monitor_playing:
            self._monitor_timer.start()
            self.pause_btn.setText("\u275a\u275a")  # ❚❚ pause
        else:
            self._monitor_timer.stop()
            self.pause_btn.setText("\u25b6")  # ▶ play

    def _on_monitor_speed(self, speed: float):
        self._monitor_speed = speed
        self._monitor_timer.setInterval(int(50 / speed))
        self._update_statusbar()

    def _on_monitor_leads(self, active_leads: list):
        """Rebuild monitor strips with selected leads."""
        layout = self.monitor_area.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._monitor_strips = []
        for lead_name in active_leads:
            strip = EkgCellCanvas()
            strip.draw_border = True
            strip.show_cal = False
            self._monitor_strips.append((lead_name, strip))
            layout.addWidget(strip, stretch=1)
        # Restart monitor playback with new leads
        self._start_monitor()

    def _update_statusbar(self):
        if self._view_mode == 0:
            self.st_left.setText("<b>10 mm/mV</b> | <b>25 mm/s</b> | 0.05-150 Hz")
            self.st_center.setText("1/2/3: Widok | \u2190/\u2192: Przewin")
            self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b> | V = <b>0.85 mV</b>")
        elif self._view_mode == 1:
            self.st_left.setText("<b>10 mm/mV</b> | <b>25 mm/s</b>")
            self.st_center.setText("Zaznacz region na sygnale | Cmd+Z: Cofnij | Cmd+Shift+Z: Ponow")
            self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b>")
        elif self._view_mode == 2:
            speed_label = f"{self._monitor_speed:g}x"
            self.st_left.setText(f"<b>Monitor</b> | <b>25 mm/s</b> | {speed_label}")
            self.st_center.setText("Space: Pauza | Esc: Wyjdz z monitora | \u2191\u2193: Predkosc")
            self.st_right.setText(f"t = <b>{self._monitor_t:.2f} s</b> / {self.duration:.2f} s")
