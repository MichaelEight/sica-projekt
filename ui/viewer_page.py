"""Main EKG viewer page with toolbar, view modes, panels, navigation bar."""
import glob
import os
import time

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QStackedWidget, QSlider,
                                QSizePolicy, QComboBox, QApplication)

import ui.theme as T
from ui.theme import STANDARD_LEADS, TARGET_CLASSES
from ui.widgets import make_logo, make_separator
from ui.ekg_canvas import (EkgCellCanvas, TwelveLeadGrid, SingleLeadCanvas,
                            generate_demo_signal, synth_ekg, LEAD_SEEDS, LEAD_AMPS)
from ui.panels import (InfoPanel, CaliperPanel, AnnotationPanel, ResultsPanel,
                        MonitorSidebar)


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
                    background: {T.ACCENT}; color: {T.ACCENT_TEXT}; border: none;
                    padding: 6px 12px; font-size: 12px; font-weight: 500;
                """)
            else:
                btn.setStyleSheet(f"""
                    background: {T.BTN_DARK}; color: {T.TEXT_MUTED}; border: none;
                    padding: 6px 12px; font-size: 12px; font-weight: 500;
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
            self.setStyleSheet(f"background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;padding:6px 10px;border-radius:5px;")
        else:
            self.setStyleSheet(f"background:{T.BTN_DARK};color:{T.BTN_TEXT};border:none;padding:6px 10px;border-radius:5px;")

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
                    background: {T.ACCENT}; color: {T.ACCENT_TEXT}; border: 1px solid {T.ACCENT};
                    border-radius: 4px; font-family: Menlo;
                """)
            else:
                btn.setStyleSheet(f"""
                    background: {T.WHITE}; color: {T.TEXT_MUTED}; border: 1px solid {T.BORDER};
                    border-radius: 4px; font-family: Menlo;
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
        self._tool_mode = 0       # 0=select, 1=caliper, 2=annotation
        self._show_results = False
        self._model = None
        self._model_device = None
        self._model_path = None
        self._last_results = None
        self._analysis_mode = False
        self._analysis_start = None
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

        self.file_label = QLabel("00888_lr.dat | 500 Hz | 12 odpr. | 10.0 s")
        self.file_label.setStyleSheet(f"font-size:11px; color:{T.BTN_TEXT}; font-family:Menlo;")
        tb.addWidget(self.file_label)
        tb.addStretch()

        self.analysis_badge = QLabel("Analiza zakończona")
        self.analysis_badge.setStyleSheet(f"""
            font-size: 11px; background: {T.BADGE_NORM_BG}; color: {T.BADGE_NORM_TEXT};
            padding: 4px 10px; border-radius: 4px; font-weight: 600;
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

        # Tool buttons
        self.tool_btns: list[ToolbarBtn] = []
        for i, (label, active) in enumerate([("Wybierz", True), ("Suwmiarka", False), ("Adnotacja", False)]):
            btn = ToolbarBtn(label, active)
            btn.clicked.connect(lambda checked, idx=i: self._on_tool_mode(idx))
            self.tool_btns.append(btn)
            tb2.addWidget(btn)

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

        # Analysis selection toggle
        self.btn_mark_analysis = ToolbarBtn("Zaznacz do analizy", False)
        self.btn_mark_analysis.clicked.connect(self._toggle_analysis_mode)
        tb2.addWidget(self.btn_mark_analysis)

        # Analyze
        self.btn_analyze = QPushButton("Analizuj")
        self.btn_analyze.setObjectName("primary")
        self.btn_analyze.setCursor(Qt.PointingHandCursor)
        self.btn_analyze.setStyleSheet(
            f"background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;"
            f"padding:5px 14px;border-radius:5px;font-weight:600;font-size:12px;"
        )
        self.btn_analyze.clicked.connect(self._on_analyze)
        tb2.addWidget(self.btn_analyze)

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
        # Connect grid cell clicks for analysis mode
        for cell in self.grid_12.cells.values():
            cell.clicked.connect(self._on_canvas_click)
        self.grid_12.rhythm.clicked.connect(self._on_canvas_click)

        self.single_lead = SingleLeadCanvas()
        self.single_lead.draw_border = True
        self.single_lead.clicked.connect(self._on_canvas_click)
        self.view_stack.addWidget(self.single_lead)

        self.monitor_area = QWidget()
        self.monitor_area.setStyleSheet(f"background: {T.BG_SECONDARY};")
        self._monitor_strips: list[EkgCellCanvas] = []
        self._build_monitor_area()
        self.view_stack.addWidget(self.monitor_area)

        self.content.addWidget(self.view_stack, stretch=1)

        # Right panels
        self.caliper_panel = CaliperPanel()
        self.caliper_panel.hide()
        self.content.addWidget(self.caliper_panel)

        self.annot_panel = AnnotationPanel()
        self.annot_panel.hide()
        self.content.addWidget(self.annot_panel)

        self.results_panel = ResultsPanel()
        self.results_panel.hide()
        self.content.addWidget(self.results_panel)

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

        nav_buttons = [
            ("|◀ Start", self._nav_start),
            ("◀◀ -1s", lambda: self._nav_step(-1.0)),
            ("◀ -0.2s", lambda: self._nav_step(-0.2)),
            ("+0.2s ▶", lambda: self._nav_step(0.2)),
            ("+1s ▶▶", lambda: self._nav_step(1.0)),
            ("Koniec ▶|", self._nav_end),
        ]
        for label, handler in nav_buttons:
            btn = QPushButton(label)
            btn.setObjectName("nav")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(handler)
            nav.addWidget(btn)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.setValue(350)
        self.scrubber.setStyleSheet(self._scrubber_style())
        self.scrubber.valueChanged.connect(self._on_scrubber)
        nav.addWidget(self.scrubber, stretch=1)

        self.time_label = QLabel()
        self.time_label.setStyleSheet(f"font-size:12px; font-family:Menlo; color:{T.TEXT_SECONDARY};")
        nav.addWidget(self.time_label)

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
            padding: 4px 10px; border-radius: 4px; font-weight: 600;
        """)
        self.view_seg._apply_styles()
        for btn in self.tool_btns:
            btn.set_active(btn._active)

        self.btn_analyze.setStyleSheet(
            f"background:{T.ACCENT};color:{T.ACCENT_TEXT};border:none;"
            f"padding:5px 14px;border-radius:5px;font-weight:600;font-size:12px;"
        )
        self.btn_mark_analysis.set_active(self._analysis_mode)
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
        self.caliper_panel.apply_theme()
        self.annot_panel.apply_theme()
        self.results_panel.apply_theme()

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
                   ground_truth: dict | None = None):
        """Load new signal data into the viewer."""
        self._ground_truth = ground_truth
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
        self._show_results = False
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

        self.file_label.setText(
            f"{filename} | {fs} Hz | {len(leads)} odpr. | {self.duration:.1f} s"
        )
        self.analysis_badge.hide()
        self.results_panel.hide()

        # Reset analysis mode
        self._analysis_mode = False
        self._analysis_start = None
        self.btn_mark_analysis.set_active(False)
        self._clear_analysis_overlay()

        # Enable/disable analysis based on duration
        min_samples = int(10.0 * self.fs)
        if self.signal.shape[0] < min_samples:
            # Too short — block analysis entirely
            self.btn_analyze.setEnabled(False)
            self.btn_analyze.setToolTip("Analiza wymaga co najmniej 10s nagrania")
            self.btn_mark_analysis.hide()
        elif self.signal.shape[0] == min_samples:
            # Exactly 10s — analyze whole file, no marking needed
            self.btn_analyze.setEnabled(True)
            self.btn_analyze.setToolTip("")
            self.btn_mark_analysis.hide()
        else:
            # Longer than 10s — require explicit window selection
            self.btn_analyze.setEnabled(True)
            self.btn_analyze.setToolTip("")
            self.btn_mark_analysis.show()

        max_window = max(self._window_12, self._window_1)
        self._scrubber_max = max(0.0, self.duration - max_window)
        self.scrubber.setRange(0, int(self._scrubber_max * 100))
        self.scrubber.setValue(0)

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
            if self._tool_mode == 1:
                self.single_lead.calipers = [
                    (1.220, 1.384, T.ACCENT, "PR = 164 ms"),
                    (1.384, 1.472, T.PURPLE, "QRS = 88 ms"),
                    (1.432, 2.264, T.GREEN, "R-R = 832 ms"),
                ]
            else:
                self.single_lead.calipers = []
            if self._tool_mode == 2:
                self.single_lead.annotations = [(2.30, 2.85)]
            else:
                self.single_lead.annotations = []
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

        self.info_panel.setVisible(idx == 0 or (idx == 0 and self._show_results))
        self.lead_sidebar.setVisible(idx == 1)
        self.monitor_sidebar.setVisible(idx == 2)

        if idx == 0:
            self.caliper_panel.hide()
            self.annot_panel.hide()
            self.info_panel.show()
            if self._show_results:
                self.results_panel.show()
            self.navbar.show()
            self._restore_scrubber_range()
        elif idx == 1:
            self.info_panel.hide()
            self._on_tool_mode(self._tool_mode)
            self.navbar.show()
            self._restore_scrubber_range()
        elif idx == 2:
            self.info_panel.hide()
            self.caliper_panel.hide()
            self.annot_panel.hide()
            self.results_panel.hide()
            self.navbar.show()
            self._start_monitor()

        self.view_stack.setCurrentIndex(idx)
        for btn in self.tool_btns:
            btn.setVisible(idx == 1)
        self._update_statusbar()

    def _on_tool_mode(self, idx: int):
        self._tool_mode = idx
        for i, btn in enumerate(self.tool_btns):
            btn.set_active(i == idx)
        self.caliper_panel.setVisible(idx == 1 and self._view_mode == 1)
        self.annot_panel.setVisible(idx == 2 and self._view_mode == 1)
        self._refresh_single_lead()
        self._update_statusbar()

    def _on_lead_selected(self, lead: str):
        self._refresh_single_lead()

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

    def _toggle_analysis_mode(self):
        """Toggle the analysis window selection mode."""
        self._analysis_mode = not self._analysis_mode
        self.btn_mark_analysis.set_active(self._analysis_mode)
        if self._analysis_mode:
            self._analysis_start = None
            self._apply_analysis_overlay()
            self._update_statusbar()
        else:
            self._analysis_start = None
            self._clear_analysis_overlay()
            self._update_statusbar()

    def _on_canvas_click(self, t: float, v: float):
        """Handle click on EKG canvas — place analysis window if in analysis mode."""
        if not self._analysis_mode or self.signal is None:
            return
        max_start = self.duration - 10.0
        if max_start <= 0:
            return
        t = max(0.0, min(t, max_start))
        self._analysis_start = t
        self._apply_analysis_overlay()
        self._update_statusbar()

    def _apply_analysis_overlay(self):
        """Set the analysis overlay on all visible canvases."""
        if self.signal is None:
            return
        clickable_end = max(0.0, self.duration - 10.0)
        region = None
        if self._analysis_start is not None:
            region = (self._analysis_start, self._analysis_start + 10.0)
        self.grid_12.set_analysis_overlay(region, clickable_end)
        self.single_lead.analysis_region = region
        self.single_lead.analysis_clickable_end = clickable_end
        self.single_lead.update()

    def _clear_analysis_overlay(self):
        """Remove analysis overlay from all canvases."""
        self.grid_12.clear_analysis_overlay()
        self.single_lead.analysis_region = None
        self.single_lead.analysis_clickable_end = None
        self.single_lead.update()

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

        if self._analysis_start is not None:
            start_sample = int(self._analysis_start * self.fs)
        else:
            start_sample = int(self.time_pos * self.fs)

        start_sample = max(0, min(start_sample, total_samples - target_samples))
        end_sample = min(start_sample + target_samples, total_samples)

        window = self.signal[start_sample:end_sample]
        t_start = start_sample / self.fs
        t_end = end_sample / self.fs
        return window, t_start, t_end

    def _on_analyze(self):
        if self.signal is None:
            return
        # For signals > 10s, require explicit window selection
        if self.duration > 10.0 and self._analysis_start is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Zaznacz okno",
                                   "Użyj 'Zaznacz do analizy' i kliknij na sygnał,\n"
                                   "aby wybrać 10-sekundowe okno.")
            return

        # Show loading state immediately
        self.results_panel.set_loading()
        self.results_panel.show()
        self.analysis_badge.hide()
        QApplication.processEvents()

        try:
            model, device = self._ensure_model_loaded()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Błąd", f"Nie udało się załadować modelu:\n{e}")
            self.results_panel.hide()
            return

        window_signal, t_start, t_end = self._get_analysis_window()

        from model.inference_api import predict_with_model
        t0 = time.time()
        try:
            result = predict_with_model(
                model=model,
                data=window_signal,
                threshold=0.5,
                class_names=TARGET_CLASSES,
                device=device,
            )
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Błąd", f"Błąd predykcji:\n{e}")
            self.results_panel.hide()
            return
        elapsed = time.time() - t0

        probs = result["probabilities"][0]
        probabilities = {cls: float(probs[i]) for i, cls in enumerate(TARGET_CLASSES)}
        model_name = os.path.basename(self._model_path)

        self._last_results = {
            "probabilities": probabilities,
            "model_name": model_name,
            "elapsed": elapsed,
        }

        window_label = f"{t_start:.1f} – {t_end:.1f} s"

        # Resolve ground truth for this specific window
        gt = self._resolve_ground_truth(t_start, t_end)

        self._show_results = True
        self.analysis_badge.show()
        self.results_panel.set_results(probabilities, model_name, elapsed, window_label,
                                       ground_truth=gt)
        if self._view_mode == 0:
            self.info_panel.show()

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
        # Reset sidebar state to match
        self.monitor_sidebar._paused = False
        self.monitor_sidebar.pause_btn.setText("⏸  Pauza")
        self.monitor_sidebar.pause_btn.setStyleSheet(
            self.monitor_sidebar._pause_btn_style(False))
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
        else:
            self._monitor_timer.stop()

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
        # Analysis mode overrides center statusbar
        if self._analysis_mode:
            if self._analysis_start is not None:
                ae = self._analysis_start + 10.0
                self.st_center.setText(
                    f"Okno: <b>{self._analysis_start:.1f} – {ae:.1f} s</b> | "
                    f"Kliknij Analizuj lub wybierz inny punkt | Esc: Wyjdź"
                )
            else:
                self.st_center.setText(
                    "Kliknij na sygnał, aby wybrać okno 10s do analizy | Esc: Wyjdź"
                )

        if self._view_mode == 0:
            self.st_left.setText("<b>10 mm/mV</b> | <b>25 mm/s</b> | 0.05-150 Hz")
            if not self._analysis_mode:
                self.st_center.setText("V: Wybierz | C: Suwmiarka | A: Adnotacja | G: Wzmocnienie | 1/2/3: Widok")
            self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b> | V = <b>0.85 mV</b>")
        elif self._view_mode == 1:
            if self._tool_mode == 1:
                self.st_left.setText("<b>Suwmiarka</b> | <b>10 mm/mV</b> | <b>25 mm/s</b>")
                self.st_center.setText("Kliknij 2 punkty | Delete: Usuń | Esc: Wyjdź")
                self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b>")
            elif self._tool_mode == 2:
                self.st_left.setText("<b>Adnotacja</b> | <b>10 mm/mV</b>")
                self.st_center.setText("Przeciągnij, aby zaznaczyć | Enter: Zapisz | Esc: Anuluj")
                self.st_right.setText("Zaznaczenie: <b>2.30 — 2.85 s</b>")
            else:
                self.st_left.setText("<b>10 mm/mV</b> | <b>25 mm/s</b>")
                self.st_center.setText("Tab: Następne | Shift+Tab: Poprzednie | ←/→: Przewiń")
                self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b> | V = <b>0.92 mV</b>")
        elif self._view_mode == 2:
            speed_label = f"{self._monitor_speed:g}x"
            self.st_left.setText(f"<b>Monitor</b> | <b>25 mm/s</b> | {speed_label}")
            self.st_center.setText("Space: Pauza | Esc: Wyjdź z monitora | ↑↓: Prędkość")
            self.st_right.setText(f"t = <b>{self._monitor_t:.2f} s</b> / {self.duration:.2f} s")
