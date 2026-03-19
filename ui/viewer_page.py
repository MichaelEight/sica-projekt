"""Main EKG viewer page with toolbar, view modes, panels, navigation bar."""
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QStackedWidget, QSlider,
                                QSizePolicy)

from ui.theme import (TOPBAR, ACCENT, BTN_DARK, WHITE, BORDER, TEXT, TEXT_MUTED,
                       TEXT_DIM, STANDARD_LEADS, GREEN, PURPLE)
from ui.ekg_canvas import (EkgCellCanvas, TwelveLeadGrid, SingleLeadCanvas,
                            generate_demo_signal, synth_ekg, LEAD_SEEDS, LEAD_AMPS)
from ui.panels import (InfoPanel, CaliperPanel, AnnotationPanel, ResultsPanel,
                        MonitorSidebar)


# ── Segmented Control ──────────────────────────
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
        for i, btn in enumerate(self.buttons):
            if i == idx:
                btn.setStyleSheet(f"""
                    background: {ACCENT}; color: white; border: none;
                    padding: 6px 12px; font-size: 12px; font-weight: 500;
                """)
            else:
                btn.setStyleSheet(f"""
                    background: {BTN_DARK}; color: #888; border: none;
                    padding: 6px 12px; font-size: 12px; font-weight: 500;
                """)
        self.changed.emit(idx)

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
            self.setStyleSheet(f"background:{ACCENT};color:white;border:none;padding:6px 10px;border-radius:5px;")
        else:
            self.setStyleSheet(f"background:{BTN_DARK};color:#ccc;border:none;padding:6px 10px;border-radius:5px;")

    def is_active(self) -> bool:
        return self._active


# ── Lead Sidebar (for 1-lead mode) ─────────────
class LeadSidebar(QWidget):
    """Sidebar with lead selection buttons."""

    lead_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(58)
        self.setStyleSheet(f"background: {WHITE}; border-right: 1px solid {BORDER};")

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
                sep.setStyleSheet(f"background: {BORDER};")
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
        for lead, btn in self.buttons.items():
            if lead == self._active:
                btn.setStyleSheet(f"""
                    background: {ACCENT}; color: white; border: 1px solid {ACCENT};
                    border-radius: 4px; font-family: Menlo;
                """)
            else:
                btn.setStyleSheet(f"""
                    background: {WHITE}; color: {TEXT_DIM}; border: 1px solid {BORDER};
                    border-radius: 4px; font-family: Menlo;
                """)

    def active_lead(self) -> str:
        return self._active


# ── Separator ──────────────────────────────────
def _sep():
    sep = QFrame()
    sep.setFixedSize(1, 24)
    sep.setStyleSheet("background: #444;")
    return sep


# ── Viewer Page ────────────────────────────────
class ViewerPage(QWidget):
    """Main EKG viewer with toolbar, view modes, panels, navigation."""

    open_file = Signal()         # request to open a new file
    show_report = Signal()       # request to show report page

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal = None
        self.leads = STANDARD_LEADS
        self.fs = 500
        self.filename = ""
        self.duration = 10.0
        self.time_pos = 3.5       # current time position
        self._view_mode = 0       # 0=12lead, 1=1lead, 2=monitor
        self._tool_mode = 0       # 0=select, 1=caliper, 2=annotation
        self._show_results = False
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(50)
        self._monitor_timer.timeout.connect(self._monitor_tick)
        self._monitor_playing = False
        self._monitor_t = 0.0

        self._build_ui()

    # ── Build UI ──
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top toolbar ──
        self.toolbar = QWidget()
        self.toolbar.setFixedHeight(48)
        self.toolbar.setStyleSheet(f"background: {TOPBAR};")
        tb = QHBoxLayout(self.toolbar)
        tb.setContentsMargins(14, 0, 14, 0)
        tb.setSpacing(8)

        # Logo
        logo = QLabel()
        logo.setText('<span style="color:#4a9eff;font-weight:600;">EKG</span>'
                     ' <span style="color:white;font-weight:600;">Assistant</span>')
        logo.setFont(QFont(".AppleSystemUIFont", 14))
        logo.setTextFormat(Qt.RichText)
        tb.addWidget(logo)
        tb.addWidget(_sep())

        # File info
        self.file_label = QLabel("00888_lr.dat | 500 Hz | 12 odpr. | 10.0 s")
        self.file_label.setStyleSheet("font-size:12px; color:#ccc; font-family:Menlo;")
        tb.addWidget(self.file_label)
        tb.addStretch()

        # Analysis badge (shown after AI analysis)
        self.analysis_badge = QLabel("Analiza zako\u0144czona")
        self.analysis_badge.setStyleSheet(f"""
            font-size: 12px; background: #d1fae5; color: #065f46;
            padding: 5px 12px; border-radius: 5px; font-weight: 600;
        """)
        self.analysis_badge.hide()
        tb.addWidget(self.analysis_badge)

        # View mode segmented control
        self.view_seg = SegmentedControl(["12-Lead", "1-Lead", "Monitor"])
        self.view_seg.changed.connect(self._on_view_mode)
        tb.addWidget(self.view_seg)
        tb.addWidget(_sep())

        # Tools
        self.tool_btns: list[ToolbarBtn] = []
        for i, (label, active) in enumerate([("Wybierz", True), ("Suwmiarka", False), ("Adnotacja", False)]):
            btn = ToolbarBtn(label, active)
            btn.clicked.connect(lambda checked, idx=i: self._on_tool_mode(idx))
            self.tool_btns.append(btn)
            tb.addWidget(btn)
        tb.addWidget(_sep())

        # Settings
        for label in ["10 mm/mV", "25 mm/s", "0.05-150 Hz"]:
            btn = ToolbarBtn(label, False)
            tb.addWidget(btn)
        tb.addWidget(_sep())

        # Actions
        self.btn_analyze = QPushButton("Analizuj")
        self.btn_analyze.setStyleSheet(f"""
            background:{ACCENT};color:white;font-size:12px;padding:6px 14px;
            border-radius:5px;border:none;font-weight:600;
        """)
        self.btn_analyze.setCursor(Qt.PointingHandCursor)
        self.btn_analyze.clicked.connect(self._on_analyze)
        tb.addWidget(self.btn_analyze)

        self.btn_report = QPushButton("Raport")
        self.btn_report.setStyleSheet(f"""
            background:{BTN_DARK};color:#ccc;font-size:12px;padding:6px 14px;
            border-radius:5px;border:none;
        """)
        self.btn_report.setCursor(Qt.PointingHandCursor)
        self.btn_report.clicked.connect(self.show_report.emit)
        tb.addWidget(self.btn_report)

        btn_load = QPushButton("Wczytaj plik")
        btn_load.setStyleSheet(f"""
            background:{BTN_DARK};color:#ccc;font-size:12px;padding:6px 14px;
            border-radius:5px;border:none;
        """)
        btn_load.setCursor(Qt.PointingHandCursor)
        btn_load.clicked.connect(self.open_file.emit)
        tb.addWidget(btn_load)

        outer.addWidget(self.toolbar)

        # ── Content area ──
        self.content = QHBoxLayout()
        self.content.setContentsMargins(0, 0, 0, 0)
        self.content.setSpacing(0)

        # Left panels
        self.info_panel = InfoPanel()
        self.content.addWidget(self.info_panel)

        self.monitor_sidebar = MonitorSidebar()
        self.monitor_sidebar.hide()
        self.content.addWidget(self.monitor_sidebar)

        self.lead_sidebar = LeadSidebar()
        self.lead_sidebar.lead_selected.connect(self._on_lead_selected)
        self.lead_sidebar.hide()
        self.content.addWidget(self.lead_sidebar)

        # Center: stacked views
        self.view_stack = QStackedWidget()
        self.view_stack.setStyleSheet(f"background: #f9fafb;")

        self.grid_12 = TwelveLeadGrid()
        self.view_stack.addWidget(self.grid_12)  # idx 0

        self.single_lead = SingleLeadCanvas()
        self.single_lead.draw_border = True
        self.view_stack.addWidget(self.single_lead)  # idx 1

        self.monitor_area = QWidget()
        self.monitor_area.setStyleSheet(f"background: #f9fafb;")
        self._monitor_strips: list[EkgCellCanvas] = []
        self._build_monitor_area()
        self.view_stack.addWidget(self.monitor_area)  # idx 2

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

        # ── Navigation bar ──
        self.navbar = QWidget()
        self.navbar.setFixedHeight(48)
        self.navbar.setStyleSheet(f"background:{WHITE}; border-top:1px solid {BORDER};")
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

        # Scrubber
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.setValue(350)
        self.scrubber.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px; background: {BORDER}; border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 14px; height: 14px; margin: -4px 0;
                background: {ACCENT}; border: 2px solid white;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT}; border-radius: 3px;
            }}
        """)
        self.scrubber.valueChanged.connect(self._on_scrubber)
        nav.addWidget(self.scrubber, stretch=1)

        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-size:12px; font-family:Menlo; color:#4b5563;")
        nav.addWidget(self.time_label)

        self.speed_label = QLabel("25 mm/s")
        self.speed_label.setStyleSheet(f"font-size:11px; color:{TEXT_DIM};")
        nav.addWidget(self.speed_label)

        outer.addWidget(self.navbar)

        # ── Status bar ──
        self.statusbar = QWidget()
        self.statusbar.setFixedHeight(28)
        self.statusbar.setStyleSheet(f"background:{WHITE}; border-top:1px solid {BORDER};")
        sb = QHBoxLayout(self.statusbar)
        sb.setContentsMargins(14, 0, 14, 0)
        sb.setSpacing(16)

        self.st_left = QLabel()
        self.st_left.setStyleSheet(f"font-size:11px; color:{TEXT_MUTED}; font-family:Menlo;")
        sb.addWidget(self.st_left)

        self.st_center = QLabel()
        self.st_center.setStyleSheet(f"font-size:10px; color:{TEXT_DIM};")
        self.st_center.setAlignment(Qt.AlignCenter)
        sb.addWidget(self.st_center, stretch=1)

        self.st_right = QLabel()
        self.st_right.setStyleSheet(f"font-size:11px; color:{TEXT_MUTED}; font-family:Menlo;")
        sb.addWidget(self.st_right)

        outer.addWidget(self.statusbar)

        self._update_time_display()
        self._update_statusbar()

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

    # ── Public: set signal data ──
    def set_signal(self, signal: np.ndarray, leads: list[str], fs: int, filename: str = ""):
        self.signal = signal
        self.leads = leads
        self.fs = fs
        self.filename = filename
        self.duration = signal.shape[0] / fs
        self.time_pos = 0.0
        self._show_results = False

        self.file_label.setText(
            f"{filename} | {fs} Hz | {len(leads)} odpr. | {self.duration:.1f} s"
        )
        self.analysis_badge.hide()
        self.results_panel.hide()

        self.scrubber.setRange(0, int(self.duration * 100))
        self.scrubber.setValue(0)

        self._refresh_views()
        self._update_time_display()
        self._update_statusbar()

    def _refresh_views(self):
        if self.signal is None:
            return
        # 12-lead grid
        self.grid_12.set_signal(self.signal, self.leads, self.fs)

        # Single lead
        self._refresh_single_lead()

        # Monitor
        self._refresh_monitor()

    def _refresh_single_lead(self):
        if self.signal is None:
            return
        lead = self.lead_sidebar.active_lead()
        if lead in self.leads:
            idx = self.leads.index(lead)
            self.single_lead.set_data(lead, self.signal[:, idx], self.fs, 0, self.duration)
            # Add demo calipers when in caliper mode
            if self._tool_mode == 1:
                self.single_lead.calipers = [
                    (1.220, 1.384, ACCENT, "PR = 164 ms"),
                    (1.384, 1.472, PURPLE, "QRS = 88 ms"),
                    (1.432, 2.264, GREEN, "R-R = 832 ms"),
                ]
            else:
                self.single_lead.calipers = []
            # Add demo annotation when in annotation mode
            if self._tool_mode == 2:
                self.single_lead.annotations = [(2.30, 2.85)]
            else:
                self.single_lead.annotations = []
            self.single_lead.update()

    def _refresh_monitor(self):
        if self.signal is None:
            return
        for lead_name, strip in self._monitor_strips:
            if lead_name in self.leads:
                idx = self.leads.index(lead_name)
                strip.set_data(lead_name, self.signal[:, idx], self.fs, 0, self.duration)
                strip.set_sweep(self._monitor_t / self.duration if self.duration > 0 else 0)

    # ── View mode switching ──
    def _on_view_mode(self, idx: int):
        self._view_mode = idx
        # Stop monitor if leaving monitor mode
        if idx != 2:
            self._monitor_timer.stop()
            self._monitor_playing = False

        # Show/hide panels based on mode
        self.info_panel.setVisible(idx == 0 or (idx == 0 and self._show_results))
        self.lead_sidebar.setVisible(idx == 1)
        self.monitor_sidebar.setVisible(idx == 2)

        # Hide tool panels in 12-lead and monitor modes
        if idx == 0:
            self.caliper_panel.hide()
            self.annot_panel.hide()
            self.info_panel.show()
            if self._show_results:
                self.results_panel.show()
            self.navbar.show()
        elif idx == 1:
            self.info_panel.hide()
            self._on_tool_mode(self._tool_mode)
            self.navbar.hide()
        elif idx == 2:
            self.info_panel.hide()
            self.caliper_panel.hide()
            self.annot_panel.hide()
            self.results_panel.hide()
            self.navbar.show()
            self._start_monitor()

        self.view_stack.setCurrentIndex(idx)

        # Hide tool buttons in non-1-lead modes
        for btn in self.tool_btns:
            btn.setVisible(idx == 1)

        self._update_statusbar()

    # ── Tool mode switching ──
    def _on_tool_mode(self, idx: int):
        self._tool_mode = idx
        for i, btn in enumerate(self.tool_btns):
            btn.set_active(i == idx)

        self.caliper_panel.setVisible(idx == 1 and self._view_mode == 1)
        self.annot_panel.setVisible(idx == 2 and self._view_mode == 1)

        self._refresh_single_lead()
        self._update_statusbar()

    # ── Lead selection ──
    def _on_lead_selected(self, lead: str):
        self._refresh_single_lead()

    # ── Analyze (mock) ──
    def _on_analyze(self):
        self._show_results = True
        self.analysis_badge.show()
        self.results_panel.show()
        if self._view_mode == 0:
            self.info_panel.show()

    # ── Navigation ──
    def _nav_start(self):
        self.time_pos = 0
        self.scrubber.setValue(0)

    def _nav_end(self):
        self.time_pos = self.duration
        self.scrubber.setValue(self.scrubber.maximum())

    def _nav_step(self, dt: float):
        self.time_pos = max(0, min(self.duration, self.time_pos + dt))
        self.scrubber.setValue(int(self.time_pos * 100))

    def _on_scrubber(self, value: int):
        self.time_pos = value / 100.0
        self._update_time_display()

    def _update_time_display(self):
        self.time_label.setText(f"{self.time_pos:.2f} s / {self.duration:.2f} s")

    # ── Monitor ──
    def _start_monitor(self):
        self._monitor_t = 0.0
        self._monitor_playing = True
        self._monitor_timer.start()
        self._refresh_monitor()

    def _monitor_tick(self):
        if not self._monitor_playing:
            return
        self._monitor_t += 0.05
        if self._monitor_t > self.duration:
            self._monitor_t = 0.0
        for lead_name, strip in self._monitor_strips:
            strip.set_sweep(self._monitor_t / self.duration if self.duration > 0 else 0)
        # Update scrubber
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(int(self._monitor_t * 100))
        self.scrubber.blockSignals(False)
        self.time_pos = self._monitor_t
        self._update_time_display()

    # ── Status bar ──
    def _update_statusbar(self):
        mode_names = ["12-Lead", "1-Lead", "Monitor"]
        tool_names = ["Wybierz", "Suwmiarka", "Adnotacja"]

        if self._view_mode == 0:
            self.st_left.setText("<b>10 mm/mV</b> | <b>25 mm/s</b> | 0.05-150 Hz")
            self.st_center.setText("V: Wybierz | C: Suwmiarka | A: Adnotacja | G: Wzmocnienie | 1/2/3: Widok")
            self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b> | V = <b>0.85 mV</b>")
        elif self._view_mode == 1:
            if self._tool_mode == 1:
                self.st_left.setText("<b>Suwmiarka</b> | <b>10 mm/mV</b> | <b>25 mm/s</b>")
                self.st_center.setText("Kliknij 2 punkty | Delete: Usu\u0144 | Esc: Wyjd\u017a")
                self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b>")
            elif self._tool_mode == 2:
                self.st_left.setText("<b>Adnotacja</b> | <b>10 mm/mV</b>")
                self.st_center.setText("Przeci\u0105gnij, aby zaznaczy\u0107 | Enter: Zapisz | Esc: Anuluj")
                self.st_right.setText("Zaznaczenie: <b>2.30 \u2014 2.85 s</b>")
            else:
                self.st_left.setText("<b>10 mm/mV</b> | <b>25 mm/s</b>")
                self.st_center.setText("Tab: Nast\u0119pne | Shift+Tab: Poprzednie | \u2190/\u2192: Przewi\u0144")
                self.st_right.setText(f"t = <b>{self.time_pos:.2f} s</b> | V = <b>0.92 mV</b>")
        elif self._view_mode == 2:
            self.st_left.setText("<b>Monitor</b> | <b>25 mm/s</b> | 1x")
            self.st_center.setText("Space: Pauza | Esc: Wyjd\u017a z monitora | \u2191\u2193: Pr\u0119dko\u015b\u0107")
            self.st_right.setText(f"t = <b>{self._monitor_t:.2f} s</b>")
