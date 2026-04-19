"""Report preview page matching v2 08-report design."""
import numpy as np
from datetime import datetime
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QPainterPath
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QScrollArea, QComboBox,
                                QFileDialog, QGridLayout)
from PySide6.QtPrintSupport import QPrinter, QPrintDialog

import ui.theme as T
from ui.theme import LEAD_SEEDS, LEAD_AMPS
from ui.ekg_canvas import synth_ekg


class EkgPreviewWidget(QWidget):
    """12-lead grid + rhythm strip preview for the report."""

    def __init__(self, signal=None, leads=None, fs=500, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.leads = leads or []
        self.fs = fs
        self.setFixedHeight(360)

    def _voltage_range(self):
        if self.signal is None or self.signal.size == 0:
            return -1.5, 1.5
        mn = float(self.signal.min())
        mx = float(self.signal.max())
        pad = max((mx - mn) * 0.1, 0.2)
        return mn - pad, mx + pad

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(T.WHITE))

        grid = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
                ["III", "aVF", "V3", "V6"]]
        rhythm_h = 110
        grid_h = h - rhythm_h
        row_h = grid_h / 3
        col_w = w / 4
        v_min, v_max = self._voltage_range()
        v_range = max(v_max - v_min, 0.1)

        sq = min(col_w / 30, row_h / 12)
        p.setPen(QPen(QColor(T.GRID_MINOR), 0.3))
        x = 0.0
        while x < w:
            p.drawLine(int(x), 0, int(x), h)
            x += sq
        y = 0.0
        while y < h:
            p.drawLine(0, int(y), w, int(y))
            y += sq
        p.setPen(QPen(QColor(T.GRID_MAJOR), 0.7))
        x = 0.0
        while x < w:
            p.drawLine(int(x), 0, int(x), h)
            x += sq * 5
        y = 0.0
        while y < h:
            p.drawLine(0, int(y), w, int(y))
            y += sq * 5

        p.setPen(QPen(QColor(T.BORDER), 1))
        for c in range(1, 4):
            p.drawLine(int(c * col_w), 0, int(c * col_w), int(grid_h))
        for r in range(1, 3):
            p.drawLine(0, int(r * row_h), w, int(r * row_h))
        p.drawLine(0, int(grid_h), w, int(grid_h))

        sig_pen = QPen(QColor(T.SIGNAL_COLOR), 1.4)
        for r_i, row_leads in enumerate(grid):
            for c_i, lead in enumerate(row_leads):
                x_off = c_i * col_w
                y_top = r_i * row_h + 4
                y_bot = (r_i + 1) * row_h - 4
                cell_h = y_bot - y_top
                t_start = c_i * 2.5
                t_end = t_start + 2.5

                p.setPen(sig_pen)
                path = QPainterPath()
                if self.signal is not None and lead in self.leads:
                    lead_idx = self.leads.index(lead)
                    n_start = int(t_start * self.fs)
                    n_end = min(int(t_end * self.fs), self.signal.shape[0])
                    sig = self.signal[n_start:n_end, lead_idx]
                    n = len(sig)
                    for px_i in range(int(col_w)):
                        if n == 0:
                            break
                        idx = min(int(px_i / col_w * n), n - 1)
                        v = float(sig[idx])
                        py = y_bot - (v - v_min) / v_range * cell_h
                        if px_i == 0:
                            path.moveTo(x_off + px_i, py)
                        else:
                            path.lineTo(x_off + px_i, py)
                else:
                    seed = LEAD_SEEDS.get(lead, 0)
                    amp = LEAD_AMPS.get(lead, 1)
                    t = np.linspace(t_start, t_end, int(col_w))
                    vals = synth_ekg(t, seed, amp)
                    for px_i in range(len(vals)):
                        v = float(vals[px_i])
                        py = y_bot - (v - v_min) / v_range * cell_h
                        if px_i == 0:
                            path.moveTo(x_off + px_i, py)
                        else:
                            path.lineTo(x_off + px_i, py)
                p.drawPath(path)

                p.setFont(QFont(".AppleSystemUIFont", 10, QFont.DemiBold))
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(lead) + 10
                th = fm.height() + 2
                bg = QColor(T.WHITE)
                bg.setAlpha(220)
                p.fillRect(QRectF(x_off + 4, r_i * row_h + 4, tw, th), bg)
                p.setPen(QColor(T.TEXT))
                p.drawText(int(x_off + 9), int(r_i * row_h + 4 + fm.ascent() + 1), lead)

        ry_top = grid_h + 6
        ry_bot = h - 6
        rhythm_h_actual = ry_bot - ry_top
        p.setPen(sig_pen)
        path = QPainterPath()
        if self.signal is not None and "II" in self.leads:
            idx = self.leads.index("II")
            sig = self.signal[:, idx]
            n = len(sig)
            for px_i in range(w):
                si = min(int(px_i / w * n), n - 1)
                v = float(sig[si])
                py = ry_bot - (v - v_min) / v_range * rhythm_h_actual
                if px_i == 0:
                    path.moveTo(px_i, py)
                else:
                    path.lineTo(px_i, py)
        else:
            t = np.linspace(0, 10, w)
            vals = synth_ekg(t, 0.5, 1.0)
            for px_i in range(len(vals)):
                v = float(vals[px_i])
                py = ry_bot - (v - v_min) / v_range * rhythm_h_actual
                if px_i == 0:
                    path.moveTo(px_i, py)
                else:
                    path.lineTo(px_i, py)
        p.drawPath(path)

        p.setFont(QFont(".AppleSystemUIFont", 10, QFont.DemiBold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance("II (rytm)") + 10
        th = fm.height() + 2
        bg = QColor(T.WHITE)
        bg.setAlpha(220)
        p.fillRect(QRectF(4, ry_top, tw, th), bg)
        p.setPen(QColor(T.TEXT))
        p.drawText(9, int(ry_top + fm.ascent() + 1), "II (rytm)")
        p.end()


class ReportPage(QWidget):
    """Report preview page with export options."""

    go_back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal = None
        self.leads = []
        self.fs = 500
        self._patient_sex = ""
        self._last_measurements = None
        self._last_annotations = None
        self._last_scan = None
        self._last_results = None
        self._build_ui()
        self.apply_theme()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top bar ──
        self._topbar = QWidget()
        self._topbar.setFixedHeight(48)
        tb = QHBoxLayout(self._topbar)
        tb.setContentsMargins(14, 0, 14, 0)
        tb.setSpacing(8)

        self._logo = QLabel()
        self._logo.setFont(QFont(".AppleSystemUIFont", 14))
        self._logo.setTextFormat(Qt.RichText)
        tb.addWidget(self._logo)

        self._topbar_sep = QFrame()
        self._topbar_sep.setFixedSize(1, 24)
        tb.addWidget(self._topbar_sep)

        self.file_info = QLabel("00888_lr.dat")
        tb.addWidget(self.file_info)
        tb.addStretch()

        self._badge = QLabel("Podgl\u0105d raportu")
        tb.addWidget(self._badge)

        self.btn_back = QPushButton("\u2190  Powr\u00f3t do widoku")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self.go_back.emit)
        tb.addWidget(self.btn_back)
        outer.addWidget(self._topbar)

        # ── Content (scrollable report) ──
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignCenter)

        self.report = QWidget()
        self.report.setObjectName("reportPaper")
        self.report.setFixedWidth(780)
        r_layout = QVBoxLayout(self.report)
        r_layout.setContentsMargins(28, 28, 28, 28)
        r_layout.setSpacing(8)

        self._title = QLabel("Kardioskop \u2014 Raport badania")
        self._title.setFont(QFont(".AppleSystemUIFont", 16, QFont.DemiBold))
        self._title.setAlignment(Qt.AlignCenter)
        r_layout.addWidget(self._title)

        self._date_label = QLabel("Wygenerowano: — | Plik: —")
        self._date_label.setAlignment(Qt.AlignCenter)
        r_layout.addWidget(self._date_label)

        self._title_line = QFrame()
        self._title_line.setFrameShape(QFrame.HLine)
        r_layout.addWidget(self._title_line)

        # Patient info grid
        pgrid = QWidget()
        pg = QGridLayout(pgrid)
        pg.setSpacing(8)
        self._patient_value_labels = {}
        self._patient_name_labels = []
        patient_fields = [
            ("ID pacjenta", "patient_id", "Data badania", "date"),
            ("Wiek", "age", "Czas trwania", "duration"),
            ("Płeć", "sex", "Częstotliwość", "fs"),
        ]
        for row_i, (l1, k1, l2, k2) in enumerate(patient_fields):
            for col_i, (label, key) in enumerate([(l1, k1), (l2, k2)]):
                lbl = QLabel(label)
                val = QLabel("—")
                pg.addWidget(lbl, row_i, col_i * 2)
                pg.addWidget(val, row_i, col_i * 2 + 1)
                self._patient_name_labels.append(lbl)
                self._patient_value_labels[key] = val
        r_layout.addWidget(pgrid)

        self._sep_line = QFrame()
        self._sep_line.setFrameShape(QFrame.HLine)
        r_layout.addWidget(self._sep_line)

        self.ecg_preview = EkgPreviewWidget()
        self.ecg_preview.hide()

        # Measurements table
        self._sec_meas = QLabel("POMIARY")
        r_layout.addWidget(self._sec_meas)

        table = QWidget()
        t_layout = QVBoxLayout(table)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(0)
        self._header_row = QWidget()
        hr_layout = QHBoxLayout(self._header_row)
        hr_layout.setContentsMargins(10, 6, 10, 6)
        self._header_labels = []
        for text, w_pct in [("Parametr", 150), ("Warto\u015b\u0107", 100), ("Norma", 130), ("Status", 80)]:
            lbl = QLabel(text)
            lbl.setFixedWidth(w_pct)
            hr_layout.addWidget(lbl)
            self._header_labels.append(lbl)
        t_layout.addWidget(self._header_row)

        self._meas_rows = {}
        self._meas_table_layout = t_layout
        default_measurements = [
            ("HR", "—", "60-100 bpm", "—"),
            ("PR interval", "—", "120-200 ms", "—"),
            ("QRS", "—", "<120 ms", "—"),
            ("QT", "—", "zależny od HR", "—"),
            ("QTc (Bazett)", "—", "<450 ms", "—"),
            ("Oś", "—", "-30° do +90°", "—"),
        ]
        for param, val, norm, status in default_measurements:
            row = self._create_meas_row(param, val, norm, status)
            t_layout.addWidget(row)
        r_layout.addWidget(table)

        # AI Analysis
        self._sec_ai = QLabel("ANALIZA AI")
        r_layout.addWidget(self._sec_ai)

        self._ai_box = QFrame()
        self._ai_box.setObjectName("aiBox")
        ai_layout = QVBoxLayout(self._ai_box)
        ai_layout.setContentsMargins(16, 14, 16, 14)
        ai_layout.setSpacing(10)
        self._ai_diag = QLabel("Brak analizy")
        self._ai_diag.setWordWrap(True)
        ai_layout.addWidget(self._ai_diag)
        self._ai_model = QLabel("")
        ai_layout.addWidget(self._ai_model)

        self._ai_per_class = QWidget()
        self._ai_per_class.setStyleSheet("background: transparent;")
        self._ai_per_class_layout = QVBoxLayout(self._ai_per_class)
        self._ai_per_class_layout.setContentsMargins(0, 4, 0, 4)
        self._ai_per_class_layout.setSpacing(0)
        ai_layout.addWidget(self._ai_per_class)

        self._ai_healthy = QLabel("")
        self._ai_healthy.setWordWrap(True)
        ai_layout.addWidget(self._ai_healthy)
        r_layout.addWidget(self._ai_box)

        # Annotations
        self._ann_header = QLabel("ADNOTACJE (0)")
        r_layout.addWidget(self._ann_header)

        self._ann_container = QWidget()
        self._ann_layout = QVBoxLayout(self._ann_container)
        self._ann_layout.setContentsMargins(0, 0, 0, 0)
        self._ann_layout.setSpacing(0)
        self._ann_empty_label = QLabel("Brak adnotacji")
        self._ann_layout.addWidget(self._ann_empty_label)
        r_layout.addWidget(self._ann_container)

        self._disc = QLabel(
            "Wynik analizy AI ma charakter pomocniczy i nie stanowi diagnozy medycznej.\n"
            "Ostateczna decyzja diagnostyczna nale\u017cy do lekarza specjalisty."
        )
        self._disc.setAlignment(Qt.AlignCenter)
        self._disc.setWordWrap(True)
        r_layout.addWidget(self._disc)

        r_layout.addStretch()

        self._scroll_content = QWidget()
        sc_layout = QVBoxLayout(self._scroll_content)
        sc_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        sc_layout.setContentsMargins(12, 12, 12, 20)
        sc_layout.addWidget(self.report)
        self._scroll.setWidget(self._scroll_content)
        outer.addWidget(self._scroll, stretch=1)

        # ── Export bar ──
        self._export_bar = QWidget()
        self._export_bar.setFixedHeight(48)
        eb = QHBoxLayout(self._export_bar)
        eb.setContentsMargins(20, 0, 20, 0)
        eb.setSpacing(12)
        eb.setAlignment(Qt.AlignCenter)

        self._btn_pdf = QPushButton("Eksportuj PDF")
        self._btn_pdf.setCursor(Qt.PointingHandCursor)
        self._btn_pdf.clicked.connect(self._export_pdf)
        eb.addWidget(self._btn_pdf)

        self._btn_png = QPushButton("Eksportuj PNG")
        self._btn_png.setCursor(Qt.PointingHandCursor)
        self._btn_png.clicked.connect(self._export_png)
        eb.addWidget(self._btn_png)

        self._btn_print = QPushButton("Drukuj")
        self._btn_print.setCursor(Qt.PointingHandCursor)
        self._btn_print.clicked.connect(self._print)
        eb.addWidget(self._btn_print)

        eb.addStretch()
        self._page_lbl = QLabel("Format:")
        eb.addWidget(self._page_lbl)
        self._page_sel = QComboBox()
        self._page_sel.addItems(["A4", "Letter"])
        self._page_sel.setFixedWidth(90)
        self._page_sel.setCursor(Qt.PointingHandCursor)
        eb.addWidget(self._page_sel)

        outer.addWidget(self._export_bar)

    def apply_theme(self):
        """Re-apply all theme-dependent styles after dark/light toggle."""
        self._topbar.setStyleSheet(f"background: {T.TOPBAR};")
        self._logo.setText(
            f'<span style="color:#4a9eff;font-weight:700;">Kardio</span>'
            f'<span style="color:white;font-weight:700;">skop</span>'
        )
        self._topbar_sep.setStyleSheet(f"background: {T.SEPARATOR};")
        self.file_info.setStyleSheet(
            f"font-size:12px; color:{T.BTN_TEXT}; font-family:Menlo; background: transparent;"
        )
        self._badge.setStyleSheet(
            f"font-size: 12px; color: {T.ACCENT}; font-weight: 600;"
            f" background: transparent; padding: 4px 8px;"
        )
        pressed_bg = '#047857' if T.is_dark_mode() else '#2563eb'
        self.btn_back.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {T.BTN_TEXT};
                font-size: 12px; font-weight: 500;
                padding: 6px 14px; border-radius: 6px;
                border: 1px solid {T.SEPARATOR};
            }}
            QPushButton:hover {{
                background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                border: 1px solid {T.ACCENT};
            }}
            QPushButton:pressed {{
                background: {pressed_bg};
                border: 1px solid {pressed_bg};
            }}
        """)

        self._scroll.setStyleSheet(f"QScrollArea {{ background: {T.BG}; border: none; }}")
        self._scroll_content.setStyleSheet(f"background: {T.BG};")
        self.report.setStyleSheet(f"""
            QWidget#reportPaper {{
                background: {T.WHITE};
                border: 1px solid {T.BORDER};
                border-radius: 8px;
            }}
        """)
        self._title.setStyleSheet(f"color: {T.TEXT}; background: transparent;")
        self._date_label.setStyleSheet(
            f"font-size: 12px; color: {T.TEXT_MUTED}; background: transparent;"
        )
        self._title_line.setStyleSheet(
            f"background: {T.TEXT}; border: none; max-height: 2px;"
        )
        for lbl in self._patient_name_labels:
            lbl.setStyleSheet(
                f"font-size: 12px; color: {T.TEXT_MUTED};"
                f" min-width: 110px; background: transparent;"
            )
        for val in self._patient_value_labels.values():
            val.setStyleSheet(
                f"font-size: 13px; font-weight: 600; font-family: Menlo;"
                f" color: {T.TEXT}; background: transparent;"
            )
        self._sep_line.setStyleSheet(
            f"background: {T.BORDER}; border: none; max-height: 1px;"
        )

        section_style = (
            f"font-size: 13px; font-weight: 700; letter-spacing: 0.5px;"
            f" margin-top: 10px; color: {T.TEXT}; background: transparent;"
        )
        self._sec_meas.setStyleSheet(section_style)
        self._sec_ai.setStyleSheet(section_style)
        self._ann_header.setStyleSheet(section_style)

        self._header_row.setStyleSheet(
            f"border-bottom: 1px solid {T.BORDER}; background: transparent;"
        )
        for lbl in self._header_labels:
            lbl.setStyleSheet(
                f"font-size: 12px; color: {T.TEXT_MUTED};"
                f" font-weight: 500; background: transparent;"
            )

        self._ai_box.setStyleSheet(f"""
            QFrame#aiBox {{
                background: {T.AMBER_BG};
                border: 1px solid {T.AMBER_BORDER};
                border-radius: 8px;
            }}
        """)
        self._ai_diag.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {T.AMBER_TEXT};"
            f" background: transparent;"
        )
        self._ai_model.setStyleSheet(
            f"font-size: 11px; color: {T.TEXT_MUTED}; background: transparent;"
        )
        self._ai_healthy.setStyleSheet(
            f"font-size: 11px; color: {T.AMBER_SUB}; font-family: Menlo;"
            f" padding-top: 8px; background: transparent;"
            f" border-top: 1px solid {T.AMBER_BORDER};"
        )

        self._ann_empty_label.setStyleSheet(
            f"font-size: 12px; color: {T.TEXT_MUTED};"
            f" padding: 6px 0; background: transparent;"
        )

        self._disc.setStyleSheet(f"""
            font-size: 11px; color: {T.TEXT_DIM};
            margin-top: 16px; padding-top: 12px;
            border-top: 1px solid {T.BORDER};
            background: transparent;
        """)

        self._export_bar.setStyleSheet(
            f"background: {T.WHITE}; border-top: 1px solid {T.BORDER};"
        )
        self._btn_pdf.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 20px; border-radius: 6px;
                font-size: 13px; font-weight: 500;
                background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                border: 1px solid {T.ACCENT};
            }}
        """)
        secondary_btn = f"""
            QPushButton {{
                padding: 8px 20px; border-radius: 6px;
                font-size: 13px; font-weight: 500;
                background: {T.WHITE}; color: {T.TEXT_SECONDARY};
                border: 1px solid {T.BORDER};
            }}
        """
        self._btn_png.setStyleSheet(secondary_btn)
        self._btn_print.setStyleSheet(secondary_btn)
        self._page_lbl.setStyleSheet(
            f"font-size: 12px; color: {T.TEXT_MUTED}; background: transparent;"
        )
        self._page_sel.setStyleSheet(f"""
            QComboBox {{
                font-size: 12px; padding: 6px 10px;
                border: 1px solid {T.BORDER}; border-radius: 6px;
                background: {T.WHITE}; color: {T.TEXT};
            }}
            QComboBox::drop-down {{
                border: none; width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background: {T.WHITE}; color: {T.TEXT};
                selection-background-color: {T.ACCENT};
                selection-color: {T.ACCENT_TEXT};
                border: 1px solid {T.BORDER};
            }}
        """)

        # Re-render dynamic content so row colors match current theme
        if self._last_measurements is not None:
            self.set_measurements(self._last_measurements)
        else:
            self._rebuild_default_meas_rows()
        if self._last_scan is not None:
            scan, model_name = self._last_scan
            self.set_scan_summary(scan, model_name=model_name)
        if self._last_results is not None:
            probs, model_name, elapsed = self._last_results
            self.set_results(probs, model_name, elapsed)
        if self._last_annotations is not None:
            self.set_annotations(self._last_annotations)

    def _rebuild_default_meas_rows(self):
        while self._meas_table_layout.count() > 1:
            item = self._meas_table_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        defaults = [
            ("HR", "—", "60-100 bpm", "—"),
            ("PR interval", "—", "120-200 ms", "—"),
            ("QRS", "—", "<120 ms", "—"),
            ("QT", "—", "zależny od HR", "—"),
            ("QTc (Bazett)", "—", "<450 ms", "—"),
            ("Oś", "—", "-30° do +90°", "—"),
        ]
        for param, val, norm, status in defaults:
            self._meas_table_layout.addWidget(
                self._create_meas_row(param, val, norm, status)
            )

    def set_signal(self, signal, leads, fs, filename=""):
        self.signal = signal
        self.leads = leads
        self.fs = fs
        self.file_info.setText(filename)
        self.ecg_preview.signal = signal
        self.ecg_preview.leads = leads
        self.ecg_preview.fs = fs
        self.ecg_preview.update()

        now = datetime.now().strftime("%d.%m.%Y, %H:%M")
        self._date_label.setText(f"Wygenerowano: {now} | Plik: {filename or '—'}")

    def set_patient_info(self, patient_id="", age="", sex="", date="", duration="", fs=""):
        sex_display = {"M": "Mężczyzna", "K": "Kobieta"}.get(str(sex), str(sex) if sex else "—")
        values = {
            "patient_id": str(patient_id) if patient_id else "—",
            "age": f"{age} lat" if age else "—",
            "sex": sex_display,
            "date": str(date) if date else "—",
            "duration": f"{duration} s" if duration else "—",
            "fs": f"{fs} Hz" if fs else "—",
        }
        for key, val in values.items():
            if key in self._patient_value_labels:
                self._patient_value_labels[key].setText(val)
        self._patient_sex = str(sex) if sex else ""

    def _create_meas_row(self, param, val, norm, status):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(10, 5, 10, 5)

        status_color = T.GREEN
        if status in ("Wydłużony", "Skrócony", "Odchylona"):
            status_color = T.RED
        elif status == "—":
            status_color = T.TEXT_MUTED

        for text, w_pct, is_mono, is_status in [
            (param, 150, True, False), (val, 100, True, False),
            (norm, 130, True, False), (status, 80, False, True)
        ]:
            lbl = QLabel(text)
            lbl.setFixedWidth(w_pct)
            style = f"font-size: 13px; color: {T.TEXT}; background: transparent;"
            if is_mono:
                style += " font-family: Menlo;"
            if is_status:
                style += f" color: {status_color}; font-weight: 600;"
            lbl.setStyleSheet(style)
            rl.addWidget(lbl)
        row.setStyleSheet(
            f"background: transparent;"
            f" border-bottom: 1px solid {T.BORDER_LIGHT};"
        )
        return row

    def set_measurements(self, measurements_dict):
        self._last_measurements = measurements_dict
        sex = getattr(self, "_patient_sex", "")

        if sex == "K":
            qtc_threshold = 460
            qtc_range = "<460 ms"
        else:
            qtc_threshold = 450
            qtc_range = "<450 ms"

        def _fmt_val(val, unit):
            if val == "N/A" or val is None or val == "":
                return "—"
            if isinstance(val, str):
                stripped = val.strip()
                if stripped.endswith(unit):
                    return stripped
                try:
                    v = float(stripped.split()[0])
                except (ValueError, IndexError):
                    return stripped
            else:
                try:
                    v = float(val)
                except (ValueError, TypeError):
                    return f"{val} {unit}"
            if v == int(v):
                return f"{int(v)} {unit}"
            return f"{v:.1f} {unit}"

        def _to_num(val):
            if val == "N/A" or val is None or val == "":
                return None
            if isinstance(val, str):
                try:
                    return float(val.strip().split()[0])
                except (ValueError, IndexError):
                    return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def _status(val, lo, hi):
            v = _to_num(val)
            if v is None:
                return "—"
            if lo is not None and v < lo:
                return "Skrócony"
            if hi is not None and v > hi:
                return "Wydłużony"
            return "Norma"

        def _axis_status(val):
            v = _to_num(val)
            if v is None:
                return "—"
            if -30 <= v <= 90:
                return "Norma"
            return "Odchylona"

        hr = measurements_dict.get("hr")
        pr = measurements_dict.get("pr")
        qrs = measurements_dict.get("qrs")
        qt = measurements_dict.get("qt")
        qtc = measurements_dict.get("qtc")
        axis = measurements_dict.get("axis")

        rows_data = [
            ("HR", _fmt_val(hr, "bpm"), "60-100 bpm", _status(hr, 60, 100)),
            ("PR interval", _fmt_val(pr, "ms"), "120-200 ms", _status(pr, 120, 200)),
            ("QRS", _fmt_val(qrs, "ms"), "<120 ms", _status(qrs, None, 120)),
            ("QT", _fmt_val(qt, "ms"), "zależny od HR", "—" if qt == "N/A" or qt is None or qt == "" else "Norma"),
            ("QTc (Bazett)", _fmt_val(qtc, "ms"), qtc_range, _status(qtc, None, qtc_threshold)),
            ("Oś", _fmt_val(axis, "°"), "-30° do +90°", _axis_status(axis)),
        ]

        while self._meas_table_layout.count() > 1:
            item = self._meas_table_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        for param, val, norm, status in rows_data:
            row = self._create_meas_row(param, val, norm, status)
            self._meas_table_layout.addWidget(row)

    def set_annotations(self, annotations):
        self._last_annotations = annotations
        while self._ann_layout.count():
            item = self._ann_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not annotations:
            self._ann_header.setText("ADNOTACJE (0)")
            no_ann = QLabel("Brak adnotacji")
            no_ann.setStyleSheet(
                f"font-size: 12px; color: {T.TEXT_MUTED};"
                f" padding: 6px 0; background: transparent;"
            )
            self._ann_empty_label = no_ann
            self._ann_layout.addWidget(no_ann)
            return

        self._ann_header.setText(f"ADNOTACJE ({len(annotations)})")
        for ann in annotations:
            lead = ann.get("lead", "?")
            t1 = ann.get("t1", 0)
            t2 = ann.get("t2", 0)
            category = ann.get("category", "")
            note = ann.get("note", "")

            meta = f"{lead}: {t1:.2f} — {t2:.2f} s | {category}"
            item = QWidget()
            item.setStyleSheet(
                f"background: transparent;"
                f" border-bottom: 1px solid {T.BORDER_LIGHT};"
            )
            il = QVBoxLayout(item)
            il.setContentsMargins(0, 5, 0, 5)
            meta_lbl = QLabel(meta)
            meta_lbl.setStyleSheet(
                f"font-family: Menlo; color: {T.TEXT_MUTED};"
                f" font-size: 12px; background: transparent;"
            )
            il.addWidget(meta_lbl)
            if note:
                text_lbl = QLabel(note)
                text_lbl.setStyleSheet(
                    f"color: {T.TEXT_SECONDARY}; font-size: 12px;"
                    f" background: transparent;"
                )
                il.addWidget(text_lbl)
            self._ann_layout.addWidget(item)

    def set_results(self, probabilities: dict, model_name: str = "", elapsed: float = 0.0):
        self._last_results = (probabilities, model_name, elapsed)
        from ui.theme import CLASS_NAMES_PL
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_cls, top_prob = sorted_items[0]
        self._ai_diag.setText(f"{CLASS_NAMES_PL.get(top_cls, top_cls)} — {top_prob * 100:.1f}%")
        self._ai_model.setText(f"Model: {model_name} | Czas: {elapsed:.1f} s")

    def set_scan_summary(self, scan_results: list, model_name: str = ""):
        self._last_scan = (scan_results, model_name)
        from ui.theme import CLASS_NAMES_PL, TARGET_CLASSES

        while self._ai_per_class_layout.count():
            item = self._ai_per_class_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not scan_results:
            self._ai_diag.setText("Brak wyników skanowania")
            self._ai_healthy.setText("")
            self._ai_model.setText(f"Model: {model_name}" if model_name else "")
            return

        per_class_max: dict[str, tuple[float, float, float]] = {}
        valid_classes = set(TARGET_CLASSES) - {"class_healthy"}
        for r in scan_results:
            probs = r.get("probs") or {}
            for cls, p in probs.items():
                if cls not in valid_classes:
                    continue
                cur = per_class_max.get(cls)
                if cur is None or p > cur[0]:
                    per_class_max[cls] = (float(p), float(r.get("t_start", 0)), float(r.get("t_end", 0)))

        if per_class_max:
            top_cls, (top_p, top_s, top_e) = max(per_class_max.items(), key=lambda kv: kv[1][0])
            self._ai_diag.setText(
                f"Najwyższe: {CLASS_NAMES_PL.get(top_cls, top_cls)} — {top_p * 100:.1f}% "
                f"({top_s:.1f}–{top_e:.1f} s)"
            )

        hdr = QWidget()
        hdr.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 6)
        hl.setSpacing(8)
        for text, w in [("Jednostka chorobowa", 240), ("Maks. %", 70), ("Czas", 100)]:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"font-size: 10px; font-weight: 700; color: {T.TEXT_DIM};"
                f" letter-spacing: 0.5px; background: transparent;"
            )
            lbl.setFixedWidth(w)
            hl.addWidget(lbl)
        hl.addStretch()
        self._ai_per_class_layout.addWidget(hdr)

        sorted_classes = [
            (cls, vals) for cls, vals in
            sorted(per_class_max.items(), key=lambda kv: kv[1][0], reverse=True)
            if vals[0] >= 0.01
        ]
        for i, (cls, (p, s, e)) in enumerate(sorted_classes):
            if i > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setFixedHeight(1)
                sep.setStyleSheet(
                    f"background: {T.AMBER_BORDER}; border: none; max-height: 1px;"
                )
                self._ai_per_class_layout.addWidget(sep)
            row = QWidget()
            row.setObjectName(f"aiClassRow_{i}")
            row.setStyleSheet(f"QWidget#aiClassRow_{i} {{ background: transparent; }}")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 6, 0, 6)
            rl.setSpacing(8)
            name_lbl = QLabel(CLASS_NAMES_PL.get(cls, cls))
            name_lbl.setStyleSheet(
                f"font-size: 12px; color: {T.TEXT}; background: transparent;"
            )
            name_lbl.setFixedWidth(240)
            rl.addWidget(name_lbl)
            pct_lbl = QLabel(f"{p * 100:.1f}%")
            pct_lbl.setStyleSheet(
                f"font-size: 12px; font-family: Menlo; font-weight: 600;"
                f" background: transparent;"
                f" color: {T.RED if p >= 0.5 else T.TEXT_MUTED};"
            )
            pct_lbl.setFixedWidth(70)
            rl.addWidget(pct_lbl)
            time_lbl = QLabel(f"{s:.1f}–{e:.1f} s")
            time_lbl.setStyleSheet(
                f"font-size: 11px; font-family: Menlo; color: {T.TEXT_MUTED};"
                f" background: transparent;"
            )
            time_lbl.setFixedWidth(100)
            rl.addWidget(time_lbl)
            rl.addStretch()
            self._ai_per_class_layout.addWidget(row)

        healthy_dominant_probs = []
        for r in scan_results:
            probs = r.get("probs") or {}
            if not probs:
                continue
            top = max(probs, key=probs.get)
            if top == "class_healthy":
                healthy_dominant_probs.append(float(probs[top]))

        if healthy_dominant_probs:
            avg = sum(healthy_dominant_probs) / len(healthy_dominant_probs) * 100
            mn = min(healthy_dominant_probs) * 100
            self._ai_healthy.setText(
                f"Zdrowy (gdy dominuje): średnia {avg:.1f}% · minimum {mn:.1f}%"
            )
        else:
            self._ai_healthy.setText("Zdrowy nie był dominującą klasą")

        self._ai_model.setText(f"Model: {model_name}" if model_name else "")

    def _selected_page_size(self):
        from PySide6.QtGui import QPageSize
        if self._page_sel.currentText() == "Letter":
            return QPageSize(QPageSize.PageSizeId.Letter)
        return QPageSize(QPageSize.PageSizeId.A4)

    def _grab_full_report(self):
        self.report.adjustSize()
        return self.report.grab()

    def _render_pixmap_to_printer(self, printer):
        from PySide6.QtCore import QRect
        pix = self._grab_full_report()
        painter = QPainter()
        if not painter.begin(printer):
            return False
        try:
            page = painter.viewport()
            src_w = pix.width()
            src_h = pix.height()
            if src_w <= 0 or src_h <= 0:
                return False
            scale = min(page.width() / src_w, page.height() / src_h)
            tw = int(src_w * scale)
            th = int(src_h * scale)
            x = page.x() + (page.width() - tw) // 2
            y = page.y() + (page.height() - th) // 2
            painter.drawPixmap(QRect(x, y, tw, th), pix)
        finally:
            painter.end()
        return True

    def _export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Eksportuj PDF", "raport_ekg.pdf", "PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageSize(self._selected_page_size())
        ok = self._render_pixmap_to_printer(printer)
        from PySide6.QtWidgets import QMessageBox
        if ok:
            QMessageBox.information(self, "Eksport PDF", f"Zapisano: {path}")
        else:
            QMessageBox.warning(self, "Eksport PDF", "Nie udało się zapisać pliku.")

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Eksportuj PNG", "raport_ekg.png", "PNG (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        pixmap = self._grab_full_report()
        from PySide6.QtWidgets import QMessageBox
        if pixmap.save(path, "PNG"):
            QMessageBox.information(self, "Eksport PNG", f"Zapisano: {path}")
        else:
            QMessageBox.warning(self, "Eksport PNG", "Nie udało się zapisać pliku.")

    def _print(self):
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPageSize(self._selected_page_size())
        dialog = QPrintDialog(printer, self)
        if dialog.exec() == QPrintDialog.DialogCode.Accepted:
            self._render_pixmap_to_printer(printer)
