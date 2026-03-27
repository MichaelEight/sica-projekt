"""Report preview page matching v2 08-report design."""
import numpy as np
from datetime import datetime
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QPainterPath
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QScrollArea, QComboBox,
                                QFileDialog)
from PySide6.QtPrintSupport import QPrinter, QPrintDialog

import ui.theme as T
from ui.theme import LEAD_SEEDS, LEAD_AMPS
from ui.ekg_canvas import synth_ekg
from ui.widgets import make_logo, make_separator, make_action_btn


class EkgPreviewWidget(QWidget):
    """Small EKG preview for the report (12-lead thumbnail)."""

    def __init__(self, signal=None, leads=None, fs=500, parent=None):
        super().__init__(parent)
        self.signal = signal
        self.leads = leads or []
        self.fs = fs
        self.setFixedHeight(160)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(T.WHITE))

        grid = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
                ["III", "aVF", "V3", "V6"]]
        row_h = h / 4
        col_w = w / 4

        sq_x = col_w / 60
        sq_y = row_h / 15
        painter.setPen(QPen(QColor(T.GRID_MINOR), 0.3))
        for x in np.arange(0, w, sq_x):
            painter.drawLine(int(x), 0, int(x), h)
        for y in np.arange(0, h, sq_y):
            painter.drawLine(0, int(y), w, int(y))
        painter.setPen(QPen(QColor(T.GRID_MAJOR), 0.6))
        for x in np.arange(0, w, sq_x * 5):
            painter.drawLine(int(x), 0, int(x), h)
        for y in np.arange(0, h, sq_y * 5):
            painter.drawLine(0, int(y), w, int(y))

        painter.setPen(QPen(QColor(T.BAR_BG), 1))
        for c in range(1, 4):
            painter.drawLine(int(c * col_w), 0, int(c * col_w), int(3 * row_h))
        for r in range(1, 4):
            painter.drawLine(0, int(r * row_h), w, int(r * row_h))

        painter.setPen(QPen(QColor(T.SIGNAL_COLOR), 1.2))
        for r_i, row_leads in enumerate(grid):
            for c_i, lead in enumerate(row_leads):
                x_off = c_i * col_w
                y_mid = r_i * row_h + row_h / 2
                mv_px = row_h / 3
                seed = LEAD_SEEDS.get(lead, 0)
                amp = LEAD_AMPS.get(lead, 1)
                t_start = c_i * 2.5
                t_end = t_start + 2.5

                if self.signal is not None and lead in self.leads:
                    lead_idx = self.leads.index(lead)
                    n_start = int(t_start * self.fs)
                    n_end = min(int(t_end * self.fs), self.signal.shape[0])
                    sig = self.signal[n_start:n_end, lead_idx]
                    path = QPainterPath()
                    for px_i in range(int(col_w)):
                        frac = px_i / col_w
                        idx = min(int(frac * len(sig)), len(sig) - 1)
                        v = sig[idx]
                        py = y_mid - v * mv_px
                        if px_i == 0:
                            path.moveTo(x_off + px_i, py)
                        else:
                            path.lineTo(x_off + px_i, py)
                    painter.drawPath(path)
                else:
                    t = np.linspace(t_start, t_end, int(col_w))
                    vals = synth_ekg(t, seed, amp)
                    path = QPainterPath()
                    for px_i in range(len(vals)):
                        py = y_mid - vals[px_i] * mv_px
                        if px_i == 0:
                            path.moveTo(x_off + px_i, py)
                        else:
                            path.lineTo(x_off + px_i, py)
                    painter.drawPath(path)

                painter.setPen(QColor(T.TEXT))
                painter.setFont(QFont("Menlo", 8, QFont.Bold))
                painter.drawText(int(x_off + 4), int(r_i * row_h + 14), lead)
                painter.setPen(QPen(QColor(T.SIGNAL_COLOR), 1.2))

        # Rhythm strip (row 4 = II)
        y_mid = 3 * row_h + row_h / 2
        mv_px = row_h / 3
        if self.signal is not None and "II" in self.leads:
            idx = self.leads.index("II")
            sig = self.signal[:, idx]
            path = QPainterPath()
            for px_i in range(w):
                frac = px_i / w
                si = min(int(frac * len(sig)), len(sig) - 1)
                py = y_mid - sig[si] * mv_px
                if px_i == 0:
                    path.moveTo(px_i, py)
                else:
                    path.lineTo(px_i, py)
            painter.drawPath(path)
        else:
            t = np.linspace(0, 10, w)
            vals = synth_ekg(t, 0.5, 1.0)
            path = QPainterPath()
            for px_i in range(len(vals)):
                py = y_mid - vals[px_i] * mv_px
                if px_i == 0:
                    path.moveTo(px_i, py)
                else:
                    path.lineTo(px_i, py)
            painter.drawPath(path)

        painter.setPen(QColor(T.TEXT))
        painter.setFont(QFont("Menlo", 8, QFont.Bold))
        painter.drawText(4, int(3 * row_h + 14), "II")
        painter.end()


class ReportPage(QWidget):
    """Report preview page with export options."""

    go_back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal = None
        self.leads = []
        self.fs = 500
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(48)
        topbar.setStyleSheet(f"background: {T.TOPBAR};")
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(14, 0, 14, 0)
        tb.setSpacing(8)

        logo = make_logo(14)
        tb.addWidget(logo)

        tb.addWidget(make_separator(1, 24))

        self.file_info = QLabel("00888_lr.dat")
        self.file_info.setStyleSheet(f"font-size:12px; color:{T.BTN_TEXT}; font-family:Menlo;")
        tb.addWidget(self.file_info)
        tb.addStretch()

        badge = QLabel("Podgląd raportu")
        badge.setStyleSheet(f"""
            font-size: 12px; background: {T.BADGE_BLUE_BG}; color: {T.BADGE_BLUE_TEXT};
            padding: 5px 12px; border-radius: 5px; font-weight: 600;
        """)
        tb.addWidget(badge)

        tb.addWidget(make_separator(1, 24))

        btn_back = QPushButton("Powrót do widoku")
        btn_back.setObjectName("secondary")
        btn_back.setStyleSheet(f"background:{T.BTN_DARK};color:{T.BTN_TEXT};font-size:12px;padding:6px 14px;border-radius:5px;border:none;")
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(self.go_back.emit)
        tb.addWidget(btn_back)
        outer.addWidget(topbar)

        # Content (scrollable report)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background: {T.WHITE}; border: none;")
        scroll.setAlignment(Qt.AlignCenter)

        self.report = QWidget()
        self.report.setFixedWidth(780)
        self.report.setStyleSheet(f"""
            QWidget {{
                background: {T.WHITE}; border: 1px solid {T.BORDER};
                border-radius: 8px;
            }}
        """)
        r_layout = QVBoxLayout(self.report)
        r_layout.setContentsMargins(28, 28, 28, 28)
        r_layout.setSpacing(8)

        title = QLabel("EKG Assistant — Raport badania")
        title.setFont(QFont(".AppleSystemUIFont", 16, QFont.DemiBold))
        title.setAlignment(Qt.AlignCenter)
        r_layout.addWidget(title)

        self._date_label = QLabel("Wygenerowano: — | Plik: —")
        self._date_label.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED};")
        self._date_label.setAlignment(Qt.AlignCenter)
        r_layout.addWidget(self._date_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {T.TEXT}; border: none; max-height: 2px;")
        r_layout.addWidget(line)

        # Patient info grid
        pgrid = QWidget()
        from PySide6.QtWidgets import QGridLayout
        pg = QGridLayout(pgrid)
        pg.setSpacing(8)
        self._patient_value_labels = {}
        patient_fields = [
            ("ID pacjenta", "patient_id", "Data badania", "date"),
            ("Wiek", "age", "Czas trwania", "duration"),
            ("Płeć", "sex", "Częstotliwość", "fs"),
        ]
        for row_i, (l1, k1, l2, k2) in enumerate(patient_fields):
            for col_i, (label, key) in enumerate([(l1, k1), (l2, k2)]):
                lbl = QLabel(label)
                lbl.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; min-width: 110px;")
                val = QLabel("—")
                val.setStyleSheet("font-size: 13px; font-weight: 600; font-family: Menlo;")
                pg.addWidget(lbl, row_i, col_i * 2)
                pg.addWidget(val, row_i, col_i * 2 + 1)
                self._patient_value_labels[key] = val
        r_layout.addWidget(pgrid)

        sep_line = QFrame()
        sep_line.setFrameShape(QFrame.HLine)
        sep_line.setStyleSheet(f"background: {T.BORDER}; border: none; max-height: 1px;")
        r_layout.addWidget(sep_line)

        # EKG preview
        sec_ecg = QLabel("ZAPIS EKG (12 ODPROWADZEŃ)")
        sec_ecg.setStyleSheet("font-size: 13px; font-weight: 700; letter-spacing: 0.5px; margin-top: 10px;")
        r_layout.addWidget(sec_ecg)

        self.ecg_preview = EkgPreviewWidget()
        self.ecg_preview.setStyleSheet(f"border: 1px solid {T.BORDER}; border-radius: 4px;")
        r_layout.addWidget(self.ecg_preview)

        ecg_meta = QLabel("25 mm/s | 10 mm/mV | 0.05-150 Hz")
        ecg_meta.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED}; font-family: Menlo;")
        r_layout.addWidget(ecg_meta)

        # Measurements table
        sec_meas = QLabel("POMIARY")
        sec_meas.setStyleSheet("font-size: 13px; font-weight: 700; letter-spacing: 0.5px; margin-top: 10px;")
        r_layout.addWidget(sec_meas)

        table = QWidget()
        t_layout = QVBoxLayout(table)
        t_layout.setContentsMargins(0, 0, 0, 0)
        t_layout.setSpacing(0)
        header_row = QWidget()
        hr_layout = QHBoxLayout(header_row)
        hr_layout.setContentsMargins(10, 6, 10, 6)
        for text, w_pct in [("Parametr", 150), ("Wartość", 100), ("Norma", 130), ("Status", 80)]:
            lbl = QLabel(text)
            lbl.setFixedWidth(w_pct)
            lbl.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; font-weight: 500;")
            hr_layout.addWidget(lbl)
        header_row.setStyleSheet(f"border-bottom: 1px solid {T.BORDER};")
        t_layout.addWidget(header_row)

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
        sec_ai = QLabel("ANALIZA AI")
        sec_ai.setStyleSheet("font-size: 13px; font-weight: 700; letter-spacing: 0.5px; margin-top: 10px;")
        r_layout.addWidget(sec_ai)

        ai_box = QFrame()
        ai_box.setStyleSheet(f"""
            QFrame {{
                background: {T.AMBER_BG}; border: 1px solid {T.AMBER_BORDER};
                border-radius: 8px; padding: 12px;
            }}
        """)
        ai_layout = QVBoxLayout(ai_box)
        ai_layout.setContentsMargins(12, 12, 12, 12)
        self._ai_diag = QLabel("Brak analizy")
        self._ai_diag.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {T.AMBER_TEXT};")
        ai_layout.addWidget(self._ai_diag)
        self._ai_conf = QLabel("")
        self._ai_conf.setStyleSheet(f"font-size: 12px; color: {T.AMBER_SUB};")
        self._ai_conf.setWordWrap(True)
        ai_layout.addWidget(self._ai_conf)
        self._ai_model = QLabel("")
        self._ai_model.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED};")
        ai_layout.addWidget(self._ai_model)
        r_layout.addWidget(ai_box)

        # Annotations
        self._ann_header = QLabel("ADNOTACJE (0)")
        self._ann_header.setStyleSheet("font-size: 13px; font-weight: 700; letter-spacing: 0.5px; margin-top: 10px;")
        r_layout.addWidget(self._ann_header)

        self._ann_container = QWidget()
        self._ann_layout = QVBoxLayout(self._ann_container)
        self._ann_layout.setContentsMargins(0, 0, 0, 0)
        self._ann_layout.setSpacing(0)
        no_ann = QLabel("Brak adnotacji")
        no_ann.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; padding: 6px 0;")
        self._ann_layout.addWidget(no_ann)
        r_layout.addWidget(self._ann_container)

        # Disclaimer
        disc = QLabel(
            "Wynik analizy AI ma charakter pomocniczy i nie stanowi diagnozy medycznej.\n"
            "Ostateczna decyzja diagnostyczna należy do lekarza specjalisty."
        )
        disc.setStyleSheet(f"""
            font-size: 11px; color: {T.TEXT_DIM}; text-align: center;
            margin-top: 16px; padding-top: 12px;
            border-top: 1px solid {T.BORDER};
        """)
        disc.setAlignment(Qt.AlignCenter)
        disc.setWordWrap(True)
        r_layout.addWidget(disc)

        r_layout.addStretch()

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        sc_layout = QVBoxLayout(scroll_content)
        sc_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        sc_layout.setContentsMargins(12, 12, 12, 20)
        sc_layout.addWidget(self.report)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, stretch=1)

        # Export bar
        export_bar = QWidget()
        export_bar.setFixedHeight(48)
        export_bar.setStyleSheet(f"background: {T.WHITE}; border-top: 1px solid {T.BORDER};")
        eb = QHBoxLayout(export_bar)
        eb.setContentsMargins(20, 0, 20, 0)
        eb.setSpacing(12)
        eb.setAlignment(Qt.AlignCenter)

        btn_pdf = QPushButton("Eksportuj PDF")
        btn_pdf.setStyleSheet(f"""
            padding: 8px 20px; border-radius: 6px; font-size: 13px; font-weight: 500;
            background: {T.ACCENT}; color: {T.ACCENT_TEXT}; border: 1px solid {T.ACCENT};
        """)
        btn_pdf.setCursor(Qt.PointingHandCursor)
        btn_pdf.clicked.connect(self._export_pdf)
        eb.addWidget(btn_pdf)

        secondary_style = f"""
            padding: 8px 20px; border-radius: 6px; font-size: 13px; font-weight: 500;
            background: {T.WHITE}; color: {T.TEXT_SECONDARY}; border: 1px solid {T.BORDER};
        """
        for label, handler in [("Eksportuj PNG", self._export_png), ("Drukuj", self._print)]:
            btn = QPushButton(label)
            btn.setStyleSheet(secondary_style)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(handler)
            eb.addWidget(btn)

        eb.addStretch()
        page_sel = QComboBox()
        page_sel.addItems(["A4", "Letter"])
        page_sel.setStyleSheet(f"font-size: 12px; padding: 6px 10px; border: 1px solid {T.BORDER}; border-radius: 6px;")
        eb.addWidget(page_sel)

        outer.addWidget(export_bar)

    def set_signal(self, signal, leads, fs, filename=""):
        self.signal = signal
        self.leads = leads
        self.fs = fs
        self.file_info.setText(filename)
        self.ecg_preview.signal = signal
        self.ecg_preview.leads = leads
        self.ecg_preview.fs = fs
        self.ecg_preview.update()

        # Update date label with current datetime and filename
        now = datetime.now().strftime("%d.%m.%Y, %H:%M")
        self._date_label.setText(f"Wygenerowano: {now} | Plik: {filename or '—'}")

    def set_patient_info(self, patient_id="", age="", sex="", date="", duration="", fs=""):
        """Update patient info grid with dynamic values."""
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
        # Store sex for QTc range display
        self._patient_sex = str(sex) if sex else ""

    def _create_meas_row(self, param, val, norm, status):
        """Create a single measurement row widget."""
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
            style = "font-size: 13px;"
            if is_mono:
                style += " font-family: Menlo;"
            if is_status:
                style += f" color: {status_color}; font-weight: 600;"
            lbl.setStyleSheet(style)
            rl.addWidget(lbl)
        row.setStyleSheet(f"border-bottom: 1px solid {T.BORDER_LIGHT};")
        return row

    def set_measurements(self, measurements_dict):
        """Update measurements table with dynamic values.

        measurements_dict keys: hr, pr, qrs, qt, qtc, axis
        Each value should be numeric or 'N/A'.
        """
        sex = getattr(self, "_patient_sex", "")

        # Determine QTc threshold based on sex
        if sex == "K":
            qtc_threshold = 460
            qtc_range = "<460 ms"
        else:
            qtc_threshold = 450
            qtc_range = "<450 ms"

        def _fmt_val(val, unit):
            if val == "N/A" or val is None or val == "":
                return "—"
            try:
                v = float(val)
                if v == int(v):
                    return f"{int(v)} {unit}"
                return f"{v:.0f} {unit}"
            except (ValueError, TypeError):
                return f"{val} {unit}"

        def _status(val, lo, hi):
            """Return status string given value and normal range bounds."""
            if val == "N/A" or val is None or val == "":
                return "—"
            try:
                v = float(val)
            except (ValueError, TypeError):
                return "—"
            if lo is not None and v < lo:
                return "Skrócony"
            if hi is not None and v > hi:
                return "Wydłużony"
            return "Norma"

        def _axis_status(val):
            if val == "N/A" or val is None or val == "":
                return "—"
            try:
                v = float(val)
            except (ValueError, TypeError):
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

        # Remove old measurement rows (everything after the header row)
        while self._meas_table_layout.count() > 1:
            item = self._meas_table_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        for param, val, norm, status in rows_data:
            row = self._create_meas_row(param, val, norm, status)
            self._meas_table_layout.addWidget(row)

    def set_annotations(self, annotations):
        """Update the annotations section with dynamic data.

        annotations: list of dicts with keys like 'lead', 't1', 't2', 'category', 'note'
        """
        # Clear existing
        while self._ann_layout.count():
            item = self._ann_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not annotations:
            self._ann_header.setText("ADNOTACJE (0)")
            no_ann = QLabel("Brak adnotacji")
            no_ann.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; padding: 6px 0;")
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
            item.setStyleSheet(f"border-bottom: 1px solid {T.BORDER_LIGHT};")
            il = QVBoxLayout(item)
            il.setContentsMargins(0, 5, 0, 5)
            meta_lbl = QLabel(meta)
            meta_lbl.setStyleSheet(f"font-family: Menlo; color: {T.TEXT_MUTED}; font-size: 12px;")
            il.addWidget(meta_lbl)
            if note:
                text_lbl = QLabel(note)
                text_lbl.setStyleSheet(f"color: {T.TEXT_SECONDARY}; font-size: 12px;")
                il.addWidget(text_lbl)
            self._ann_layout.addWidget(item)

    def set_results(self, probabilities: dict, model_name: str = "", elapsed: float = 0.0):
        """Update AI analysis section with real results."""
        from ui.theme import CLASS_NAMES_PL
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_cls, top_prob = sorted_items[0]
        self._ai_diag.setText(f"{CLASS_NAMES_PL.get(top_cls, top_cls)} — {top_prob * 100:.1f}%")
        others = [f"{CLASS_NAMES_PL.get(c, c)}: {p * 100:.1f}%" for c, p in sorted_items[1:]]
        self._ai_conf.setText(" | ".join(others))
        self._ai_model.setText(f"Model: {model_name} | Czas: {elapsed:.1f} s")

    def _export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Eksportuj PDF", "raport_ekg.pdf", "PDF (*.pdf)")
        if path:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(path)
            painter = QPainter(printer)
            self.report.render(painter)
            painter.end()

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Eksportuj PNG", "raport_ekg.png", "PNG (*.png)")
        if path:
            pixmap = self.report.grab()
            pixmap.save(path, "PNG")

    def _print(self):
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)
        if dialog.exec() == QPrintDialog.Accepted:
            painter = QPainter(printer)
            self.report.render(painter)
            painter.end()
