"""Report preview page matching v2 08-report design."""
import numpy as np
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

        date = QLabel("Wygenerowano: 15.03.2026, 22:15 | Plik: 00888_lr.dat")
        date.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED};")
        date.setAlignment(Qt.AlignCenter)
        r_layout.addWidget(date)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {T.TEXT}; border: none; max-height: 2px;")
        r_layout.addWidget(line)

        # Patient info grid
        pgrid = QWidget()
        from PySide6.QtWidgets import QGridLayout
        pg = QGridLayout(pgrid)
        pg.setSpacing(8)
        patient_data = [
            ("ID pacjenta", "00888", "Data badania", "15.03.2026"),
            ("Wiek", "62 lat", "Czas trwania", "10.0 s"),
            ("Płeć", "Mężczyzna", "Częstotliwość", "500 Hz"),
        ]
        for row_i, (l1, v1, l2, v2) in enumerate(patient_data):
            for col_i, (label, value) in enumerate([(l1, v1), (l2, v2)]):
                lbl = QLabel(label)
                lbl.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; min-width: 110px;")
                val = QLabel(value)
                val.setStyleSheet("font-size: 13px; font-weight: 600; font-family: Menlo;")
                pg.addWidget(lbl, row_i, col_i * 2)
                pg.addWidget(val, row_i, col_i * 2 + 1)
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

        measurements = [
            ("HR", "72 bpm", "60-100 bpm", "Norma"),
            ("PR interval", "164 ms", "120-200 ms", "Norma"),
            ("QRS", "88 ms", "<120 ms", "Norma"),
            ("QT", "392 ms", "350-440 ms", "Norma"),
            ("QTc", "429 ms", "350-440 ms", "Norma"),
            ("Oś", "+55°", "-30° do +90°", "Norma"),
        ]
        for param, val, norm, status in measurements:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(10, 5, 10, 5)
            for text, w_pct, is_mono, is_green in [
                (param, 150, True, False), (val, 100, True, False),
                (norm, 130, True, False), (status, 80, False, True)
            ]:
                lbl = QLabel(text)
                lbl.setFixedWidth(w_pct)
                style = "font-size: 13px;"
                if is_mono:
                    style += " font-family: Menlo;"
                if is_green:
                    style += f" color: {T.GREEN}; font-weight: 600;"
                lbl.setStyleSheet(style)
                rl.addWidget(lbl)
            row.setStyleSheet(f"border-bottom: 1px solid {T.BORDER_LIGHT};")
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
        ai_diag = QLabel("Zawał mięśnia sercowego (MI) — 87.2%")
        ai_diag.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {T.AMBER_TEXT};")
        ai_layout.addWidget(ai_diag)
        ai_conf = QLabel("NORM: 5.8% | ISC_: 3.1% | NST_: 1.9% | LBBB: 0.8% | RBBB: 0.5%")
        ai_conf.setStyleSheet(f"font-size: 12px; color: {T.AMBER_SUB};")
        ai_conf.setWordWrap(True)
        ai_layout.addWidget(ai_conf)
        ai_model = QLabel("Model: Inception1D | Czas: 1.2 s")
        ai_model.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED};")
        ai_layout.addWidget(ai_model)
        r_layout.addWidget(ai_box)

        # Annotations
        sec_ann = QLabel("ADNOTACJE (2)")
        sec_ann.setStyleSheet("font-size: 13px; font-weight: 700; letter-spacing: 0.5px; margin-top: 10px;")
        r_layout.addWidget(sec_ann)

        for meta, text in [
            ("II: 0.40 — 1.20 s | Norma", "Prawidłowy kompleks PQRST, rytm zatokowy"),
            ("V1: 3.10 — 3.60 s | Do weryfikacji", "Szerokie S, możliwe RBBB"),
        ]:
            item = QWidget()
            item.setStyleSheet(f"border-bottom: 1px solid {T.BORDER_LIGHT};")
            il = QVBoxLayout(item)
            il.setContentsMargins(0, 5, 0, 5)
            meta_lbl = QLabel(meta)
            meta_lbl.setStyleSheet(f"font-family: Menlo; color: {T.TEXT_MUTED}; font-size: 12px;")
            il.addWidget(meta_lbl)
            text_lbl = QLabel(text)
            text_lbl.setStyleSheet(f"color: {T.TEXT_SECONDARY}; font-size: 12px;")
            il.addWidget(text_lbl)
            r_layout.addWidget(item)

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
