"""Side panels: Info, Caliper, Annotation, AI Results, Monitor sidebar."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QComboBox, QTextEdit,
                                QScrollArea, QSizePolicy)

from ui.theme import (ACCENT, WHITE, BORDER, BORDER_LIGHT, TEXT, TEXT_MUTED,
                       TEXT_DIM, GREEN, YELLOW, PURPLE, AMBER_BG, AMBER_BORDER,
                       AMBER_TEXT, AMBER_SUB)


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        font-size: 11px; font-weight: 700; color: {TEXT_DIM};
        text-transform: uppercase; letter-spacing: 0.5px;
        padding-bottom: 4px; border-bottom: 1px solid {BORDER_LIGHT};
    """)
    return lbl


def _info_row(label: str, value: str, unit: str = "") -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
    val_text = f'{value} <span style="font-size:10px;color:{TEXT_DIM};">{unit}</span>' if unit else value
    val = QLabel(val_text)
    val.setTextFormat(Qt.RichText)
    val.setStyleSheet("font-weight: 600; font-family: Menlo; font-size: 13px;")
    val.setAlignment(Qt.AlignRight)
    layout.addWidget(lbl)
    layout.addStretch()
    layout.addWidget(val)
    return row


# ── Info Panel (Patient + Measurements) ────────
class InfoPanel(QWidget):
    """Left info panel for 12-lead view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setStyleSheet(f"background: {WHITE}; border-right: 1px solid {BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        # Patient section
        layout.addWidget(_section_header("Pacjent"))
        self.patient_rows = {}
        for key, label, val, unit in [
            ("id", "ID", "00888", ""),
            ("age", "Wiek", "62", "lat"),
            ("sex", "P\u0142e\u0107", "M", ""),
            ("date", "Data", "15.03.26", ""),
        ]:
            row = _info_row(label, val, unit)
            self.patient_rows[key] = row
            layout.addWidget(row)

        layout.addSpacing(10)

        # Measurements section
        layout.addWidget(_section_header("Pomiary"))
        self.meas_rows = {}
        for key, label, val, unit in [
            ("hr", "HR", "72", "bpm"),
            ("pr", "PR", "164", "ms"),
            ("qrs", "QRS", "88", "ms"),
            ("qt", "QT", "392", "ms"),
            ("qtc", "QTc", "429", "ms"),
            ("axis", "O\u015b", "+55\u00b0", ""),
        ]:
            row = _info_row(label, val, unit)
            self.meas_rows[key] = row
            layout.addWidget(row)

        layout.addStretch()

    def set_patient(self, patient_id="", age="", sex="", date=""):
        """Update patient info — placeholder for real data."""
        pass

    def set_measurements(self, hr="", pr="", qrs="", qt_val="", qtc="", axis=""):
        """Update measurements — placeholder for real data."""
        pass


# ── Caliper Panel ──────────────────────────────
class CaliperPanel(QWidget):
    """Right panel for caliper measurements."""

    add_caliper = Signal()
    clear_all = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setStyleSheet(f"background: {WHITE}; border-left: 1px solid {BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"border-bottom: 1px solid {BORDER}; padding: 12px 14px;")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(14, 12, 14, 12)
        h_title = QLabel("Suwmiarka")
        h_title.setFont(QFont(".AppleSystemUIFont", 13, QFont.DemiBold))
        h_layout.addWidget(h_title)
        h_sub = QLabel("Kliknij 2 punkty na sygnale, aby zmierzy\u0107")
        h_sub.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        h_layout.addWidget(h_sub)
        layout.addWidget(header)

        # Measurements scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        self.meas_container = QWidget()
        self.meas_layout = QVBoxLayout(self.meas_container)
        self.meas_layout.setContentsMargins(0, 0, 0, 0)
        self.meas_layout.setSpacing(0)
        self.meas_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.meas_container)
        layout.addWidget(scroll, stretch=1)

        # Demo measurements
        self._add_measurement("Pomiar 1 \u2014 PR interval", "164 ms", "Odpr: II | 1.220 s \u2192 1.384 s",
                              ACCENT, "blue")
        self._add_measurement("Pomiar 2 \u2014 QRS", "88 ms", "Odpr: II | 1.384 s \u2192 1.472 s",
                              PURPLE, "purple")
        self._add_measurement("Pomiar 3 \u2014 R-R", "832 ms", "Odpr: II | 1.432 s \u2192 2.264 s",
                              GREEN, "green")

        # HR box
        hr_box = QFrame()
        hr_box.setStyleSheet(f"""
            QFrame {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px;
                      margin: 6px 10px; padding: 10px 12px; }}
        """)
        hr_layout = QVBoxLayout(hr_box)
        hr_layout.setContentsMargins(10, 10, 10, 10)
        hr_label = QLabel("HR Z R-R")
        hr_label.setStyleSheet(f"font-size: 11px; color: {GREEN}; font-weight: 600;")
        hr_layout.addWidget(hr_label)
        hr_value = QLabel("72 bpm")
        hr_value.setStyleSheet(f"font-size: 24px; font-weight: 700; font-family: Menlo; color: {GREEN};")
        hr_layout.addWidget(hr_value)
        hr_detail = QLabel("Pomiar 3 (R-R = 832 ms)")
        hr_detail.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        hr_layout.addWidget(hr_detail)
        self.meas_layout.addWidget(hr_box)

        # Actions
        actions = QWidget()
        actions.setStyleSheet(f"border-top: 1px solid {BORDER};")
        a_layout = QVBoxLayout(actions)
        a_layout.setContentsMargins(10, 10, 10, 10)
        a_layout.setSpacing(6)
        btn_add = QPushButton("Dodaj suwmiark\u0119")
        btn_add.setObjectName("primary")
        btn_add.clicked.connect(self.add_caliper.emit)
        a_layout.addWidget(btn_add)
        btn_clear = QPushButton("Wyczy\u015b\u0107 wszystkie")
        btn_clear.setStyleSheet(f"""
            padding: 8px; border-radius: 6px; border: 1px solid {BORDER};
            background: {WHITE}; color: #4b5563; font-size: 12px;
        """)
        btn_clear.clicked.connect(self.clear_all.emit)
        a_layout.addWidget(btn_clear)
        layout.addWidget(actions)

    def _add_measurement(self, title: str, value: str, detail: str, color: str, color_class: str):
        bg_colors = {"blue": "#eff6ff", "purple": "#f5f3ff", "green": "#f0fdf4"}
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {bg_colors.get(color_class, '#f9fafb')};
                border-left: 4px solid {color};
                border-radius: 6px;
                margin: 6px 10px;
                padding: 10px 12px;
            }}
        """)
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(10, 8, 10, 8)
        c_layout.setSpacing(2)

        num = QLabel(title)
        num.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {TEXT_MUTED};")
        c_layout.addWidget(num)

        val = QLabel(value)
        val.setStyleSheet(f"font-size: 20px; font-weight: 700; font-family: Menlo; color: {color};")
        c_layout.addWidget(val)

        det = QLabel(detail)
        det.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        c_layout.addWidget(det)

        self.meas_layout.addWidget(card)


# ── Annotation Panel ───────────────────────────
class AnnotationPanel(QWidget):
    """Right panel for annotations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self.setStyleSheet(f"background: {WHITE}; border-left: 1px solid {BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"padding: 12px 14px; border-bottom: 1px solid {BORDER};")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(14, 12, 14, 12)
        h_title = QLabel("Adnotacje")
        h_title.setFont(QFont(".AppleSystemUIFont", 13, QFont.DemiBold))
        h_layout.addWidget(h_title)
        layout.addWidget(header)

        # New annotation form
        form = QWidget()
        form.setStyleSheet(f"border-bottom: 1px solid {BORDER};")
        f_layout = QVBoxLayout(form)
        f_layout.setContentsMargins(12, 12, 12, 12)
        f_layout.setSpacing(8)

        f_header = QLabel("Nowa adnotacja")
        f_header.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {ACCENT};")
        f_layout.addWidget(f_header)

        region = QLabel("II: 2.30 s \u2014 2.85 s")
        region.setStyleSheet(f"""
            font-size: 12px; font-family: Menlo; color: #4b5563;
            background: #eff6ff; padding: 6px 8px; border-radius: 4px;
        """)
        f_layout.addWidget(region)

        cat_label = QLabel("Kategoria")
        cat_label.setStyleSheet(f"font-size: 12px; color: {TEXT_MUTED};")
        f_layout.addWidget(cat_label)
        self.category = QComboBox()
        self.category.addItems(["Patologia", "Norma", "Artefakt", "Do weryfikacji"])
        self.category.setStyleSheet(f"""
            padding: 7px 8px; border: 1px solid {BORDER}; border-radius: 6px; font-size: 13px;
        """)
        f_layout.addWidget(self.category)

        note_label = QLabel("Notatka")
        note_label.setStyleSheet(f"font-size: 12px; color: {TEXT_MUTED};")
        f_layout.addWidget(note_label)
        self.note_edit = QTextEdit()
        self.note_edit.setPlaceholderText("Podejrzenie uniesienia ST")
        self.note_edit.setMaximumHeight(60)
        self.note_edit.setStyleSheet(f"""
            padding: 7px 8px; border: 1px solid {BORDER}; border-radius: 6px; font-size: 13px;
        """)
        f_layout.addWidget(self.note_edit)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Zapisz")
        btn_save.setObjectName("primary")
        btn_row.addWidget(btn_save)
        btn_png = QPushButton("Eksportuj PNG")
        btn_png.setStyleSheet(f"""
            padding: 7px; border-radius: 6px; border: 1px solid {BORDER};
            background: {WHITE}; color: #4b5563; font-size: 12px;
        """)
        btn_row.addWidget(btn_png)
        f_layout.addLayout(btn_row)
        layout.addWidget(form)

        # Saved annotations
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        saved_w = QWidget()
        saved_layout = QVBoxLayout(saved_w)
        saved_layout.setContentsMargins(12, 0, 12, 0)
        saved_layout.setAlignment(Qt.AlignTop)

        saved_header = QLabel("ZAPISANE (2)")
        saved_header.setStyleSheet(f"""
            font-size: 12px; font-weight: 600; color: {TEXT_DIM};
            text-transform: uppercase; margin-top: 12px; margin-bottom: 8px;
        """)
        saved_layout.addWidget(saved_header)

        # Demo annotations
        for meta, badge, badge_cls, text, border_color in [
            ("II: 0.40 \u2014 1.20 s", "Norma", "gn",
             "Prawid\u0142owy kompleks PQRST, rytm zatokowy", GREEN),
            ("V1: 3.10 \u2014 3.60 s", "Do weryfikacji", "yl",
             "Szerokie S w V1, mo\u017cliwe RBBB", YELLOW),
        ]:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background: #f9fafb; border-left: 3px solid {border_color};
                    border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;
                }}
            """)
            c_layout = QVBoxLayout(card)
            c_layout.setContentsMargins(10, 8, 10, 8)
            c_layout.setSpacing(4)

            meta_row = QHBoxLayout()
            meta_lbl = QLabel(meta)
            meta_lbl.setStyleSheet(f"font-size: 11px; font-family: Menlo; color: {TEXT_MUTED};")
            meta_row.addWidget(meta_lbl)
            badge_colors = {"gn": (GREEN, "#d1fae5", "#065f46"),
                            "yl": (YELLOW, "#fef3c7", "#92400e")}
            bg_c, text_c = badge_colors[badge_cls][1], badge_colors[badge_cls][2]
            badge_lbl = QLabel(badge)
            badge_lbl.setStyleSheet(f"""
                font-size: 10px; padding: 1px 6px; border-radius: 3px;
                font-weight: 600; background: {bg_c}; color: {text_c};
            """)
            meta_row.addWidget(badge_lbl)
            meta_row.addStretch()
            c_layout.addLayout(meta_row)

            text_lbl = QLabel(text)
            text_lbl.setStyleSheet("font-size: 12px; color: #4b5563;")
            text_lbl.setWordWrap(True)
            c_layout.addWidget(text_lbl)

            goto = QLabel(f'<a style="color:{ACCENT}; font-weight:500;">Przejd\u017a \u2192</a>')
            goto.setTextFormat(Qt.RichText)
            goto.setStyleSheet("font-size: 11px;")
            c_layout.addWidget(goto)

            saved_layout.addWidget(card)

        saved_layout.addStretch()
        scroll.setWidget(saved_w)
        layout.addWidget(scroll, stretch=1)

        # Export all
        footer = QWidget()
        footer.setStyleSheet(f"border-top: 1px solid {BORDER};")
        ft_layout = QVBoxLayout(footer)
        ft_layout.setContentsMargins(10, 10, 10, 10)
        btn_export = QPushButton("Eksportuj wszystkie adnotacje")
        btn_export.setStyleSheet(f"""
            padding: 8px; border-radius: 6px; border: 1px solid {BORDER};
            background: {WHITE}; color: #4b5563; font-size: 12px; font-weight: 500;
        """)
        ft_layout.addWidget(btn_export)
        layout.addWidget(footer)


# ── Results Panel (AI) ─────────────────────────
class ResultsPanel(QWidget):
    """Right panel showing AI analysis results."""

    rerun = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self.setStyleSheet(f"background: {WHITE}; border-left: 1px solid {BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"padding: 12px 14px; border-bottom: 1px solid {BORDER};")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(14, 12, 14, 12)
        h_title = QLabel("Wyniki analizy")
        h_title.setFont(QFont(".AppleSystemUIFont", 13, QFont.DemiBold))
        h_layout.addWidget(h_title)
        h_sub = QLabel("Model: Inception1D | Czas: 1.2 s")
        h_sub.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        h_layout.addWidget(h_sub)
        layout.addWidget(header)

        # Scroll area for results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        content = QWidget()
        c_layout = QVBoxLayout(content)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setAlignment(Qt.AlignTop)

        # Top prediction card
        pred_card = QFrame()
        pred_card.setStyleSheet(f"""
            QFrame {{
                background: {AMBER_BG}; border: 1px solid {AMBER_BORDER};
                border-radius: 8px; margin: 10px 12px; padding: 12px;
            }}
        """)
        pc_layout = QVBoxLayout(pred_card)
        pc_layout.setContentsMargins(12, 12, 12, 12)
        pc_layout.setSpacing(4)
        pred_name = QLabel("Zawa\u0142 mi\u0119\u015bnia sercowego (MI)")
        pred_name.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {AMBER_TEXT};")
        pred_name.setWordWrap(True)
        pc_layout.addWidget(pred_name)
        pred_conf_lbl = QLabel("Pewno\u015b\u0107 modelu")
        pred_conf_lbl.setStyleSheet(f"font-size: 11px; color: {AMBER_SUB};")
        pc_layout.addWidget(pred_conf_lbl)
        pred_pct = QLabel("87.2%")
        pred_pct.setStyleSheet(f"font-size: 30px; font-weight: 700; font-family: Menlo; color: {AMBER_TEXT};")
        pc_layout.addWidget(pred_pct)
        # Progress bar
        bar_bg = QFrame()
        bar_bg.setFixedHeight(8)
        bar_bg.setStyleSheet(f"background: {BORDER}; border-radius: 4px;")
        bar_fill = QFrame(bar_bg)
        bar_fill.setFixedHeight(8)
        bar_fill.setFixedWidth(int(200 * 0.87))
        bar_fill.setStyleSheet("background: #f59e0b; border-radius: 4px;")
        pc_layout.addWidget(bar_bg)
        c_layout.addWidget(pred_card)

        # All classes section
        cls_header = QLabel("WSZYSTKIE KLASY")
        cls_header.setStyleSheet(f"""
            font-size: 11px; font-weight: 700; color: {TEXT_DIM};
            text-transform: uppercase; margin: 8px 12px;
        """)
        c_layout.addWidget(cls_header)

        classes = [
            ("Zawa\u0142 (MI)", 87.2, True),
            ("Zdrowy (NORM)", 5.8, False),
            ("Niedokrwienne (ISC_)", 3.1, False),
            ("Niespecyficzne (NST_)", 1.9, False),
            ("LBBB", 0.8, False),
            ("RBBB", 0.5, False),
            ("Przerost LK (LVH)", 0.4, False),
            ("Przerost PK (RVH)", 0.3, False),
        ]
        for name, pct, is_top in classes:
            row = QWidget()
            row.setStyleSheet("margin: 0 12px;")
            r_layout = QHBoxLayout(row)
            r_layout.setContentsMargins(12, 3, 12, 3)
            r_layout.setSpacing(6)

            name_lbl = QLabel(name)
            name_lbl.setFixedWidth(150)
            name_lbl.setStyleSheet(
                f"font-size: 12px; font-weight: {'600' if is_top else '400'}; "
                f"color: {TEXT if is_top else '#4b5563'};"
            )
            r_layout.addWidget(name_lbl)

            bar = QFrame()
            bar.setFixedHeight(6)
            bar.setStyleSheet("background: #f3f4f6; border-radius: 3px;")
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            fill = QFrame(bar)
            fill.setFixedHeight(6)
            fill.setFixedWidth(max(1, int(pct)))
            fill.setStyleSheet(f"background: {'#f59e0b' if is_top else '#d1d5db'}; border-radius: 3px;")
            r_layout.addWidget(bar)

            pct_lbl = QLabel(f"{pct}%")
            pct_lbl.setFixedWidth(40)
            pct_lbl.setAlignment(Qt.AlignRight)
            pct_lbl.setStyleSheet(
                f"font-size: 11px; font-family: Menlo; "
                f"color: {TEXT if is_top else TEXT_MUTED}; "
                f"font-weight: {'600' if is_top else '400'};"
            )
            r_layout.addWidget(pct_lbl)
            c_layout.addWidget(row)

        c_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        # Footer
        footer = QWidget()
        footer.setStyleSheet(f"border-top: 1px solid {BORDER};")
        ft_layout = QVBoxLayout(footer)
        ft_layout.setContentsMargins(10, 10, 10, 10)
        ft_layout.setSpacing(6)
        btn_rerun = QPushButton("Pon\u00f3w analiz\u0119")
        btn_rerun.setStyleSheet(f"""
            padding: 8px; border-radius: 6px; border: 1px solid {BORDER};
            background: {WHITE}; color: #4b5563; font-size: 12px; font-weight: 500;
        """)
        btn_rerun.clicked.connect(self.rerun.emit)
        ft_layout.addWidget(btn_rerun)
        disc = QLabel("Wynik ma charakter pomocniczy.\nOstateczna decyzja diagnostyczna nale\u017cy do lekarza.")
        disc.setStyleSheet(f"font-size: 11px; color: {TEXT_DIM}; text-align: center;")
        disc.setAlignment(Qt.AlignCenter)
        disc.setWordWrap(True)
        ft_layout.addWidget(disc)
        layout.addWidget(footer)


# ── Monitor Sidebar ────────────────────────────
class MonitorSidebar(QWidget):
    """Left sidebar for monitor mode controls."""

    speed_changed = Signal(float)
    leads_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setStyleSheet(f"background: {WHITE}; border-right: 1px solid {BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # HR display
        hr_frame = QFrame()
        hr_frame.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {GREEN}; border-radius: 10px;
                background: #f0fdf4; padding: 16px 0;
            }}
        """)
        hr_layout = QVBoxLayout(hr_frame)
        hr_layout.setAlignment(Qt.AlignCenter)
        heart = QLabel("\u2665")
        heart.setStyleSheet("font-size: 18px;")
        heart.setAlignment(Qt.AlignCenter)
        hr_layout.addWidget(heart)
        self.hr_num = QLabel("72")
        self.hr_num.setStyleSheet(f"font-size: 48px; font-weight: 700; font-family: Menlo; color: {GREEN};")
        self.hr_num.setAlignment(Qt.AlignCenter)
        hr_layout.addWidget(self.hr_num)
        hr_unit = QLabel("BPM")
        hr_unit.setStyleSheet(f"font-size: 14px; color: {GREEN};")
        hr_unit.setAlignment(Qt.AlignCenter)
        hr_layout.addWidget(hr_unit)
        layout.addWidget(hr_frame)

        # Playback speed
        self._add_pills_section(layout, "Pr\u0119dko\u015b\u0107 odtwarzania",
                                ["0.5x", "1x", "2x"], 1)

        # Sweep speed
        self._add_pills_section(layout, "Pr\u0119dko\u015b\u0107 przesuwu",
                                ["25 mm/s", "50 mm/s"], 0)

        # Lead selection
        sec_lbl = QLabel("ODPROWADZENIA")
        sec_lbl.setStyleSheet(f"""
            font-size: 11px; font-weight: 700; color: {TEXT_DIM};
            text-transform: uppercase; letter-spacing: 0.5px;
        """)
        layout.addWidget(sec_lbl)

        from PySide6.QtWidgets import QGridLayout
        lead_grid = QGridLayout()
        lead_grid.setSpacing(4)
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        active = {"II", "V1", "V5"}
        self.lead_btns = {}
        for i, lead in enumerate(leads):
            btn = QPushButton(lead)
            is_active = lead in active
            btn.setStyleSheet(self._lead_btn_style(is_active))
            btn.setFixedSize(48, 28)
            btn.setProperty("active", is_active)
            btn.clicked.connect(lambda checked, b=btn, l=lead: self._toggle_lead(b, l))
            self.lead_btns[lead] = btn
            lead_grid.addWidget(btn, i // 3, i % 3)
        layout.addLayout(lead_grid)

        layout.addStretch()

    def _lead_btn_style(self, active: bool) -> str:
        if active:
            return f"""
                font-size: 11px; padding: 4px 8px; border: 1px solid {ACCENT};
                border-radius: 4px; background: #eef4ff; color: {ACCENT};
                font-family: Menlo; font-weight: 600;
            """
        return f"""
            font-size: 11px; padding: 4px 8px; border: 1px solid {BORDER};
            border-radius: 4px; background: {WHITE}; color: {TEXT_DIM};
            font-family: Menlo; font-weight: 600;
        """

    def _toggle_lead(self, btn, lead):
        is_active = not btn.property("active")
        btn.setProperty("active", is_active)
        btn.setStyleSheet(self._lead_btn_style(is_active))
        active_leads = [l for l, b in self.lead_btns.items() if b.property("active")]
        self.leads_changed.emit(active_leads)

    def _add_pills_section(self, layout, title, options, active_idx):
        lbl = QLabel(title.upper())
        lbl.setStyleSheet(f"""
            font-size: 11px; font-weight: 700; color: {TEXT_DIM};
            text-transform: uppercase; letter-spacing: 0.5px;
        """)
        layout.addWidget(lbl)
        pills = QHBoxLayout()
        pills.setSpacing(3)
        for i, opt in enumerate(options):
            btn = QPushButton(opt)
            is_active = i == active_idx
            btn.setStyleSheet(f"""
                font-size: 11px; padding: 4px 10px; border: 1px solid {BORDER};
                border-radius: 4px;
                background: {ACCENT if is_active else WHITE};
                color: {'white' if is_active else TEXT_MUTED};
            """)
            pills.addWidget(btn)
        layout.addLayout(pills)
