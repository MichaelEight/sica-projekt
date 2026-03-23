"""Side panels: Info, Caliper, Annotation, AI Results, Monitor sidebar."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QComboBox, QTextEdit,
                                QScrollArea, QSizePolicy)

import ui.theme as T
from ui.widgets import section_header, info_row, make_action_btn


# Info Panel (Patient + Measurements)
class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        layout.addWidget(section_header("Pacjent"))
        self.patient_rows = {}
        for key, label, val, unit in [
            ("id", "ID", "00888", ""),
            ("age", "Wiek", "62", "lat"),
            ("sex", "Płeć", "M", ""),
            ("date", "Data", "15.03.26", ""),
        ]:
            row = info_row(label, val, unit)
            self.patient_rows[key] = row
            layout.addWidget(row)

        layout.addSpacing(10)

        layout.addWidget(section_header("Pomiary"))
        self.meas_rows = {}
        for key, label, val, unit in [
            ("hr", "HR", "72", "bpm"),
            ("pr", "PR", "164", "ms"),
            ("qrs", "QRS", "88", "ms"),
            ("qt", "QT", "392", "ms"),
            ("qtc", "QTc", "429", "ms"),
            ("axis", "Oś", "+55°", ""),
        ]:
            row = info_row(label, val, unit)
            self.meas_rows[key] = row
            layout.addWidget(row)

        layout.addStretch()

    def set_patient(self, patient_id="", age="", sex="", date=""):
        pass

    def set_measurements(self, hr="", pr="", qrs="", qt_val="", qtc="", axis=""):
        pass

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")


# Caliper Panel
class CaliperPanel(QWidget):
    add_caliper = Signal()
    clear_all = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"border-bottom: 1px solid {T.BORDER}; padding: 12px 14px;")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(14, 12, 14, 12)
        h_title = QLabel("Suwmiarka")
        h_title.setFont(QFont(".AppleSystemUIFont", 13, QFont.DemiBold))
        h_layout.addWidget(h_title)
        h_sub = QLabel("Kliknij 2 punkty na sygnale, aby zmierzyć")
        h_sub.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
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
        self._add_measurement("Pomiar 1 — PR interval", "164 ms",
                              "Odpr: II | 1.220 s → 1.384 s", T.ACCENT, "blue")
        self._add_measurement("Pomiar 2 — QRS", "88 ms",
                              "Odpr: II | 1.384 s → 1.472 s", T.PURPLE, "purple")
        self._add_measurement("Pomiar 3 — R-R", "832 ms",
                              "Odpr: II | 1.432 s → 2.264 s", T.GREEN, "green")

        # HR box
        hr_box = QFrame()
        hr_box.setStyleSheet(f"""
            QFrame {{ background: {T.GREEN_BG}; border: 1px solid {T.GREEN_BORDER}; border-radius: 6px;
                      margin: 6px 10px; padding: 10px 12px; }}
        """)
        hr_layout = QVBoxLayout(hr_box)
        hr_layout.setContentsMargins(10, 10, 10, 10)
        hr_label = QLabel("HR Z R-R")
        hr_label.setStyleSheet(f"font-size: 11px; color: {T.GREEN}; font-weight: 600;")
        hr_layout.addWidget(hr_label)
        hr_value = QLabel("72 bpm")
        hr_value.setStyleSheet(f"font-size: 24px; font-weight: 700; font-family: Menlo; color: {T.GREEN};")
        hr_layout.addWidget(hr_value)
        hr_detail = QLabel("Pomiar 3 (R-R = 832 ms)")
        hr_detail.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED};")
        hr_layout.addWidget(hr_detail)
        self.meas_layout.addWidget(hr_box)

        # Actions
        actions = QWidget()
        actions.setStyleSheet(f"border-top: 1px solid {T.BORDER};")
        a_layout = QVBoxLayout(actions)
        a_layout.setContentsMargins(10, 10, 10, 10)
        a_layout.setSpacing(6)
        btn_add = make_action_btn("Dodaj suwmiarkę", primary=True)
        btn_add.clicked.connect(self.add_caliper.emit)
        a_layout.addWidget(btn_add)
        btn_clear = make_action_btn("Wyczyść wszystkie")
        btn_clear.clicked.connect(self.clear_all.emit)
        a_layout.addWidget(btn_clear)
        layout.addWidget(actions)

    def _add_measurement(self, title, value, detail, color, color_class):
        bg_colors = {"blue": T.BLUE_BG, "purple": T.PURPLE_BG, "green": T.GREEN_BG}
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {bg_colors.get(color_class, T.BG_SECONDARY)};
                border-left: 4px solid {color};
                border-radius: 6px;
                margin: 6px 10px;
                padding: 10px 12px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {T.TEXT_MUTED};")
        card_layout.addWidget(title_lbl)

        val_lbl = QLabel(value)
        val_lbl.setStyleSheet(f"font-size: 20px; font-weight: 700; font-family: Menlo; color: {color};")
        card_layout.addWidget(val_lbl)

        detail_lbl = QLabel(detail)
        detail_lbl.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED};")
        card_layout.addWidget(detail_lbl)

        self.meas_layout.addWidget(card)

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")


# Annotation Panel
class AnnotationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"padding: 12px 14px; border-bottom: 1px solid {T.BORDER};")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(14, 12, 14, 12)
        h_title = QLabel("Adnotacje")
        h_title.setFont(QFont(".AppleSystemUIFont", 13, QFont.DemiBold))
        h_layout.addWidget(h_title)
        layout.addWidget(header)

        # New annotation form
        form = QWidget()
        form.setStyleSheet(f"border-bottom: 1px solid {T.BORDER};")
        f_layout = QVBoxLayout(form)
        f_layout.setContentsMargins(12, 12, 12, 12)
        f_layout.setSpacing(8)

        f_header = QLabel("Nowa adnotacja")
        f_header.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {T.ACCENT};")
        f_layout.addWidget(f_header)

        region = QLabel("II: 2.30 s — 2.85 s")
        region.setStyleSheet(f"""
            font-size: 12px; font-family: Menlo; color: {T.TEXT_SECONDARY};
            background: {T.BLUE_BG}; padding: 6px 8px; border-radius: 4px;
        """)
        f_layout.addWidget(region)

        cat_label = QLabel("Kategoria")
        cat_label.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED};")
        f_layout.addWidget(cat_label)
        self.category = QComboBox()
        self.category.addItems(["Patologia", "Norma", "Artefakt", "Do weryfikacji"])
        self.category.setStyleSheet(f"""
            padding: 7px 8px; border: 1px solid {T.BORDER}; border-radius: 6px; font-size: 13px;
        """)
        f_layout.addWidget(self.category)

        note_label = QLabel("Notatka")
        note_label.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED};")
        f_layout.addWidget(note_label)
        self.note_edit = QTextEdit()
        self.note_edit.setPlaceholderText("Podejrzenie uniesienia ST")
        self.note_edit.setMaximumHeight(60)
        self.note_edit.setStyleSheet(f"""
            padding: 7px 8px; border: 1px solid {T.BORDER}; border-radius: 6px; font-size: 13px;
        """)
        f_layout.addWidget(self.note_edit)

        btn_row = QHBoxLayout()
        btn_save = make_action_btn("Zapisz", primary=True)
        btn_row.addWidget(btn_save)
        btn_png = make_action_btn("Eksportuj PNG")
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
            font-size: 12px; font-weight: 600; color: {T.TEXT_DIM};
            text-transform: uppercase; margin-top: 12px; margin-bottom: 8px;
        """)
        saved_layout.addWidget(saved_header)

        for meta, badge, badge_cls, text, border_color in [
            ("II: 0.40 — 1.20 s", "Norma", "gn",
             "Prawidłowy kompleks PQRST, rytm zatokowy", T.GREEN),
            ("V1: 3.10 — 3.60 s", "Do weryfikacji", "yl",
             "Szerokie S w V1, możliwe RBBB", T.YELLOW),
        ]:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background: {T.BG_SECONDARY}; border-left: 3px solid {border_color};
                    border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;
                }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(4)

            meta_row = QHBoxLayout()
            meta_lbl = QLabel(meta)
            meta_lbl.setStyleSheet(f"font-size: 11px; font-family: Menlo; color: {T.TEXT_MUTED};")
            meta_row.addWidget(meta_lbl)
            badge_colors = {"gn": (T.GREEN, T.BADGE_NORM_BG, T.BADGE_NORM_TEXT),
                            "yl": (T.YELLOW, T.BADGE_WARN_BG, T.BADGE_WARN_TEXT)}
            bg_c, text_c = badge_colors[badge_cls][1], badge_colors[badge_cls][2]
            badge_lbl = QLabel(badge)
            badge_lbl.setStyleSheet(f"""
                font-size: 10px; padding: 1px 6px; border-radius: 3px;
                font-weight: 600; background: {bg_c}; color: {text_c};
            """)
            meta_row.addWidget(badge_lbl)
            meta_row.addStretch()
            card_layout.addLayout(meta_row)

            text_lbl = QLabel(text)
            text_lbl.setStyleSheet(f"font-size: 12px; color: {T.TEXT_SECONDARY};")
            text_lbl.setWordWrap(True)
            card_layout.addWidget(text_lbl)

            goto = QLabel(f'<a style="color:{T.ACCENT}; font-weight:500;">Przejdź →</a>')
            goto.setTextFormat(Qt.RichText)
            goto.setStyleSheet("font-size: 11px;")
            card_layout.addWidget(goto)

            saved_layout.addWidget(card)

        saved_layout.addStretch()
        scroll.setWidget(saved_w)
        layout.addWidget(scroll, stretch=1)

        # Export all
        footer = QWidget()
        footer.setStyleSheet(f"border-top: 1px solid {T.BORDER};")
        ft_layout = QVBoxLayout(footer)
        ft_layout.setContentsMargins(10, 10, 10, 10)
        btn_export = make_action_btn("Eksportuj wszystkie adnotacje")
        ft_layout.addWidget(btn_export)
        layout.addWidget(footer)

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")


# Results Panel (AI)
class ResultsPanel(QWidget):
    rerun = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(f"border-bottom: 1px solid {T.BORDER};")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(10, 8, 10, 8)
        h_layout.setSpacing(2)
        h_title = QLabel("Wyniki analizy")
        h_title.setFont(QFont(".AppleSystemUIFont", 13, QFont.DemiBold))
        h_layout.addWidget(h_title)
        self._model_label = QLabel("")
        self._model_label.setStyleSheet(f"font-size: 10px; color: {T.TEXT_MUTED};")
        self._model_label.setWordWrap(True)
        h_layout.addWidget(self._model_label)
        layout.addWidget(header)

        # Loading indicator
        self._loading_label = QLabel("Analizuję...")
        self._loading_label.setAlignment(Qt.AlignCenter)
        self._loading_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {T.ACCENT}; padding: 40px;"
        )
        self._loading_label.hide()
        layout.addWidget(self._loading_label)

        # Scroll area for results
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("border: none;")
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content_layout.setAlignment(Qt.AlignTop)

        # Top prediction card
        self._pred_card = QFrame()
        self._pred_card.setStyleSheet(f"""
            QFrame {{
                background: {T.AMBER_BG}; border: 1px solid {T.AMBER_BORDER};
                border-radius: 6px; margin: 6px 8px;
            }}
        """)
        pc_layout = QVBoxLayout(self._pred_card)
        pc_layout.setContentsMargins(10, 8, 10, 8)
        pc_layout.setSpacing(2)
        self._pred_name = QLabel("")
        self._pred_name.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {T.AMBER_TEXT};")
        self._pred_name.setWordWrap(True)
        pc_layout.addWidget(self._pred_name)
        # Percentage + bar in one row
        pct_row = QHBoxLayout()
        pct_row.setSpacing(8)
        self._pred_pct = QLabel("")
        self._pred_pct.setStyleSheet(f"font-size: 26px; font-weight: 700; font-family: Menlo; color: {T.AMBER_TEXT};")
        pct_row.addWidget(self._pred_pct)
        self._bar_bg = QFrame()
        self._bar_bg.setFixedHeight(8)
        self._bar_bg.setStyleSheet(f"background: {T.BORDER}; border-radius: 4px;")
        self._bar_bg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._bar_fill = QFrame(self._bar_bg)
        self._bar_fill.setFixedHeight(8)
        self._bar_fill.setFixedWidth(0)
        self._bar_fill.setStyleSheet(f"background: {T.AMBER}; border-radius: 4px;")
        pct_row.addWidget(self._bar_bg)
        pc_layout.addLayout(pct_row)
        self._content_layout.addWidget(self._pred_card)

        # All classes header
        cls_header = QLabel("WSZYSTKIE KLASY")
        cls_header.setStyleSheet(f"""
            font-size: 10px; font-weight: 700; color: {T.TEXT_DIM};
            text-transform: uppercase; padding: 4px 8px 2px 8px;
        """)
        self._content_layout.addWidget(cls_header)

        # Container for dynamic class rows
        self._classes_container = QWidget()
        self._classes_layout = QVBoxLayout(self._classes_container)
        self._classes_layout.setContentsMargins(0, 0, 0, 0)
        self._classes_layout.setSpacing(0)
        self._content_layout.addWidget(self._classes_container)

        self._content_layout.addStretch()
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, stretch=1)

        # Footer
        footer = QWidget()
        footer.setStyleSheet(f"border-top: 1px solid {T.BORDER};")
        ft_layout = QVBoxLayout(footer)
        ft_layout.setContentsMargins(8, 6, 8, 6)
        ft_layout.setSpacing(4)
        self._btn_rerun = make_action_btn("Ponów analizę")
        self._btn_rerun.clicked.connect(self.rerun.emit)
        ft_layout.addWidget(self._btn_rerun)
        disc = QLabel("Wynik ma charakter pomocniczy.\nDecyzja diagnostyczna należy do lekarza.")
        disc.setStyleSheet(f"font-size: 10px; color: {T.TEXT_DIM};")
        disc.setAlignment(Qt.AlignCenter)
        disc.setWordWrap(True)
        ft_layout.addWidget(disc)
        layout.addWidget(footer)

    def set_loading(self):
        """Show loading state before inference starts."""
        self._scroll.hide()
        self._loading_label.show()
        self._model_label.setText("Ładowanie modelu...")
        self._btn_rerun.setEnabled(False)

    def set_results(self, probabilities: dict, model_name: str = "",
                    elapsed: float = 0.0, window_label: str = "",
                    ground_truth: dict | None = None):
        """Populate panel with real inference results."""
        from ui.theme import CLASS_NAMES_PL

        self._loading_label.hide()
        self._scroll.show()
        self._btn_rerun.setEnabled(True)

        # Header — model info on one line, window + time on second
        line1 = f"Model: {model_name}"
        line2_parts = []
        if window_label:
            line2_parts.append(f"Okno: {window_label}")
        line2_parts.append(f"Obliczenia: {elapsed:.2f} s")
        self._model_label.setText(f"{line1}\n{' | '.join(line2_parts)}")

        # Sort by probability descending
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_cls, top_prob = sorted_items[0]

        # Top prediction card
        pred_text = CLASS_NAMES_PL.get(top_cls, top_cls)
        if ground_truth:
            gt_sorted = sorted(ground_truth.items(), key=lambda x: x[1], reverse=True)
            gt_top_cls, gt_top_prob = gt_sorted[0]
            if gt_top_prob > 0:
                gt_name = CLASS_NAMES_PL.get(gt_top_cls, gt_top_cls)
                match = top_cls == gt_top_cls
                self._pred_card.setStyleSheet(f"""
                    QFrame {{
                        background: {T.GREEN_BG if match else T.AMBER_BG};
                        border: 1px solid {T.GREEN_BORDER if match else T.AMBER_BORDER};
                        border-radius: 6px; margin: 6px 8px;
                    }}
                """)
            else:
                match = None
        else:
            match = None

        if match is None:
            self._pred_card.setStyleSheet(f"""
                QFrame {{
                    background: {T.AMBER_BG}; border: 1px solid {T.AMBER_BORDER};
                    border-radius: 6px; margin: 6px 8px;
                }}
            """)

        self._pred_name.setText(pred_text)
        self._pred_pct.setText(f"{top_prob * 100:.1f}%")
        bar_max_w = self._bar_bg.width() if self._bar_bg.width() > 10 else 150
        self._bar_fill.setFixedWidth(max(1, int(bar_max_w * top_prob)))

        # Clear old class rows
        while self._classes_layout.count():
            item = self._classes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Ground truth header if available
        if ground_truth:
            gt_header = QLabel("PREDYKCJA vs ADNOTACJA")
            gt_header.setStyleSheet(f"""
                font-size: 10px; font-weight: 700; color: {T.TEXT_DIM};
                text-transform: uppercase; padding: 2px 0;
            """)
            self._classes_layout.addWidget(gt_header)

        # Build class rows — stacked: name line, then bar+pct line
        for cls, prob in sorted_items:
            is_top = (cls == top_cls)
            gt_val = ground_truth.get(cls, 0.0) if ground_truth else None

            row = QWidget()
            r_layout = QVBoxLayout(row)
            r_layout.setContentsMargins(8, 2, 8, 2)
            r_layout.setSpacing(1)

            # Name
            name_lbl = QLabel(CLASS_NAMES_PL.get(cls, cls))
            name_lbl.setStyleSheet(
                f"font-size: 11px; font-weight: {'600' if is_top else '400'}; "
                f"color: {T.TEXT if is_top else T.TEXT_SECONDARY};"
            )
            r_layout.addWidget(name_lbl)

            # Prediction bar + percentage
            bar_row = QHBoxLayout()
            bar_row.setSpacing(4)
            bar_row.setContentsMargins(0, 0, 0, 0)

            pred_lbl = QLabel("P")
            pred_lbl.setFixedWidth(12)
            pred_lbl.setStyleSheet(f"font-size: 9px; color: {T.TEXT_DIM}; font-weight: 600;")
            bar_row.addWidget(pred_lbl)

            bar = QFrame()
            bar.setFixedHeight(6)
            bar.setStyleSheet(f"background: {T.BORDER_LIGHT}; border-radius: 3px;")
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            fill = QFrame(bar)
            fill.setFixedHeight(6)
            fill.setFixedWidth(max(1, int(prob * 200)))
            fill.setStyleSheet(f"background: {T.AMBER if is_top else T.BAR_BG}; border-radius: 3px;")
            bar_row.addWidget(bar)

            pct_lbl = QLabel(f"{prob * 100:.1f}%")
            pct_lbl.setFixedWidth(50)
            pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pct_lbl.setStyleSheet(
                f"font-size: 11px; font-family: Menlo; "
                f"color: {T.TEXT if is_top else T.TEXT_MUTED}; "
                f"font-weight: {'600' if is_top else '400'};"
            )
            bar_row.addWidget(pct_lbl)
            r_layout.addLayout(bar_row)

            # Ground truth indicator (if available)
            if gt_val is not None:
                gt_row = QHBoxLayout()
                gt_row.setSpacing(4)
                gt_row.setContentsMargins(0, 0, 0, 0)

                gt_label = QLabel("A")
                gt_label.setFixedWidth(12)
                present = gt_val > 0.0
                color = T.GREEN if present else T.TEXT_DIM
                gt_label.setStyleSheet(f"font-size: 9px; color: {color}; font-weight: 600;")
                gt_row.addWidget(gt_label)

                gt_text = QLabel("TAK" if present else "NIE")
                gt_text.setStyleSheet(
                    f"font-size: 11px; font-weight: 600; color: {color};"
                )
                gt_row.addWidget(gt_text)
                gt_row.addStretch()
                r_layout.addLayout(gt_row)

            self._classes_layout.addWidget(row)

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")


# Monitor Sidebar
class MonitorSidebar(QWidget):
    speed_changed = Signal(float)
    leads_changed = Signal(list)
    pause_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        # Pause button
        self._paused = False
        self.pause_btn = QPushButton("⏸  Pauza")
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.setFocusPolicy(Qt.NoFocus)
        self.pause_btn.setStyleSheet(self._pause_btn_style(False))
        self.pause_btn.clicked.connect(self._on_pause)
        layout.addWidget(self.pause_btn)

        # Playback speed
        self._pill_groups = {}
        self._pill_active_idx = {}
        self._add_pills_section(layout, "Prędkość odtwarzania",
                                ["0.5x", "1x", "2x"], 1, "speed")

        # Lead selection
        sec_lbl = QLabel("ODPROWADZENIA")
        sec_lbl.setStyleSheet(f"""
            font-size: 11px; font-weight: 700; color: {T.TEXT_DIM};
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
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setProperty("active", is_active)
            btn.clicked.connect(lambda checked, b=btn, l=lead: self._toggle_lead(b, l))
            self.lead_btns[lead] = btn
            lead_grid.addWidget(btn, i // 3, i % 3)
        layout.addLayout(lead_grid)

        layout.addStretch()

    def _pause_btn_style(self, paused):
        if paused:
            return f"""
                font-size: 12px; padding: 8px 14px; border: 1px solid {T.ACCENT};
                border-radius: 6px; background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                font-weight: 600;
            """
        return f"""
            font-size: 12px; padding: 8px 14px; border: 1px solid {T.BORDER};
            border-radius: 6px; background: {T.WHITE}; color: {T.TEXT};
            font-weight: 600;
        """

    def _lead_btn_style(self, active):
        if active:
            return f"""
                font-size: 11px; padding: 4px 8px; border: 1px solid {T.ACCENT};
                border-radius: 4px; background: {T.ICON_BG}; color: {T.ACCENT};
                font-family: Menlo; font-weight: 600;
            """
        return f"""
            font-size: 11px; padding: 4px 8px; border: 1px solid {T.BORDER};
            border-radius: 4px; background: {T.WHITE}; color: {T.TEXT_DIM};
            font-family: Menlo; font-weight: 600;
        """

    def _toggle_lead(self, btn, lead):
        is_active = not btn.property("active")
        btn.setProperty("active", is_active)
        btn.setStyleSheet(self._lead_btn_style(is_active))
        active_leads = [ld for ld, b in self.lead_btns.items() if b.property("active")]
        self.leads_changed.emit(active_leads)

    def _on_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.pause_btn.setText("▶  Wznów")
        else:
            self.pause_btn.setText("⏸  Pauza")
        self.pause_btn.setStyleSheet(self._pause_btn_style(self._paused))
        self.pause_toggled.emit(self._paused)

    def _pill_style(self, active):
        return f"""
            font-size: 11px; padding: 4px 10px; border: 1px solid {T.BORDER};
            border-radius: 4px;
            background: {T.ACCENT if active else T.WHITE};
            color: {T.ACCENT_TEXT if active else T.TEXT_MUTED};
        """

    def _add_pills_section(self, layout, title, options, active_idx, group_name=""):
        lbl = QLabel(title.upper())
        lbl.setStyleSheet(f"""
            font-size: 11px; font-weight: 700; color: {T.TEXT_DIM};
            text-transform: uppercase; letter-spacing: 0.5px;
        """)
        layout.addWidget(lbl)
        pills = QHBoxLayout()
        pills.setSpacing(3)
        btns = []
        for i, opt in enumerate(options):
            btn = QPushButton(opt)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(self._pill_style(i == active_idx))
            btn.clicked.connect(lambda checked, idx=i, g=group_name: self._on_pill(g, idx))
            btns.append(btn)
            pills.addWidget(btn)
        self._pill_groups[group_name] = btns
        self._pill_active_idx[group_name] = active_idx
        layout.addLayout(pills)

    def _on_pill(self, group_name, idx):
        btns = self._pill_groups.get(group_name, [])
        for i, btn in enumerate(btns):
            btn.setStyleSheet(self._pill_style(i == idx))
        self._pill_active_idx[group_name] = idx
        if group_name == "speed":
            speeds = [0.5, 1.0, 2.0]
            self.speed_changed.emit(speeds[idx])

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")
        self.pause_btn.setStyleSheet(self._pause_btn_style(self._paused))
        for lead, btn in self.lead_btns.items():
            btn.setStyleSheet(self._lead_btn_style(btn.property("active")))
        for group_name, btns in self._pill_groups.items():
            active_idx = self._pill_active_idx.get(group_name, 0)
            for i, btn in enumerate(btns):
                btn.setStyleSheet(self._pill_style(i == active_idx))
