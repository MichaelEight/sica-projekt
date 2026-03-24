"""Side panels: Info, Caliper, Annotation, AI Results, Monitor sidebar."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QFrame, QComboBox, QTextEdit,
                                QScrollArea, QSizePolicy, QLineEdit)

import ui.theme as T
from ui.widgets import section_header, info_row, make_action_btn


# Info Panel (Patient + Measurements)
class InfoPanel(QWidget):
    patient_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        layout.addWidget(section_header("Pacjent"))
        self._patient_fields = {}
        for key, label in [
            ("name", "Imię"),
            ("id", "ID"),
            ("age", "Wiek"),
            ("sex", "Płeć"),
            ("date", "Data"),
        ]:
            row = self._editable_info_row(key, label)
            layout.addWidget(row)

        layout.addSpacing(10)

        layout.addWidget(section_header("Pomiary"))
        self._meas_labels = {}
        for key, label, val, unit in [
            ("hr", "HR", "", "bpm"),
            ("pr", "PR", "", "ms"),
            ("qrs", "QRS", "", "ms"),
            ("qt", "QT", "", "ms"),
            ("qtc", "QTc", "", "ms"),
            ("axis", "Oś", "", ""),
        ]:
            row = info_row(label, val, unit)
            self._meas_labels[key] = row
            layout.addWidget(row)

        layout.addStretch()

    def _editable_info_row(self, key, label):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px;")
        edit = QLineEdit()
        edit.setStyleSheet(
            f"font-weight: 600; font-family: Menlo; font-size: 13px; "
            f"border: 1px solid {T.BORDER}; border-radius: 3px; padding: 1px 4px;"
        )
        edit.setAlignment(Qt.AlignRight)
        edit.setFixedWidth(70)
        edit.textChanged.connect(self.patient_changed.emit)
        lay.addWidget(lbl)
        lay.addStretch()
        lay.addWidget(edit)
        self._patient_fields[key] = edit
        return row

    def set_patient(self, patient_id="", age="", sex="", date="", name=""):
        self._patient_fields["id"].setText(str(patient_id))
        self._patient_fields["age"].setText(str(age))
        self._patient_fields["sex"].setText(str(sex))
        self._patient_fields["date"].setText(str(date))
        self._patient_fields["name"].setText(str(name))

    def set_measurements(self, hr="", pr="", qrs="", qt_val="", qtc="", axis=""):
        mapping = {"hr": hr, "pr": pr, "qrs": qrs, "qt": qt_val, "qtc": qtc, "axis": axis}
        units = {"hr": "bpm", "pr": "ms", "qrs": "ms", "qt": "ms", "qtc": "ms", "axis": ""}
        labels = {"hr": "HR", "pr": "PR", "qrs": "QRS", "qt": "QT", "qtc": "QTc", "axis": "Oś"}
        for key, val in mapping.items():
            old_row = self._meas_labels[key]
            parent_layout = old_row.parentWidget().layout() if old_row.parentWidget() else self.layout()
            idx = self.layout().indexOf(old_row)
            old_row.setParent(None)
            old_row.deleteLater()
            new_row = info_row(labels[key], str(val), units[key])
            self._meas_labels[key] = new_row
            self.layout().insertWidget(idx, new_row)

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-right: 1px solid {T.BORDER};")


# Caliper Panel
class CaliperPanel(QWidget):
    add_caliper = Signal()
    clear_all = Signal()
    calipers_changed = Signal()
    caliper_deleted = Signal(int)

    _COLORS = [
        (T.ACCENT, T.BLUE_BG, "blue"),
        (T.PURPLE, T.PURPLE_BG, "purple"),
        (T.GREEN, T.GREEN_BG, "green"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")
        self._calipers: list[dict] = []

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

        # HR box (hidden by default)
        self._hr_box = QFrame()
        self._hr_box.setStyleSheet(f"""
            QFrame {{ background: {T.GREEN_BG}; border: 1px solid {T.GREEN_BORDER}; border-radius: 6px;
                      margin: 6px 10px; padding: 10px 12px; }}
        """)
        hr_layout = QVBoxLayout(self._hr_box)
        hr_layout.setContentsMargins(10, 10, 10, 10)
        self._hr_label = QLabel("HR Z R-R")
        self._hr_label.setStyleSheet(f"font-size: 11px; color: {T.GREEN}; font-weight: 600;")
        hr_layout.addWidget(self._hr_label)
        self._hr_value = QLabel("")
        self._hr_value.setStyleSheet(f"font-size: 24px; font-weight: 700; font-family: Menlo; color: {T.GREEN};")
        hr_layout.addWidget(self._hr_value)
        self._hr_detail = QLabel("")
        self._hr_detail.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED};")
        hr_layout.addWidget(self._hr_detail)
        self._hr_box.hide()
        self.meas_layout.addWidget(self._hr_box)

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

    def set_calipers(self, calipers_list: list[dict]):
        self._calipers = list(calipers_list)
        self._rebuild_cards()

    def _rebuild_cards(self):
        # Remove all cards except the HR box
        while self.meas_layout.count():
            item = self.meas_layout.takeAt(0)
            w = item.widget()
            if w and w is not self._hr_box:
                w.deleteLater()

        # Re-add caliper cards
        for idx, cal in enumerate(self._calipers):
            color, bg, color_class = self._COLORS[idx % len(self._COLORS)]
            dt_ms = abs(cal["t2"] - cal["t1"]) * 1000
            label = cal.get("label", f"{dt_ms:.0f} ms")
            detail = f"Odpr: {cal['lead']} | {cal['t1']:.3f} s → {cal['t2']:.3f} s"
            card = self._add_measurement(label, f"{dt_ms:.0f} ms", detail, color, color_class, idx)
            self.meas_layout.addWidget(card)

        # HR box: auto-calculate from last R-R caliper
        rr_cal = None
        for cal in reversed(self._calipers):
            if "r-r" in cal.get("label", "").lower() or "rr" in cal.get("label", "").lower():
                rr_cal = cal
                break
        if rr_cal:
            rr_ms = abs(rr_cal["t2"] - rr_cal["t1"]) * 1000
            if rr_ms > 0:
                hr = 60000.0 / rr_ms
                self._hr_value.setText(f"{hr:.0f} bpm")
                self._hr_detail.setText(f"R-R = {rr_ms:.0f} ms")
                self._hr_box.show()
            else:
                self._hr_box.hide()
        else:
            self._hr_box.hide()
        self.meas_layout.addWidget(self._hr_box)

    def _add_measurement(self, title, value, detail, color, color_class, index=-1):
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

        # Title row with delete button
        title_row = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {T.TEXT_MUTED};")
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 12px; border: none; background: transparent;
                color: {T.TEXT_DIM}; font-weight: 600;
            }}
            QPushButton:hover {{ color: {T.RED}; }}
        """)
        del_btn.clicked.connect(lambda checked, i=index: self._on_delete(i))
        title_row.addWidget(del_btn)
        card_layout.addLayout(title_row)

        val_lbl = QLabel(value)
        val_lbl.setStyleSheet(f"font-size: 20px; font-weight: 700; font-family: Menlo; color: {color};")
        card_layout.addWidget(val_lbl)

        detail_lbl = QLabel(detail)
        detail_lbl.setStyleSheet(f"font-size: 11px; color: {T.TEXT_MUTED};")
        card_layout.addWidget(detail_lbl)

        return card

    def _on_delete(self, index):
        if 0 <= index < len(self._calipers):
            self._calipers.pop(index)
            self._rebuild_cards()
            self.caliper_deleted.emit(index)
            self.calipers_changed.emit()

    def apply_theme(self):
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")


# Annotation Panel
class AnnotationPanel(QWidget):
    annotation_saved = Signal(str, str)
    annotations_changed = Signal()
    annotation_deleted = Signal(int)

    _BADGE_STYLES = {
        "Norma": (T.GREEN, T.BADGE_NORM_BG, T.BADGE_NORM_TEXT),
        "Patologia": (T.RED, T.AMBER_BG, T.AMBER_TEXT),
        "Do weryfikacji": (T.YELLOW, T.BADGE_WARN_BG, T.BADGE_WARN_TEXT),
        "Artefakt": (T.TEXT_DIM, T.TAG_BG, T.TEXT_DIM),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self.setStyleSheet(f"background: {T.WHITE}; border-left: 1px solid {T.BORDER};")
        self._annotations: list[dict] = []

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

        self._region_label = QLabel("—")
        self._region_label.setStyleSheet(f"""
            font-size: 12px; font-family: Menlo; color: {T.TEXT_SECONDARY};
            background: {T.BLUE_BG}; padding: 6px 8px; border-radius: 4px;
        """)
        f_layout.addWidget(self._region_label)

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
        btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(btn_save)
        btn_png = make_action_btn("Eksportuj PNG")
        btn_row.addWidget(btn_png)
        f_layout.addLayout(btn_row)
        layout.addWidget(form)

        # Saved annotations scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        self._saved_widget = QWidget()
        self._saved_layout = QVBoxLayout(self._saved_widget)
        self._saved_layout.setContentsMargins(12, 0, 12, 0)
        self._saved_layout.setAlignment(Qt.AlignTop)

        self._saved_header = QLabel("ZAPISANE (0)")
        self._saved_header.setStyleSheet(f"""
            font-size: 12px; font-weight: 600; color: {T.TEXT_DIM};
            text-transform: uppercase; margin-top: 12px; margin-bottom: 8px;
        """)
        self._saved_layout.addWidget(self._saved_header)

        self._saved_layout.addStretch()
        scroll.setWidget(self._saved_widget)
        layout.addWidget(scroll, stretch=1)

        # Export all
        footer = QWidget()
        footer.setStyleSheet(f"border-top: 1px solid {T.BORDER};")
        ft_layout = QVBoxLayout(footer)
        ft_layout.setContentsMargins(10, 10, 10, 10)
        btn_export = make_action_btn("Eksportuj wszystkie adnotacje")
        ft_layout.addWidget(btn_export)
        layout.addWidget(footer)

    def set_form_region(self, lead: str, t1: float, t2: float):
        self._region_label.setText(f"{lead}: {t1:.2f} s — {t2:.2f} s")

    def set_annotations(self, annotations_list: list[dict]):
        self._annotations = list(annotations_list)
        self._rebuild_saved_cards()

    def _on_save(self):
        cat = self.category.currentText()
        note = self.note_edit.toPlainText()
        self.annotation_saved.emit(cat, note)

    def _rebuild_saved_cards(self):
        # Remove all widgets except the header and final stretch
        while self._saved_layout.count() > 0:
            item = self._saved_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Re-add header
        self._saved_header = QLabel(f"ZAPISANE ({len(self._annotations)})")
        self._saved_header.setStyleSheet(f"""
            font-size: 12px; font-weight: 600; color: {T.TEXT_DIM};
            text-transform: uppercase; margin-top: 12px; margin-bottom: 8px;
        """)
        self._saved_layout.addWidget(self._saved_header)

        for idx, ann in enumerate(self._annotations):
            cat = ann.get("category", "")
            border_color, bg_c, text_c = self._BADGE_STYLES.get(
                cat, (T.TEXT_DIM, T.TAG_BG, T.TEXT_DIM)
            )
            meta_text = f"{ann['lead']}: {ann['t1']:.2f} — {ann['t2']:.2f} s"

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

            # Meta row: lead + time, badge, delete button
            meta_row = QHBoxLayout()
            meta_lbl = QLabel(meta_text)
            meta_lbl.setStyleSheet(f"font-size: 11px; font-family: Menlo; color: {T.TEXT_MUTED};")
            meta_row.addWidget(meta_lbl)
            badge_lbl = QLabel(cat)
            badge_lbl.setStyleSheet(f"""
                font-size: 10px; padding: 1px 6px; border-radius: 3px;
                font-weight: 600; background: {bg_c}; color: {text_c};
            """)
            meta_row.addWidget(badge_lbl)

            # Source badge: ANN (doctor) vs ANN.INF (AI model)
            source = ann.get("source", "user")
            if source == "model":
                src_lbl = QLabel("AI")
                src_lbl.setStyleSheet(f"""
                    font-size: 9px; padding: 1px 5px; border-radius: 3px;
                    font-weight: 700; background: {T.PURPLE_BG}; color: {T.PURPLE};
                """)
            else:
                src_lbl = QLabel("Lekarz")
                src_lbl.setStyleSheet(f"""
                    font-size: 9px; padding: 1px 5px; border-radius: 3px;
                    font-weight: 700; background: {T.BLUE_BG}; color: {T.BADGE_BLUE_TEXT};
                """)
            meta_row.addWidget(src_lbl)
            meta_row.addStretch()

            del_btn = QPushButton("✕")
            del_btn.setFixedSize(20, 20)
            del_btn.setCursor(Qt.PointingHandCursor)
            del_btn.setStyleSheet(f"""
                QPushButton {{
                    font-size: 12px; border: none; background: transparent;
                    color: {T.TEXT_DIM}; font-weight: 600;
                }}
                QPushButton:hover {{ color: {T.RED}; }}
            """)
            del_btn.clicked.connect(lambda checked, i=idx: self._on_delete(i))
            meta_row.addWidget(del_btn)
            card_layout.addLayout(meta_row)

            note_text = ann.get("note", "")
            if note_text:
                text_lbl = QLabel(note_text)
                text_lbl.setStyleSheet(f"font-size: 12px; color: {T.TEXT_SECONDARY};")
                text_lbl.setWordWrap(True)
                card_layout.addWidget(text_lbl)

            self._saved_layout.addWidget(card)

        self._saved_layout.addStretch()

    def _on_delete(self, index):
        if 0 <= index < len(self._annotations):
            self._annotations.pop(index)
            self._rebuild_saved_cards()
            self.annotation_deleted.emit(index)
            self.annotations_changed.emit()

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

        self._paused = False

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

    def _lead_btn_style(self, active):
        if active:
            return f"""
                QPushButton {{
                    font-size: 11px; padding: 4px 8px; border: 1px solid {T.ACCENT};
                    border-radius: 4px; background: {T.ICON_BG}; color: {T.ACCENT};
                    font-family: Menlo; font-weight: 600;
                }}
                QPushButton:hover {{ background: {T.ACCENT}; color: {T.ACCENT_TEXT}; }}
            """
        return f"""
            QPushButton {{
                font-size: 11px; padding: 4px 8px; border: 1px solid {T.BORDER};
                border-radius: 4px; background: {T.WHITE}; color: {T.TEXT_DIM};
                font-family: Menlo; font-weight: 600;
            }}
            QPushButton:hover {{ background: {T.BG_SECONDARY}; color: {T.TEXT}; }}
        """

    def _toggle_lead(self, btn, lead):
        is_active = not btn.property("active")
        btn.setProperty("active", is_active)
        btn.setStyleSheet(self._lead_btn_style(is_active))
        active_leads = [ld for ld, b in self.lead_btns.items() if b.property("active")]
        self.leads_changed.emit(active_leads)

    def _pill_style(self, active):
        if active:
            return f"""
                QPushButton {{
                    font-size: 11px; padding: 4px 10px; border: 1px solid {T.BORDER};
                    border-radius: 4px; background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                }}
                QPushButton:hover {{ background: {T.GREEN}; }}
            """
        return f"""
            QPushButton {{
                font-size: 11px; padding: 4px 10px; border: 1px solid {T.BORDER};
                border-radius: 4px; background: {T.WHITE}; color: {T.TEXT_MUTED};
            }}
            QPushButton:hover {{ background: {T.BG_SECONDARY}; color: {T.TEXT}; }}
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
