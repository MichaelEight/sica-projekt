"""Unified markings panel — replaces CaliperPanel, AnnotationPanel, ResultsPanel."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QSizePolicy,
)

import ui.theme as T


# ---------------------------------------------------------------------------
# Filter pills widget
# ---------------------------------------------------------------------------

class _FilterPills(QWidget):
    """Row of toggle pill buttons for filtering marking types."""

    filter_changed = Signal()

    _PILL_KEYS = ("all", "measures", "annotations", "ai")
    _PILL_LABELS = {
        "all": "Wszystkie",
        "measures": "Pomiary",
        "annotations": "Adnotacje",
        "ai": "AI",
    }
    # Which marking types belong to each filter group
    _GROUP_TYPES = {
        "measures": {"pr", "qrs", "qt", "rr", "custom"},
        "annotations": {"annotation"},
        "ai": {"scan"},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons: dict[str, QPushButton] = {}
        self._active: dict[str, bool] = {k: True for k in self._PILL_KEYS}

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        for key in self._PILL_KEYS:
            btn = QPushButton(self._PILL_LABELS[key])
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(24)
            btn.clicked.connect(lambda checked=False, k=key: self._toggle(k))
            lay.addWidget(btn)
            self._buttons[key] = btn

        lay.addStretch()
        self.apply_theme()

    # -- public helpers --

    def accepts(self, marking_type: str) -> bool:
        """Return True if the current filter allows this marking type."""
        for group, types in self._GROUP_TYPES.items():
            if marking_type in types:
                return self._active[group]
        return True  # unknown types always shown

    def apply_theme(self):
        for key, btn in self._buttons.items():
            self._style_pill(btn, self._active[key])

    # -- internals --

    def _toggle(self, key: str):
        if key == "all":
            new_state = not self._active["all"]
            for k in self._PILL_KEYS:
                self._active[k] = new_state
        else:
            self._active[key] = not self._active[key]
            # sync "all" automatically
            others = [self._active[k] for k in self._PILL_KEYS if k != "all"]
            self._active["all"] = all(others)

        self.apply_theme()
        self.filter_changed.emit()

    @staticmethod
    def _style_pill(btn: QPushButton, active: bool):
        if active:
            bg, fg = T.ACCENT, T.ACCENT_TEXT
        else:
            bg, fg = T.BG_SECONDARY, T.TEXT_MUTED
        btn.setStyleSheet(
            f"QPushButton {{ border: none; background: {bg}; color: {fg};"
            f"  font-size: 11px; font-family: 'Helvetica Neue';"
            f"  padding: 4px 12px; border-radius: 12px; }}"
        )


# ---------------------------------------------------------------------------
# Single marking card
# ---------------------------------------------------------------------------

class _MarkingCard(QFrame):
    """Compact card representing one marking."""

    clicked = Signal(str)
    hovered = Signal(str)
    unhovered = Signal()
    delete_clicked = Signal(str)

    def __init__(self, marking, parent=None):
        super().__init__(parent)
        self.marking = marking
        self._selected = False

        self.setCursor(Qt.PointingHandCursor)

        # -- build layout --
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 10, 8)
        root.setSpacing(3)

        # Row 1: dot + lead:time + delete
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        row1.addWidget(self._dot, 0, Qt.AlignVCenter)

        t1 = getattr(marking, "t1", 0.0) or 0.0
        t2 = getattr(marking, "t2", 0.0) or 0.0
        lead = getattr(marking, "lead", "") or ""
        self._meta_label = QLabel(f"{lead}: {t1:.2f} \u2014 {t2:.2f} s")
        row1.addWidget(self._meta_label, 1)

        self._del_btn = QPushButton("\u2715")
        self._del_btn.setFixedSize(20, 20)
        self._del_btn.setCursor(Qt.PointingHandCursor)
        self._del_btn.clicked.connect(lambda: self.delete_clicked.emit(marking.id))
        row1.addWidget(self._del_btn, 0, Qt.AlignVCenter)

        root.addLayout(row1)

        # Row 2: label + source badge
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        label_text = getattr(marking, "label", "") or marking.type
        self._label = QLabel(label_text)
        row2.addWidget(self._label, 1)

        source = getattr(marking, "source", "") or ""
        if source:
            self._badge = QLabel(source)
            self._badge.setFixedHeight(18)
            self._badge.setAlignment(Qt.AlignCenter)
            row2.addWidget(self._badge, 0)
        else:
            self._badge = None

        root.addLayout(row2)

        # Row 3 (annotations only): category + note
        category = getattr(marking, "category", "") or ""
        note = getattr(marking, "note", "") or ""
        self._cat_badge = None
        self._note_label = None

        if marking.type == "annotation" and (category or note):
            row3 = QHBoxLayout()
            row3.setSpacing(6)
            if category:
                self._cat_badge = QLabel(category)
                self._cat_badge.setFixedHeight(18)
                self._cat_badge.setAlignment(Qt.AlignCenter)
                row3.addWidget(self._cat_badge, 0)
            if note:
                self._note_label = QLabel(note)
                self._note_label.setWordWrap(True)
                row3.addWidget(self._note_label, 1)
            else:
                row3.addStretch()
            root.addLayout(row3)

        self.apply_theme()

    # -- public --

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_border()

    def apply_theme(self):
        color = self._type_color()
        # dot
        self._dot.setStyleSheet(
            f"border: none; background: {color}; border-radius: 4px;"
        )
        # meta label (monospace)
        self._meta_label.setStyleSheet(
            f"border: none; color: {T.TEXT_MUTED}; font-family: Menlo;"
            f"  font-size: 12px;"
        )
        # delete button
        self._del_btn.setStyleSheet(
            f"QPushButton {{ border: none; color: {T.TEXT_DIM};"
            f"  font-size: 13px; background: transparent; }}"
            f"QPushButton:hover {{ color: {T.RED}; }}"
        )
        # label
        self._label.setStyleSheet(
            f"border: none; color: {color}; font-size: 13px;"
            f"  font-weight: 600; font-family: 'Helvetica Neue';"
        )
        # source badge
        if self._badge:
            self._badge.setStyleSheet(
                f"border: none; background: {T.TAG_BG}; color: {T.TEXT_MUTED};"
                f"  font-size: 10px; font-family: 'Helvetica Neue';"
                f"  padding: 1px 8px; border-radius: 8px;"
            )
        # category badge
        if self._cat_badge:
            self._cat_badge.setStyleSheet(
                f"border: none; background: {T.PURPLE_BG}; color: {T.PURPLE};"
                f"  font-size: 10px; font-family: 'Helvetica Neue';"
                f"  padding: 1px 8px; border-radius: 8px;"
            )
        # note label
        if self._note_label:
            self._note_label.setStyleSheet(
                f"border: none; color: {T.TEXT_DIM}; font-size: 11px;"
                f"  font-family: 'Helvetica Neue';"
            )
        self._apply_border()

    # -- internals --

    def _type_color(self) -> str:
        m = {
            "annotation": T.PURPLE,
            "pr": T.AMBER_TEXT if T.is_dark_mode() else "#f97316",
            "qrs": T.RED,
            "qt": T.GREEN,
            "rr": T.ACCENT,
            "custom": T.TEXT_DIM,
            "scan": T.AMBER_TEXT,
        }
        return m.get(self.marking.type, T.TEXT_DIM)

    def _apply_border(self):
        color = self._type_color()
        if self._selected:
            bg = T.BLUE_BG
            border_color = T.ACCENT
        else:
            bg = T.BG_SECONDARY
            border_color = "transparent"
        self.setStyleSheet(
            f"_MarkingCard {{"
            f"  border-left: 3px solid {color};"
            f"  border-top: 1px solid {border_color};"
            f"  border-right: 1px solid {border_color};"
            f"  border-bottom: 1px solid {border_color};"
            f"  border-radius: 6px;"
            f"  background: {bg};"
            f"}}"
            f"_MarkingCard:hover {{"
            f"  border-left: 3px solid {color};"
            f"  border-top: 1px solid {T.BORDER};"
            f"  border-right: 1px solid {T.BORDER};"
            f"  border-bottom: 1px solid {T.BORDER};"
            f"  background: {bg};"
            f"}}"
        )

    # -- events --

    def mousePressEvent(self, event):
        self.clicked.emit(self.marking.id)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.hovered.emit(self.marking.id)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unhovered.emit()
        super().leaveEvent(event)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class MarkingsPanel(QWidget):
    """Unified markings side-panel with list + annotation form views."""

    marking_hovered = Signal(str)
    marking_unhovered = Signal()
    marking_selected = Signal(str)
    marking_deleted = Signal(str)
    marking_edited = Signal(str, str, str)  # id, field, new_value
    annotation_created = Signal(str, str, str, float, float)  # lead, category, note, t1, t2
    marking_focus = Signal(str)  # id — scroll canvas to show this marking
    undo_requested = Signal()
    redo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._markings: list = []
        self._cards: list[_MarkingCard] = []
        self._selected_id: str | None = None
        self._current_lead: str = ""
        self._lead_filter_active: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(0)

        # --- 1. Header row ---
        header = QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet("border: none;")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(0, 0, 0, 0)
        h_lay.setSpacing(6)

        self._title = QLabel("Oznaczenia")
        h_lay.addWidget(self._title, 1)

        self._undo_btn = QPushButton("Cofnij")
        self._undo_btn.setFixedHeight(26)
        self._undo_btn.setCursor(Qt.PointingHandCursor)
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        h_lay.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("Ponów")
        self._redo_btn.setFixedHeight(26)
        self._redo_btn.setCursor(Qt.PointingHandCursor)
        self._redo_btn.clicked.connect(self.redo_requested.emit)
        h_lay.addWidget(self._redo_btn)

        root.addWidget(header)
        root.addWidget(self._separator())

        # --- 2. Lead scope toggle (two-button segmented control) ---
        scope_row = QHBoxLayout()
        scope_row.setContentsMargins(0, 4, 0, 4)
        scope_row.setSpacing(0)

        self._scope_all_btn = QPushButton("Wszystkie")
        self._scope_all_btn.setFixedHeight(26)
        self._scope_all_btn.setCursor(Qt.PointingHandCursor)
        self._scope_all_btn.clicked.connect(lambda: self._set_lead_filter(False))
        scope_row.addWidget(self._scope_all_btn)

        self._scope_lead_btn = QPushButton("—")
        self._scope_lead_btn.setFixedHeight(26)
        self._scope_lead_btn.setCursor(Qt.PointingHandCursor)
        self._scope_lead_btn.clicked.connect(lambda: self._set_lead_filter(True))
        scope_row.addWidget(self._scope_lead_btn)

        scope_row.addStretch()
        root.addLayout(scope_row)

        # --- 3. Filter pills ---
        self._pills = _FilterPills()
        self._pills.filter_changed.connect(self._rebuild_visible)
        root.addWidget(self._pills)
        root.addSpacing(4)
        root.addWidget(self._separator())
        root.addSpacing(4)

        # --- 4. Search bar ---
        self._search = QLineEdit()
        self._search.setPlaceholderText("Szukaj...")
        self._search.setFixedHeight(28)
        self._search.textChanged.connect(self._rebuild_visible)
        root.addWidget(self._search)
        root.addSpacing(4)

        # --- 5. Stacked content: list view vs annotation form ---
        root.addWidget(self._separator())
        root.addSpacing(4)

        from PySide6.QtWidgets import QStackedWidget, QComboBox, QTextEdit
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {T.BG_SECONDARY}; border: none;")

        # Page 0: card list
        list_page = QWidget()
        list_page.setAutoFillBackground(True)
        list_page.setStyleSheet(f"background: {T.BG_SECONDARY}; border: none;")
        list_lay = QVBoxLayout(list_page)
        list_lay.setContentsMargins(0, 0, 0, 0)
        list_lay.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 4, 0, 4)
        self._card_layout.setSpacing(4)
        self._card_layout.addStretch()
        self._scroll.setWidget(self._card_container)
        list_lay.addWidget(self._scroll, 1)

        self._stack.addWidget(list_page)  # index 0

        # Page 1: annotation form
        form_page = QWidget()
        form_page.setAutoFillBackground(True)
        form_page.setStyleSheet(f"background: {T.BG_SECONDARY}; border: none;")
        form_lay = QVBoxLayout(form_page)
        form_lay.setContentsMargins(0, 10, 0, 10)
        form_lay.setSpacing(8)

        form_title = QLabel("Nowa adnotacja")
        form_title.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {T.ACCENT}; border: none;")
        form_lay.addWidget(form_title)

        self._form_region = QLabel("—")
        self._form_region.setStyleSheet(
            f"font-size: 12px; font-family: Menlo; color: {T.TEXT_SECONDARY};"
            f"background: {T.BLUE_BG}; padding: 6px 10px; border-radius: 4px; border: none;"
        )
        form_lay.addWidget(self._form_region)

        cat_lbl = QLabel("Kategoria")
        cat_lbl.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; border: none;")
        form_lay.addWidget(cat_lbl)

        from PySide6.QtWidgets import QStyledItemDelegate
        self._form_category = QComboBox()
        self._form_category.addItems(["Patologia", "Norma", "Artefakt", "Do weryfikacji"])
        self._form_category.setItemDelegate(QStyledItemDelegate(self._form_category))
        self._form_category.setStyleSheet(
            f"QComboBox {{ combobox-popup: 0; padding: 6px 10px; border: 1px solid {T.BORDER};"
            f"  border-radius: 6px; font-size: 13px; background: {T.WHITE}; color: {T.TEXT}; }}"
            f"QComboBox:hover {{ border-color: {T.ACCENT}; }}"
            f"QComboBox::drop-down {{ border: none; width: 20px; }}"
            f"QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;"
            f"  border-right: 4px solid transparent; border-top: 5px solid {T.TEXT_MUTED}; margin-right: 8px; }}"
            f"QComboBox QAbstractItemView {{ background: {T.WHITE}; color: {T.TEXT};"
            f"  border: 1px solid {T.BORDER}; padding: 4px; font-size: 13px; outline: none;"
            f"  selection-background-color: {T.BLUE_BG}; selection-color: {T.ACCENT}; }}"
        )
        form_lay.addWidget(self._form_category)

        note_lbl = QLabel("Notatka")
        note_lbl.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; border: none;")
        form_lay.addWidget(note_lbl)

        self._form_note = QTextEdit()
        self._form_note.setPlaceholderText("Opis adnotacji...")
        self._form_note.setMaximumHeight(80)
        self._form_note.setStyleSheet(
            f"QTextEdit {{ padding: 6px 8px; border: 1px solid {T.BORDER}; border-radius: 6px;"
            f"  font-size: 13px; background: {T.WHITE}; color: {T.TEXT}; }}"
            f"QTextEdit:focus {{ border-color: {T.ACCENT}; }}"
        )
        form_lay.addWidget(self._form_note)

        from ui.widgets import make_action_btn
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_save = make_action_btn("Zapisz", primary=True)
        btn_save.clicked.connect(self._on_form_save)
        btn_row.addWidget(btn_save)
        btn_cancel = make_action_btn("Anuluj")
        btn_cancel.clicked.connect(self._on_form_cancel)
        btn_row.addWidget(btn_cancel)
        form_lay.addLayout(btn_row)
        form_lay.addStretch()

        self._stack.addWidget(form_page)  # index 1

        # Page 2: edit form (for existing markings)
        edit_page = QWidget()
        edit_page.setAutoFillBackground(True)
        edit_page.setStyleSheet(f"background: {T.BG_SECONDARY}; border: none;")
        edit_lay = QVBoxLayout(edit_page)
        edit_lay.setContentsMargins(0, 10, 0, 10)
        edit_lay.setSpacing(8)

        edit_title = QLabel("Edytuj oznaczenie")
        edit_title.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {T.ACCENT}; border: none;")
        edit_lay.addWidget(edit_title)

        self._edit_region = QLabel("—")
        self._edit_region.setStyleSheet(
            f"font-size: 12px; font-family: Menlo; color: {T.TEXT_SECONDARY};"
            f"background: {T.BLUE_BG}; padding: 6px 10px; border-radius: 4px; border: none;"
        )
        edit_lay.addWidget(self._edit_region)

        self._edit_cat_label = QLabel("Kategoria")
        self._edit_cat_label.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; border: none;")
        edit_lay.addWidget(self._edit_cat_label)

        self._edit_category = QComboBox()
        self._edit_category.addItems(["Patologia", "Norma", "Artefakt", "Do weryfikacji"])
        self._edit_category.setItemDelegate(QStyledItemDelegate(self._edit_category))
        self._edit_category.setStyleSheet(self._form_category.styleSheet())
        edit_lay.addWidget(self._edit_category)

        self._edit_note_label = QLabel("Notatka")
        self._edit_note_label.setStyleSheet(f"font-size: 12px; color: {T.TEXT_MUTED}; border: none;")
        edit_lay.addWidget(self._edit_note_label)

        self._edit_note = QTextEdit()
        self._edit_note.setPlaceholderText("Opis adnotacji...")
        self._edit_note.setMaximumHeight(80)
        self._edit_note.setStyleSheet(self._form_note.styleSheet())
        edit_lay.addWidget(self._edit_note)

        # Action buttons — row 1: save + focus
        edit_row1 = QHBoxLayout()
        edit_row1.setSpacing(6)
        btn_edit_save = make_action_btn("Zapisz zmiany", primary=True)
        btn_edit_save.clicked.connect(self._on_edit_save)
        edit_row1.addWidget(btn_edit_save)
        btn_edit_focus = QPushButton("Pokaż na wykresie")
        btn_edit_focus.setCursor(Qt.PointingHandCursor)
        btn_edit_focus.setStyleSheet(
            f"QPushButton {{ padding: 8px; border-radius: 6px; border: 1px solid {T.ACCENT};"
            f"  background: {T.WHITE}; color: {T.ACCENT}; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {T.BLUE_BG}; }}"
        )
        btn_edit_focus.clicked.connect(self._on_edit_focus)
        edit_row1.addWidget(btn_edit_focus)
        edit_lay.addLayout(edit_row1)

        # Action buttons — row 2: delete + back
        edit_row2 = QHBoxLayout()
        edit_row2.setSpacing(6)
        btn_edit_del = QPushButton("Usuń")
        btn_edit_del.setCursor(Qt.PointingHandCursor)
        btn_edit_del.setStyleSheet(
            f"QPushButton {{ padding: 8px; border-radius: 6px; border: 1px solid {T.RED};"
            f"  background: {T.WHITE}; color: {T.RED}; font-size: 12px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {T.RED}; color: {T.ACCENT_TEXT}; }}"
        )
        btn_edit_del.clicked.connect(self._on_edit_delete)
        edit_row2.addWidget(btn_edit_del)
        btn_edit_back = make_action_btn("Powrót")
        btn_edit_back.clicked.connect(self._on_edit_back)
        edit_row2.addWidget(btn_edit_back)
        edit_lay.addLayout(edit_row2)
        edit_lay.addStretch()

        self._stack.addWidget(edit_page)  # index 2

        root.addWidget(self._stack, 1)

        self._undo_enabled = False
        self._redo_enabled = False
        self._form_lead = ""
        self._form_t1 = 0.0
        self._form_t2 = 0.0
        self._editing_id = None

        self.apply_theme()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def set_markings(self, markings: list):
        self._markings = list(markings)
        self._rebuild_visible()

    def set_current_lead(self, lead: str):
        self._current_lead = lead
        self._style_scope_buttons()
        if self._lead_filter_active:
            self._rebuild_visible()

    def set_selected(self, marking_id: str | None):
        self._selected_id = marking_id
        for card in self._cards:
            card.set_selected(card.marking.id == marking_id)

    def show_annotation_form(self, lead: str, t1: float, t2: float):
        """Switch to annotation creation form."""
        self._form_lead = lead
        self._form_t1 = t1
        self._form_t2 = t2
        self._form_region.setText(f"{lead}: {t1:.2f} — {t2:.2f} s")
        self._form_category.setCurrentIndex(0)
        self._form_note.clear()
        self._stack.setCurrentIndex(1)

    def show_edit_form(self, marking):
        """Switch to edit form for an existing marking."""
        self._editing_id = marking.id
        self._edit_region.setText(f"{marking.lead}: {marking.t1:.2f} — {marking.t2:.2f} s")
        if marking.type == "annotation":
            self._edit_cat_label.show()
            self._edit_category.show()
            self._edit_category.setCurrentText(getattr(marking, "category", "Patologia"))
        else:
            self._edit_cat_label.hide()
            self._edit_category.hide()
        self._edit_note.setPlainText(getattr(marking, "note", ""))
        self._stack.setCurrentIndex(2)

    def show_list(self):
        self._stack.setCurrentIndex(0)

    def set_undo_enabled(self, enabled: bool):
        self._undo_enabled = enabled
        self._style_undo_redo()

    def set_redo_enabled(self, enabled: bool):
        self._redo_enabled = enabled
        self._style_undo_redo()

    def apply_theme(self):
        bg = T.BG_SECONDARY
        self.setStyleSheet(
            f"MarkingsPanel {{ background: {bg};"
            f"  border-left: 1px solid {T.BORDER};"
            f"  border-top: none; border-right: none; border-bottom: none; }}"
        )
        self._title.setStyleSheet(
            f"border: none; color: {T.TEXT}; font-size: 14px;"
            f"  font-weight: 700; font-family: 'Helvetica Neue';"
        )
        self._style_undo_redo()
        self._style_scope_buttons()
        self._search.setStyleSheet(
            f"QLineEdit {{ border: 1px solid {T.BORDER}; border-radius: 6px;"
            f"  background: {T.WHITE}; color: {T.TEXT}; font-size: 12px;"
            f"  font-family: 'Helvetica Neue'; padding: 0 8px; }}"
            f"QLineEdit::placeholder {{ color: {T.TEXT_DIM}; }}"
        )
        bg = T.BG_SECONDARY
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {bg}; }}"
            f"QScrollBar:vertical {{ width: 6px; background: transparent; }}"
            f"QScrollBar::handle:vertical {{ background: {T.BORDER}; border-radius: 3px; min-height: 20px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )
        self._card_container.setStyleSheet(f"background: {bg}; border: none;")
        self._stack.setStyleSheet(f"background: {bg}; border: none;")
        self._pills.apply_theme()
        for card in self._cards:
            card.apply_theme()

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _separator(self) -> QFrame:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {T.BORDER}; border: none;")
        return sep

    def _set_lead_filter(self, active: bool):
        self._lead_filter_active = active
        self._style_scope_buttons()
        self._rebuild_visible()

    def _style_scope_buttons(self):
        lead = self._current_lead or "—"
        self._scope_lead_btn.setText(f"Tylko {lead}")

        active_style = (
            f"QPushButton {{ border: 1px solid {T.ACCENT}; background: {T.ACCENT}; color: {T.ACCENT_TEXT};"
            f"  font-size: 11px; padding: 4px 14px; border-radius: 4px; font-weight: 600; }}"
        )
        inactive_style = (
            f"QPushButton {{ border: 1px solid {T.BORDER}; background: {T.WHITE}; color: {T.TEXT_MUTED};"
            f"  font-size: 11px; padding: 4px 14px; border-radius: 4px; }}"
            f"QPushButton:hover {{ background: {T.BG_SECONDARY}; color: {T.TEXT}; }}"
        )

        if self._lead_filter_active:
            self._scope_all_btn.setStyleSheet(inactive_style)
            self._scope_lead_btn.setStyleSheet(active_style)
        else:
            self._scope_all_btn.setStyleSheet(active_style)
            self._scope_lead_btn.setStyleSheet(inactive_style)

    def _style_undo_redo(self):
        for btn, enabled in ((self._undo_btn, self._undo_enabled), (self._redo_btn, self._redo_enabled)):
            if enabled:
                btn.setEnabled(True)
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid {T.BORDER}; background: {T.WHITE};"
                    f"  color: {T.TEXT}; font-size: 11px; border-radius: 4px; padding: 2px 8px; }}"
                    f"QPushButton:hover {{ background: {T.BG_SECONDARY}; border-color: {T.ACCENT}; color: {T.ACCENT}; }}"
                )
            else:
                btn.setEnabled(False)
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid {T.BORDER_LIGHT}; background: transparent;"
                    f"  color: {T.TEXT_DIM}; font-size: 11px; border-radius: 4px; padding: 2px 8px; }}"
                )

    def _rebuild_visible(self):
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        while self._card_layout.count():
            item = self._card_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        query = self._search.text().strip().lower()
        for m in self._markings:
            if not self._pills.accepts(m.type):
                continue
            if self._lead_filter_active and self._current_lead:
                if getattr(m, "lead", "") != self._current_lead:
                    continue
            if query:
                haystack = " ".join([
                    getattr(m, "label", "") or "", getattr(m, "lead", "") or "",
                    getattr(m, "category", "") or "", getattr(m, "note", "") or "", m.type,
                ]).lower()
                if query not in haystack:
                    continue

            card = _MarkingCard(m)
            card.clicked.connect(self._on_card_clicked)
            card.hovered.connect(self.marking_hovered.emit)
            card.unhovered.connect(self.marking_unhovered.emit)
            card.delete_clicked.connect(self.marking_deleted.emit)
            if self._selected_id and m.id == self._selected_id:
                card.set_selected(True)
            self._card_layout.addWidget(card)
            self._cards.append(card)
        self._card_layout.addStretch()

    def _on_card_clicked(self, marking_id: str):
        self._selected_id = marking_id
        for card in self._cards:
            card.set_selected(card.marking.id == marking_id)
        self.marking_selected.emit(marking_id)

        # Open edit form for the clicked marking
        for m in self._markings:
            if m.id == marking_id:
                self.show_edit_form(m)
                break

    # --- form handlers ---

    def _on_form_save(self):
        cat = self._form_category.currentText()
        note = self._form_note.toPlainText()
        self.annotation_created.emit(self._form_lead, cat, note, self._form_t1, self._form_t2)
        self._form_note.clear()
        self._stack.setCurrentIndex(0)

    def _on_form_cancel(self):
        self._form_note.clear()
        self._stack.setCurrentIndex(0)

    def _on_edit_focus(self):
        if self._editing_id:
            self.marking_focus.emit(self._editing_id)

    def _on_edit_save(self):
        if self._editing_id:
            cat = self._edit_category.currentText()
            note = self._edit_note.toPlainText()
            self.marking_edited.emit(self._editing_id, "category", cat)
            self.marking_edited.emit(self._editing_id, "note", note)
        self._stack.setCurrentIndex(0)
        self._editing_id = None

    def _on_edit_delete(self):
        if self._editing_id:
            self.marking_deleted.emit(self._editing_id)
        self._stack.setCurrentIndex(0)
        self._editing_id = None

    def _on_edit_back(self):
        self._stack.setCurrentIndex(0)
        self._editing_id = None
