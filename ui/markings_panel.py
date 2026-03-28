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

        self.setFixedHeight(self._desired_height(marking))
        self.setCursor(Qt.PointingHandCursor)

        # -- build layout --
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 7, 8, 7)
        root.setSpacing(2)

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
        self._meta_label.setFixedHeight(16)
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
        self._label.setFixedHeight(18)
        row2.addWidget(self._label, 1)

        source = getattr(marking, "source", "") or ""
        if source:
            self._badge = QLabel(source)
            self._badge.setFixedHeight(16)
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
                self._cat_badge.setFixedHeight(16)
                self._cat_badge.setAlignment(Qt.AlignCenter)
                row3.addWidget(self._cat_badge, 0)
            if note:
                self._note_label = QLabel(note)
                self._note_label.setWordWrap(True)
                self._note_label.setFixedHeight(14)
                row3.addWidget(self._note_label, 1)
            else:
                row3.addStretch()
            root.addLayout(row3)

        self.apply_theme()

    # -- geometry helper --
    @staticmethod
    def _desired_height(marking) -> int:
        base = 58
        category = getattr(marking, "category", "") or ""
        note = getattr(marking, "note", "") or ""
        if marking.type == "annotation" and (category or note):
            base += 20
        return base

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
    """Unified markings side-panel (280px wide)."""

    marking_hovered = Signal(str)
    marking_unhovered = Signal()
    marking_selected = Signal(str)
    marking_deleted = Signal(str)
    marking_edited = Signal(str, str, str)
    undo_requested = Signal()
    redo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._markings: list = []
        self._cards: list[_MarkingCard] = []
        self._selected_id: str | None = None

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

        # separator
        root.addWidget(self._separator())

        # --- 2. Filter pills ---
        self._pills = _FilterPills()
        self._pills.filter_changed.connect(self._rebuild_visible)
        root.addWidget(self._pills)
        root.addSpacing(6)

        # separator
        root.addWidget(self._separator())
        root.addSpacing(6)

        # --- 3. Search bar ---
        self._search = QLineEdit()
        self._search.setPlaceholderText("Szukaj...")
        self._search.setFixedHeight(30)
        self._search.textChanged.connect(self._rebuild_visible)
        root.addWidget(self._search)
        root.addSpacing(6)

        # --- 4. Scroll area ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(3)
        self._card_layout.addStretch()

        self._scroll.setWidget(self._card_container)
        root.addWidget(self._scroll, 1)

        # initial undo/redo state
        self._undo_enabled = False
        self._redo_enabled = False

        self.apply_theme()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def set_markings(self, markings: list):
        """Replace all cards with a new marking list."""
        self._markings = list(markings)
        self._rebuild_visible()

    def set_selected(self, marking_id: str | None):
        """Programmatically select a card (or deselect with None)."""
        self._selected_id = marking_id
        for card in self._cards:
            card.set_selected(card.marking.id == marking_id)

    def set_undo_enabled(self, enabled: bool):
        self._undo_enabled = enabled
        self._style_undo_redo()

    def set_redo_enabled(self, enabled: bool):
        self._redo_enabled = enabled
        self._style_undo_redo()

    def apply_theme(self):
        """Reapply all styles for theme switch."""
        self.setStyleSheet(f"background: {T.WHITE}; border: none;")

        # title
        self._title.setStyleSheet(
            f"border: none; color: {T.TEXT}; font-size: 14px;"
            f"  font-weight: 700; font-family: 'Helvetica Neue';"
        )

        self._style_undo_redo()

        # search
        self._search.setStyleSheet(
            f"QLineEdit {{ border: 1px solid {T.BORDER}; border-radius: 6px;"
            f"  background: {T.WHITE}; color: {T.TEXT}; font-size: 12px;"
            f"  font-family: 'Helvetica Neue'; padding: 0 8px; }}"
            f"QLineEdit::placeholder {{ color: {T.TEXT_DIM}; }}"
        )

        # scroll area
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {T.WHITE}; }}"
        )
        self._card_container.setStyleSheet(
            f"border: none; background: {T.WHITE};"
        )

        # pills
        self._pills.apply_theme()

        # cards
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

    def _style_undo_redo(self):
        for btn, enabled in (
            (self._undo_btn, self._undo_enabled),
            (self._redo_btn, self._redo_enabled),
        ):
            if enabled:
                btn.setEnabled(True)
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid {T.BORDER}; background: {T.WHITE};"
                    f"  color: {T.TEXT}; font-size: 11px; border-radius: 4px;"
                    f"  padding: 2px 8px; }}"
                    f"QPushButton:hover {{ background: {T.BG_SECONDARY};"
                    f"  border-color: {T.ACCENT}; color: {T.ACCENT}; }}"
                )
            else:
                btn.setEnabled(False)
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid {T.BORDER_LIGHT}; background: transparent;"
                    f"  color: {T.TEXT_DIM}; font-size: 11px; border-radius: 4px;"
                    f"  padding: 2px 8px; }}"
                )

    def _rebuild_visible(self):
        """Clear and recreate cards based on current filter + search."""
        # remove old cards
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        # remove stretch
        while self._card_layout.count():
            item = self._card_layout.takeAt(0)
            # stretch items have no widget
            if item.widget():
                item.widget().setParent(None)

        query = self._search.text().strip().lower()

        for m in self._markings:
            # filter by pills
            if not self._pills.accepts(m.type):
                continue
            # filter by search text
            if query:
                haystack = " ".join([
                    getattr(m, "label", "") or "",
                    getattr(m, "lead", "") or "",
                    getattr(m, "category", "") or "",
                    getattr(m, "note", "") or "",
                    m.type,
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
