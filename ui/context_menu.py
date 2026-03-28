"""Theme-aware custom context menu for EKG selection workflow."""

from PySide6.QtCore import Qt, Signal, QPoint, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsDropShadowEffect,
)

import ui.theme as T


class _MenuItem(QWidget):
    """A single clickable row inside the context menu."""

    clicked = Signal()
    hovered = Signal()
    unhovered = Signal()

    def __init__(
        self,
        label: str,
        right_text: str = "",
        has_submenu: bool = False,
        enabled: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._label = label
        self._right_text = right_text
        self._has_submenu = has_submenu
        self._enabled = enabled
        self._is_hovered = False

        self.setFixedHeight(36)
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
        self.setMouseTracking(True)

    # ── painting ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # hover background
        if self._is_hovered and self._enabled:
            p.setBrush(QColor(T.BG_SECONDARY))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(self.rect().adjusted(4, 2, -4, -2), 6, 6)

        # text color
        text_color = QColor(T.TEXT) if self._enabled else QColor(T.TEXT_DIM)
        p.setPen(text_color)
        p.setFont(self.font())

        # label
        label_rect = self.rect().adjusted(16, 0, -16, 0)
        p.drawText(label_rect, Qt.AlignLeft | Qt.AlignVCenter, self._label)

        # right side text / submenu arrow
        if self._has_submenu:
            p.drawText(label_rect, Qt.AlignRight | Qt.AlignVCenter, "\u25b8")
        elif self._right_text:
            dim_color = QColor(T.TEXT_DIM)
            p.setPen(dim_color)
            p.drawText(label_rect, Qt.AlignRight | Qt.AlignVCenter, self._right_text)

        p.end()

    # ── hover ────────────────────────────────────────────────────────

    def enterEvent(self, event):
        if self._enabled:
            self._is_hovered = True
            self.update()
            self.hovered.emit()

    def leaveEvent(self, event):
        self._is_hovered = False
        self.update()
        self.unhovered.emit()

    # ── click ────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._enabled and event.button() == Qt.LeftButton:
            self.clicked.emit()


class _Separator(QWidget):
    """Thin horizontal divider."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(9)

    def paintEvent(self, event):
        p = QPainter(self)
        pen = QPen(QColor(T.BORDER))
        pen.setWidthF(1.0)
        p.setPen(pen)
        y = self.height() // 2
        p.drawLine(12, y, self.width() - 12, y)
        p.end()


class _PopupPanel(QWidget):
    """Base rounded-corner popup panel with translucent background."""

    def __init__(self, width: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(width)

        # drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(shadow)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 6, 4, 6)
        self._layout.setSpacing(0)

    def add_item(self, item: QWidget):
        self._layout.addWidget(item)

    def finish_layout(self):
        """Call after all items are added to resize height properly."""
        self.adjustSize()

    # ── rounded background ───────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(2, 2, -2, -2), 10, 10)

        # background fill
        p.fillPath(path, QColor(T.WHITE))

        # border
        border_pen = QPen(QColor(T.BORDER))
        border_pen.setWidthF(1.0)
        p.setPen(border_pen)
        p.drawPath(path)

        p.end()


class SelectionContextMenu(QWidget):
    """Custom context menu for the EKG selection workflow.

    Usage::

        menu = SelectionContextMenu(parent)
        menu.action_selected.connect(on_action)
        menu.show_at(global_pos, selection_seconds=12.5)
    """

    action_selected = Signal(str)

    _ACTION_ANNOTATE = "annotate"
    _ACTION_SCAN = "scan"
    _ACTION_ZOOM = "zoom"
    _ACTION_EXPORT = "export_png"

    _MARK_ACTIONS = [
        ("PR", "mark_pr"),
        ("QRS", "mark_qrs"),
        ("QT", "mark_qt"),
        ("R-R", "mark_rr"),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.hide()

        self._panel: _PopupPanel | None = None
        self._submenu: _PopupPanel | None = None
        self._submenu_timer = QTimer(self)
        self._submenu_timer.setSingleShot(True)
        self._submenu_timer.setInterval(250)
        self._submenu_timer.timeout.connect(self._close_submenu)
        self._mark_item: _MenuItem | None = None

    # ── public API ───────────────────────────────────────────────────

    def show_at(self, pos: QPoint, selection_seconds: float):
        """Show the menu at *pos* (global coordinates).

        *selection_seconds* controls whether the scan action is available
        (requires >= 10 s).
        """
        self._cleanup()

        scan_enabled = selection_seconds >= 10.0
        panel = _PopupPanel(240)

        # -- Adnotacja
        item_annotate = _MenuItem("Adnotacja", parent=panel)
        item_annotate.clicked.connect(lambda: self._emit(self._ACTION_ANNOTATE))
        panel.add_item(item_annotate)

        # -- Oznacz...  (with submenu arrow)
        item_mark = _MenuItem("Oznacz\u2026", has_submenu=True, parent=panel)
        item_mark.hovered.connect(self._show_submenu)
        item_mark.unhovered.connect(self._schedule_close_submenu)
        panel.add_item(item_mark)
        self._mark_item = item_mark

        panel.add_item(_Separator(panel))

        # -- Skanuj
        right_hint = "" if scan_enabled else "min. 10 s"
        item_scan = _MenuItem("Skanuj", right_text=right_hint, enabled=scan_enabled, parent=panel)
        item_scan.clicked.connect(lambda: self._emit(self._ACTION_SCAN))
        panel.add_item(item_scan)

        panel.add_item(_Separator(panel))

        # -- Powiekszenie
        item_zoom = _MenuItem("Powi\u0119ksz", parent=panel)
        item_zoom.clicked.connect(lambda: self._emit(self._ACTION_ZOOM))
        panel.add_item(item_zoom)

        # -- Eksportuj
        item_export = _MenuItem("Eksportuj fragment PNG", parent=panel)
        item_export.clicked.connect(lambda: self._emit(self._ACTION_EXPORT))
        panel.add_item(item_export)

        panel.finish_layout()
        panel.move(pos)
        panel.show()
        self._panel = panel

    # ── submenu ──────────────────────────────────────────────────────

    def _show_submenu(self):
        self._submenu_timer.stop()

        if self._submenu is not None:
            return

        sub = _PopupPanel(120)

        for label, action in self._MARK_ACTIONS:
            mi = _MenuItem(label, parent=sub)
            mi.clicked.connect(lambda a=action: self._emit(a))
            mi.hovered.connect(self._submenu_timer.stop)
            sub.add_item(mi)

        sub.add_item(_Separator(sub))

        mi_custom = _MenuItem("Inne\u2026", parent=sub)
        mi_custom.clicked.connect(lambda: self._emit("mark_custom"))
        mi_custom.hovered.connect(self._submenu_timer.stop)
        sub.add_item(mi_custom)

        sub.finish_layout()

        # position to the right of the mark item
        if self._panel and self._mark_item:
            item_global = self._mark_item.mapToGlobal(QPoint(0, 0))
            sub.move(
                item_global.x() + self._panel.width() - 6,
                item_global.y() - 6,
            )

        sub.show()
        self._submenu = sub

    def _schedule_close_submenu(self):
        self._submenu_timer.start()

    def _close_submenu(self):
        if self._submenu is not None:
            self._submenu.close()
            self._submenu.deleteLater()
            self._submenu = None

    # ── helpers ──────────────────────────────────────────────────────

    def _emit(self, action: str):
        self.action_selected.emit(action)
        self._cleanup()

    def _cleanup(self):
        self._close_submenu()
        if self._panel is not None:
            self._panel.close()
            self._panel.deleteLater()
            self._panel = None
