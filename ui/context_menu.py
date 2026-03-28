"""Theme-aware custom context menu for EKG selection workflow."""

from PySide6.QtCore import Qt, Signal, QPoint, QTimer, QObject, QEvent
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsDropShadowEffect, QApplication,
)

import ui.theme as T


class _MenuItem(QWidget):
    """A single clickable row inside the context menu."""

    clicked = Signal()
    hovered = Signal()

    def __init__(self, label, right_text="", has_submenu=False, enabled=True, parent=None):
        super().__init__(parent)
        self._label = label
        self._right_text = right_text
        self._has_submenu = has_submenu
        self._enabled = enabled
        self._is_hovered = False
        self.setFixedHeight(34)
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
        self.setMouseTracking(True)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._is_hovered and self._enabled:
            p.setBrush(QColor(T.BG_SECONDARY))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(self.rect().adjusted(4, 2, -4, -2), 6, 6)
        text_color = QColor(T.TEXT) if self._enabled else QColor(T.TEXT_DIM)
        p.setPen(text_color)
        p.setFont(self.font())
        r = self.rect().adjusted(16, 0, -16, 0)
        p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, self._label)
        if self._has_submenu:
            p.drawText(r, Qt.AlignRight | Qt.AlignVCenter, "\u25b8")
        elif self._right_text:
            p.setPen(QColor(T.TEXT_DIM))
            p.drawText(r, Qt.AlignRight | Qt.AlignVCenter, self._right_text)
        p.end()

    def enterEvent(self, event):
        if self._enabled:
            self._is_hovered = True
            self.update()
            self.hovered.emit()

    def leaveEvent(self, event):
        self._is_hovered = False
        self.update()

    def mousePressEvent(self, event):
        if self._enabled and event.button() == Qt.LeftButton:
            self.clicked.emit()


class _Separator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(9)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setPen(QPen(QColor(T.BORDER), 1.0))
        y = self.height() // 2
        p.drawLine(12, y, self.width() - 12, y)
        p.end()


class _PopupPanel(QWidget):
    """Rounded popup panel with manual lifecycle."""

    def __init__(self, width, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedWidth(width)
        self.setMouseTracking(True)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(shadow)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 6, 4, 6)
        self._layout.setSpacing(0)

    def add_item(self, item):
        self._layout.addWidget(item)

    def finish_layout(self):
        self.adjustSize()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(2, 2, -2, -2), 10, 10)
        p.fillPath(path, QColor(T.WHITE))
        p.setPen(QPen(QColor(T.BORDER), 1.0))
        p.drawPath(path)
        p.end()


class _ClickOutsideFilter(QObject):
    """Event filter: detects clicks outside the menu to close it."""

    triggered = Signal()

    def __init__(self, menu, parent=None):
        super().__init__(parent)
        self._menu = menu

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            gpos = event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else event.globalPos()
            panel = self._menu._panel
            submenu = self._menu._submenu
            inside = False
            if panel and panel.isVisible() and panel.geometry().contains(gpos):
                inside = True
            if submenu and submenu.isVisible() and submenu.geometry().contains(gpos):
                inside = True
            if not inside and panel and panel.isVisible():
                self.triggered.emit()
                return True
        return False


class SelectionContextMenu(QWidget):
    """Custom context menu for the EKG selection workflow."""

    action_selected = Signal(str)

    _MARK_ACTIONS = [
        ("PR", "mark_pr"),
        ("QRS", "mark_qrs"),
        ("QT", "mark_qt"),
        ("R-R", "mark_rr"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        self._panel = None
        self._submenu = None
        self._mark_item = None
        self._click_filter = None

    def show_at(self, pos, selection_seconds):
        self._cleanup()

        scan_enabled = selection_seconds >= 10.0
        panel = _PopupPanel(240)

        # Adnotacja
        item = _MenuItem("Adnotacja", parent=panel)
        item.clicked.connect(lambda: self._emit("annotate"))
        item.hovered.connect(self._close_submenu)
        panel.add_item(item)

        # Oznacz...
        item_mark = _MenuItem("Oznacz\u2026", has_submenu=True, parent=panel)
        item_mark.hovered.connect(self._open_submenu)
        panel.add_item(item_mark)
        self._mark_item = item_mark

        panel.add_item(_Separator(panel))

        # Skanuj
        hint = "" if scan_enabled else "min. 10 s"
        item = _MenuItem("Skanuj", right_text=hint, enabled=scan_enabled, parent=panel)
        item.clicked.connect(lambda: self._emit("scan"))
        item.hovered.connect(self._close_submenu)
        panel.add_item(item)

        panel.add_item(_Separator(panel))

        # Powiększ
        item = _MenuItem("Powi\u0119ksz", parent=panel)
        item.clicked.connect(lambda: self._emit("zoom"))
        item.hovered.connect(self._close_submenu)
        panel.add_item(item)

        # Eksportuj
        item = _MenuItem("Eksportuj fragment PNG", parent=panel)
        item.clicked.connect(lambda: self._emit("export_png"))
        item.hovered.connect(self._close_submenu)
        panel.add_item(item)

        panel.finish_layout()
        panel.move(pos)
        panel.show()
        self._panel = panel

        # Install click-outside filter
        self._click_filter = _ClickOutsideFilter(self)
        self._click_filter.triggered.connect(self._cleanup)
        app = QApplication.instance()
        if app:
            app.installEventFilter(self._click_filter)

    def close(self):
        self._cleanup()
        super().close()

    def isVisible(self):
        return self._panel is not None and self._panel.isVisible()

    # -- submenu --

    def _open_submenu(self):
        if self._submenu is not None:
            return

        sub = _PopupPanel(120)
        for label, action in self._MARK_ACTIONS:
            mi = _MenuItem(label, parent=sub)
            mi.clicked.connect(lambda a=action: self._emit(a))
            sub.add_item(mi)

        sub.add_item(_Separator(sub))
        mi = _MenuItem("Inne\u2026", parent=sub)
        mi.clicked.connect(lambda: self._emit("mark_custom"))
        sub.add_item(mi)

        sub.finish_layout()

        if self._panel and self._mark_item:
            g = self._mark_item.mapToGlobal(QPoint(0, 0))
            sub.move(g.x() + self._panel.width() - 6, g.y() - 6)

        sub.show()
        self._submenu = sub

    def _close_submenu(self):
        if self._submenu is not None:
            self._submenu.close()
            self._submenu.deleteLater()
            self._submenu = None

    # -- helpers --

    def _emit(self, action):
        self.action_selected.emit(action)
        self._cleanup()

    def _cleanup(self):
        self._close_submenu()
        if self._panel is not None:
            self._panel.close()
            self._panel.deleteLater()
            self._panel = None
        self._mark_item = None
        if self._click_filter:
            app = QApplication.instance()
            if app:
                app.removeEventFilter(self._click_filter)
            self._click_filter = None
