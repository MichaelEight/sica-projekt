"""Shared UI helpers used across pages."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFrame

import ui.theme as T


def make_logo(font_size=14):
    logo = QLabel()
    logo.setText('<span style="color:#4a9eff;font-weight:600;">EKG</span>'
                 ' <span style="color:white;font-weight:600;">Assistant</span>')
    logo.setFont(QFont(".AppleSystemUIFont", font_size))
    logo.setTextFormat(Qt.RichText)
    return logo


def make_separator(width=1, height=24):
    sep = QFrame()
    sep.setFixedSize(width, height)
    sep.setStyleSheet(f"background: {T.SEPARATOR};")
    return sep


def section_header(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        font-size: 11px; font-weight: 700; color: {T.TEXT_DIM};
        text-transform: uppercase; letter-spacing: 0.5px;
        padding-bottom: 4px; border-bottom: 1px solid {T.BORDER_LIGHT};
    """)
    return lbl


def info_row(label, value, unit=""):
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px;")
    val_text = f'{value} <span style="font-size:10px;color:{T.TEXT_DIM};">{unit}</span>' if unit else value
    val = QLabel(val_text)
    val.setTextFormat(Qt.RichText)
    val.setStyleSheet("font-weight: 600; font-family: Menlo; font-size: 13px;")
    val.setAlignment(Qt.AlignRight)
    layout.addWidget(lbl)
    layout.addStretch()
    layout.addWidget(val)
    return row


def make_action_btn(text, primary=False):
    btn = QPushButton(text)
    btn.setCursor(Qt.PointingHandCursor)
    if primary:
        btn.setObjectName("primary")
        btn.setStyleSheet(f"""
            QPushButton {{
                padding: 8px; border-radius: 6px; border: none;
                background: {T.ACCENT}; color: {T.ACCENT_TEXT};
                font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{
                background: {T.GREEN if T.is_dark_mode() else '#3a8eef'};
            }}
        """)
    else:
        btn.setStyleSheet(f"""
            QPushButton {{
                padding: 8px; border-radius: 6px; border: 1px solid {T.BORDER};
                background: {T.WHITE}; color: {T.TEXT}; font-size: 12px;
            }}
            QPushButton:hover {{
                background: {T.BG_SECONDARY};
            }}
        """)
    return btn
