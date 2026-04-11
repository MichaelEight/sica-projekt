"""Application-wide logging setup.

Writes to logs/app.log (rotating) and stderr. Installs a global
sys.excepthook and a Qt message handler so uncaught exceptions and
Qt warnings/errors are captured with full context.
"""
import logging
import logging.handlers
import os
import sys
import traceback

from PySide6.QtCore import qInstallMessageHandler, QtMsgType

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_LOG_PATH = os.path.join(_LOG_DIR, "app.log")

_configured = False


def get_logger(name: str = "ekg") -> logging.Logger:
    global _configured
    if not _configured:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] "
            "%(filename)s:%(lineno)d %(funcName)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        root = logging.getLogger("ekg")
        root.setLevel(logging.DEBUG)
        root.propagate = False

        fh = logging.handlers.RotatingFileHandler(
            _LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        _install_excepthook(root)
        _install_qt_handler(root)

        root.info("=== logger initialized (log: %s) ===", _LOG_PATH)
        _configured = True

    return logging.getLogger(name if name.startswith("ekg") else f"ekg.{name}")


def _install_excepthook(logger: logging.Logger):
    def _hook(exc_type, exc, tb):
        logger.critical(
            "UNCAUGHT EXCEPTION:\n%s",
            "".join(traceback.format_exception(exc_type, exc, tb)),
        )
    sys.excepthook = _hook


def _install_qt_handler(logger: logging.Logger):
    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _handler(msg_type, context, message):
        level = level_map.get(msg_type, logging.INFO)
        where = ""
        if context and context.file:
            where = f" [{context.file}:{context.line} {context.function}]"
        logger.log(level, "Qt: %s%s", message, where)

    qInstallMessageHandler(_handler)
