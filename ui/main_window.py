"""Main application window orchestrating all pages."""
import os
import pickle
import threading
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QWidget, QMessageBox, QApplication

import ui.theme as T
from ui.theme import STANDARD_LEADS
from ui.app_logger import get_logger

_log = get_logger("main_window")
from ui.upload_page import UploadPage, add_recent
from ui.viewer_page import ViewerPage
from ui.report_page import ReportPage
from ui.ekg_canvas import generate_demo_signal

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

# ── Ground truth cache ──────────────────────────────────────
_GT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
_GT_CACHE_PATH = os.path.join(_GT_CACHE_DIR, "gt_lookup.pkl")
_CSV_PATH = os.path.join("data", "ptb-xl", "ptbxl_database.csv")


def _build_gt_lookup() -> dict:
    """Build {filename: ground_truth_dict} from ptbxl_database.csv.

    This avoids repeated pandas loads and string matching at runtime.
    """
    import pandas as pd
    from data.filter_data import parse_scp_codes, aggregate_classes
    from ui.theme import TARGET_CLASSES

    db = pd.read_csv(_CSV_PATH, index_col="ecg_id")
    lookup = {}
    for ecg_id, row in db.iterrows():
        scp = parse_scp_codes(row["scp_codes"])
        scores = aggregate_classes(scp)
        gt = {}
        for cls in TARGET_CLASSES:
            key = cls.replace("class_", "")
            gt[cls] = scores.get(key, 0.0) / 100.0

        # Patient info
        sex_val = row.get("sex", "")
        sex_str = "M" if sex_val == 0 else ("K" if sex_val == 1 else "")
        patient = {
            "id": str(int(ecg_id)) if ecg_id else "",
            "age": str(int(row["age"])) if not pd.isna(row.get("age", float("nan"))) else "",
            "sex": sex_str,
            "date": str(row.get("recording_date", ""))[:10],
        }

        entry = {"ground_truth": gt, "patient": patient}
        for col in ["filename_hr", "filename_lr"]:
            fn = row[col].strip() if isinstance(row[col], str) else ""
            if fn:
                lookup[fn] = entry
    return lookup


def _load_or_build_gt_cache() -> dict:
    """Load cached lookup or build + save it."""
    if not os.path.exists(_CSV_PATH):
        return {}

    csv_mtime = os.path.getmtime(_CSV_PATH)

    if os.path.exists(_GT_CACHE_PATH):
        try:
            cache_mtime = os.path.getmtime(_GT_CACHE_PATH)
            if cache_mtime >= csv_mtime:
                with open(_GT_CACHE_PATH, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass

    lookup = _build_gt_lookup()
    try:
        os.makedirs(_GT_CACHE_DIR, exist_ok=True)
        with open(_GT_CACHE_PATH, "wb") as f:
            pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return lookup


# Map common WFDB lead name variants to our standard names
_LEAD_ALIASES = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3",
    "v4": "V4", "v5": "V5", "v6": "V6",
}


def _normalize_lead_names(names: list[str]) -> list[str]:
    """Normalize WFDB lead names to standard form (I, II, aVR, V1, etc.)."""
    return [_LEAD_ALIASES.get(n.lower(), n) for n in names]


class MainWindow(QMainWindow):
    """Main application window with Upload → Viewer → Report flow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kardioskop — Analiza EKG")
        self.setMinimumSize(1200, 800)
        self.showMaximized()

        # Apply theme
        self.setStyleSheet(T.STYLESHEET)

        # Central stacked widget
        self.stack = QStackedWidget()
        self.stack.setObjectName("centralWidget")
        self.setCentralWidget(self.stack)

        # Pages
        self.upload_page = UploadPage()
        self.viewer_page = ViewerPage()
        self.report_page = ReportPage()

        self.stack.addWidget(self.upload_page)         # 0
        self.stack.addWidget(self.viewer_page)         # 1
        self.stack.addWidget(self.report_page)         # 2

        # Connect signals
        self.upload_page.file_selected.connect(self._load_file)
        self.viewer_page.open_file.connect(self._go_upload)
        self.viewer_page.show_report.connect(self._go_report)
        self.viewer_page.toggle_dark.connect(self._toggle_dark_mode)
        self.report_page.go_back.connect(self._go_viewer)

        # Show upload page
        self.stack.setCurrentIndex(0)

        # Current data
        self._signal = None
        self._leads = STANDARD_LEADS
        self._fs = 500
        self._filename = ""

        # Ground truth lookup — loaded in background thread
        self._gt_lookup = {}
        self._gt_ready = threading.Event()
        self._gt_thread = threading.Thread(target=self._bg_load_gt, daemon=True)
        self._gt_thread.start()

        self._setup_shortcuts()

    def _setup_shortcuts(self):
        def _sc(key, handler):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.ApplicationShortcut)
            s.activated.connect(handler)
            return s

        def _on_viewer(fn):
            """Wrap handler to only run when on viewer page."""
            def wrapper():
                if self.stack.currentIndex() == 1:
                    fn()
            return wrapper

        _sc(Qt.Key_1, _on_viewer(lambda: self.viewer_page._on_layout_changed("grid_3x4")))
        _sc(Qt.Key_2, _on_viewer(lambda: self.viewer_page._on_layout_changed("focus_1L")))
        _sc(Qt.Key_3, _on_viewer(lambda: self.viewer_page._on_live_toggled(not self.viewer_page._live)))
        _sc(QKeySequence("Ctrl+Z"), _on_viewer(lambda: self.viewer_page._undo()))
        _sc(QKeySequence("Ctrl+Shift+Z"), _on_viewer(lambda: self.viewer_page._redo()))
        _sc(Qt.Key_Left, _on_viewer(lambda: self.viewer_page._nav_step(-0.2)))
        _sc(Qt.Key_Right, _on_viewer(lambda: self.viewer_page._nav_step(0.2)))
        _sc(Qt.Key_Home, _on_viewer(self.viewer_page._nav_start))
        _sc(Qt.Key_End, _on_viewer(self.viewer_page._nav_end))
        space_sc = _sc(Qt.Key_Space, _on_viewer(lambda: self.viewer_page._on_navbar_pause()
                                     if self.viewer_page._live else None))
        space_sc.setContext(Qt.WindowShortcut)
        _sc(QKeySequence("Ctrl+E"), _on_viewer(self._go_report))
        _sc(QKeySequence("Ctrl+Return"), _on_viewer(self.viewer_page._run_full_analysis))
        _sc(QKeySequence("Ctrl+F"), lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen())

        def _on_escape():
            if self.stack.currentIndex() == 1:
                # Close context menu if open — and stop here
                ctx = getattr(self.viewer_page, '_context_menu', None)
                if ctx is not None and ctx.isVisible():
                    ctx.close()
                    self.viewer_page._context_menu = None
                    self.viewer_page._clear_selection_preview()
                    return
                if self.viewer_page._live:
                    self.viewer_page._on_live_toggled(False)
                    self.viewer_page.layout_switcher.set_live(False)
                else:
                    self.viewer_page._clear_selection_preview()
            elif self.stack.currentIndex() == 2:
                self._go_viewer()
        _sc(Qt.Key_Escape, _on_escape)

    def _bg_load_gt(self):
        """Background thread: load or build ground truth cache."""
        try:
            _log.info("loading ground truth cache in background")
            self._gt_lookup = _load_or_build_gt_cache()
            _log.info("gt cache loaded: %d entries", len(self._gt_lookup or {}))
        except Exception:
            _log.exception("failed to load ground truth cache")
            self._gt_lookup = {}
        self._gt_ready.set()

    def _lookup_csv_entry(self, base_path: str) -> dict | None:
        """Look up PTB-XL cache entry (contains ground_truth + patient)."""
        if not self._gt_ready.wait(timeout=10.0):
            return None
        if not self._gt_lookup:
            return None
        path = base_path.replace("\\", "/")
        for suffix in ["records500/", "records100/"]:
            idx = path.find(suffix)
            if idx >= 0:
                return self._gt_lookup.get(path[idx:])
        return None

    def _lookup_ground_truth(self, base_path: str) -> dict | list | None:
        """Look up ground truth for a record."""
        import json

        # Check for .annotations.json sidecar first
        json_path = base_path + ".annotations.json"
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                return data.get("windows", None)
            except Exception:
                pass

        # Fall back to PTB-XL CSV cache
        entry = self._lookup_csv_entry(base_path)
        if entry:
            return entry.get("ground_truth")
        return None

    def _lookup_patient_info(self, base_path: str) -> dict | None:
        """Look up patient info from PTB-XL CSV cache."""
        entry = self._lookup_csv_entry(base_path)
        if entry:
            return entry.get("patient")
        return None

    def _load_file(self, base_path: str):
        """Load a WFDB record or generate demo data."""
        _log.info("load_file: %s", base_path)
        try:
            self._load_file_impl(base_path)
        except Exception:
            _log.exception("load_file failed: base_path=%r", base_path)
            self.statusBar().clearMessage()
            QMessageBox.critical(
                self, "Błąd krytyczny",
                "Nie udało się wczytać pliku. Szczegóły w logs/app.log"
            )

    def _load_file_impl(self, base_path: str):
        # Validate file format
        ext = os.path.splitext(base_path)[1].lower()
        if ext and ext not in ('.dat', '.hea', ''):
            QMessageBox.warning(
                self, "Nieobsługiwany format",
                f"Plik \"{os.path.basename(base_path)}\" nie jest w formacie WFDB.\n"
                "Obsługiwane formaty: .dat, .hea")
            return

        # Show loading indicator
        self.statusBar().showMessage("Wczytywanie pliku...")
        QApplication.processEvents()

        dat_path = base_path + ".dat"
        hea_path = base_path + ".hea"

        if os.path.exists(dat_path) and os.path.exists(hea_path) and HAS_WFDB:
            try:
                record = wfdb.rdrecord(base_path)
                self._signal = record.p_signal.astype(np.float32)
                self._fs = int(round(float(record.fs)))
                self._leads = _normalize_lead_names(record.sig_name)
                self._filename = os.path.basename(base_path) + ".dat"

                n = self._signal.shape[0]
                duration = n / float(self._fs)

                info = f"{duration:.2f} s @ {self._fs} Hz"
                add_recent(base_path, info)
                self.statusBar().showMessage(
                    f"Wczytano {self._filename} | {n} pts @ {self._fs} Hz ({duration:.2f}s)",
                    10000,
                )
                _log.info(
                    "Load %s: shape=%s, fs=%d Hz, duration=%.3f s (no resample on load — applied per analysis window)",
                    self._filename, self._signal.shape, self._fs, duration,
                )
            except Exception as e:
                self.statusBar().clearMessage()
                QMessageBox.warning(self, "Błąd", f"Nie udało się wczytać pliku:\n{e}")
                return
        else:
            # Generate demo signal
            self._signal = generate_demo_signal(STANDARD_LEADS, fs=500, duration=10.0)
            self._leads = STANDARD_LEADS
            self._fs = 500
            self._filename = os.path.basename(base_path) + ".dat" if base_path else "demo.dat"
            add_recent(base_path or "demo", "10.0 s")

        # Look up ground truth and patient info
        self.statusBar().showMessage("Wczytywanie adnotacji...")
        QApplication.processEvents()
        ground_truth = self._lookup_ground_truth(base_path)
        patient_info = self._lookup_patient_info(base_path)

        self.viewer_page.set_signal(self._signal, self._leads, self._fs, self._filename,
                                    ground_truth=ground_truth, patient_info=patient_info,
                                    base_path=base_path)
        self.report_page.set_signal(self._signal, self._leads, self._fs, self._filename)

        self.statusBar().clearMessage()
        self.stack.setCurrentIndex(1)

    def _go_upload(self):
        self.upload_page.refresh()
        self.stack.setCurrentIndex(0)

    def _go_viewer(self):
        self.stack.setCurrentIndex(1)

    def _go_report(self):
        if self.viewer_page._last_results:
            r = self.viewer_page._last_results
            self.report_page.set_results(r["probabilities"], r["model_name"], r["elapsed"])
        # Prefer full scan summary over single-window results
        scan = getattr(self.viewer_page, "_autoscan_results", None)
        if scan:
            model_name = ""
            if self.viewer_page._model_path:
                model_name = os.path.basename(self.viewer_page._model_path)
            self.report_page.set_scan_summary(scan, model_name=model_name)

        # Pass patient info to report
        pf = self.viewer_page.info_panel._patient_fields
        duration = f"{self.viewer_page.duration:.1f}" if hasattr(self.viewer_page, "duration") else ""
        self.report_page.set_patient_info(
            patient_id=pf["id"].text() if "id" in pf else "",
            age=pf["age"].text() if "age" in pf else "",
            sex=pf["sex"].text() if "sex" in pf else "",
            date=pf["date"].text() if "date" in pf else "",
            duration=duration,
            fs=str(self.viewer_page.fs) if hasattr(self.viewer_page, "fs") else "",
        )

        # Pass measurements if available
        if hasattr(self.viewer_page, '_measurements') and self.viewer_page._measurements:
            self.report_page.set_measurements(self.viewer_page._measurements)

        # Pass annotations from marking store
        annotations = [
            {"t1": m.t1, "t2": m.t2, "lead": m.lead, "category": m.category,
             "note": m.note, "source": m.source}
            for m in self.viewer_page._marking_store.get_by_type(["annotation"])
        ]
        self.report_page.set_annotations(annotations)

        self.stack.setCurrentIndex(2)

    def _toggle_dark_mode(self):
        from ui.theme import is_dark_mode, set_dark_mode
        from ui import config
        new_dark = not is_dark_mode()
        set_dark_mode(new_dark)
        config.set_dark_mode(new_dark)
        # Re-apply global stylesheet
        self.setStyleSheet(T.STYLESHEET)
        # Re-apply all component styles
        self.viewer_page.apply_theme()
        self.report_page.apply_theme()
        # Refresh views to pick up new colors
        if self._signal is not None:
            self.viewer_page._refresh_views()

