"""Smoke tests for unified ECG viewer (post-merge of 12-lead/1-lead/monitor).

Run: python tests/test_unified_viewer.py
Headless via QT_QPA_PLATFORM=offscreen.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from ui.viewer_page import ViewerPage
from ui.ekg_canvas import generate_demo_signal, ALL_LEADS_ORDER
from ui.theme import STANDARD_LEADS
from marking_store import Marking


_FAILED = []
_PASSED = []


def _check(cond, msg):
    if cond:
        _PASSED.append(msg)
        print(f"  PASS  {msg}")
    else:
        _FAILED.append(msg)
        print(f"  FAIL  {msg}")


def _make_viewer():
    v = ViewerPage()
    sig = generate_demo_signal(STANDARD_LEADS, fs=500, duration=10.0)
    v.set_signal(sig, STANDARD_LEADS, 500, "demo.dat")
    return v


def test_initial_state():
    print("\n[test_initial_state]")
    v = _make_viewer()
    _check(v._layout_id == "grid_4x3", "default layout is grid_4x3")
    _check(v._live is False, "live mode off by default")
    _check(v._focus_lead == "II", "default focus lead is II")
    _check(len(v.grid_view.cells) == 12, "grid has 12 cells")
    _check(v.markings_panel is not None, "markings panel constructed")
    _check(v.layout_switcher is not None, "layout switcher constructed")
    _check(not v._monitor_timer.isActive(), "monitor timer idle by default")


def test_layout_switching_preserves_cells():
    print("\n[test_layout_switching_preserves_cells]")
    v = _make_viewer()
    cell_ids_before = {lead: id(c) for lead, c in v.grid_view.cells.items()}
    for lid in ["grid_4x3", "grid_3x4", "grid_2x6", "stack_1xN", "focus_1L"]:
        v._on_layout_changed(lid)
        _check(v._layout_id == lid, f"layout switched to {lid}")
    cell_ids_after = {lead: id(c) for lead, c in v.grid_view.cells.items()}
    _check(cell_ids_before == cell_ids_after,
           "cell instances preserved across layout switches")


def test_live_toggle_drives_all_visible_cells():
    print("\n[test_live_toggle_drives_all_visible_cells]")
    v = _make_viewer()
    v._on_live_toggled(True)
    _check(v._live is True, "live flag set")
    _check(v._monitor_timer.isActive(), "timer running")
    # Tick a few times
    for _ in range(5):
        v._monitor_tick()
    _check(v._monitor_t > 0, "monitor_t advances on tick")
    # Sweep written into visible cells
    sweeps = [c._sweep_pos for c in v.grid_view.cells.values()]
    _check(any(s is not None for s in sweeps),
           "at least one cell has sweep_pos set")
    v._on_live_toggled(False)
    _check(v._live is False, "live cleared")
    _check(not v._monitor_timer.isActive(), "timer stopped")
    sweeps_after = [c._sweep_pos for c in v.grid_view.cells.values()]
    _check(all(s is None for s in sweeps_after),
           "sweep cleared on all cells after live off")


def test_selection_on_arbitrary_cell():
    print("\n[test_selection_on_arbitrary_cell]")
    v = _make_viewer()
    # Pick aVR (not focus default)
    v._on_cell_selection_completed("aVR", 1.0, 3.0)
    _check(v._pending_selection == ("aVR", 1.0, 3.0),
           "pending selection records lead+window")
    _check(v._last_clicked_lead == "aVR", "last_clicked_lead updated")
    v._clear_selection_preview()
    _check(v._pending_selection is not None or
           all(c.pending_marker is None for c in v.grid_view.cells.values()),
           "preview cleared on every cell")


def test_annotation_creation_flow():
    print("\n[test_annotation_creation_flow]")
    v = _make_viewer()
    v._on_cell_selection_completed("V3", 2.0, 4.0)
    v._on_annotation_created("V3", "Patologia", "Test note", 2.0, 4.0)
    markings = list(v._marking_store.get_all())
    _check(len(markings) == 1, "1 marking added to store")
    _check(markings[0].lead == "V3", "marking lead is V3")
    _check(markings[0].type == "annotation", "marking type is annotation")


def test_markings_render_on_grid_cells():
    print("\n[test_markings_render_on_grid_cells]")
    v = _make_viewer()
    v._marking_store.add(Marking(type="annotation", lead="V5",
                                  t1=1.0, t2=2.0, label="X"))
    v._refresh_markings()
    v3 = v.grid_view.cells["V3"]
    v5 = v.grid_view.cells["V5"]
    _check(len(v5.markings) == 1, "V5 cell has the marking")
    _check(len(v3.markings) == 0, "V3 cell has none")


def test_speed_cycling():
    print("\n[test_speed_cycling]")
    v = _make_viewer()
    v._on_live_toggled(True)
    v._on_monitor_speed(2.0)
    _check(v._monitor_timer.interval() == 25, "2x → 25ms interval")
    v._on_monitor_speed(0.5)
    _check(v._monitor_timer.interval() == 100, "0.5x → 100ms interval")
    v._on_monitor_speed(1.0)
    _check(v._monitor_timer.interval() == 50, "1x → 50ms interval")
    v._on_live_toggled(False)


def test_scrubber_in_static_and_live():
    print("\n[test_scrubber_in_static_and_live]")
    v = _make_viewer()
    # Static seek
    v.scrubber.setValue(200)  # 2s
    _check(abs(v.time_pos - 2.0) < 0.01, "scrubber moves time_pos in static mode")
    # Live seek
    v._on_live_toggled(True)
    v.scrubber.setValue(500)  # 5s
    _check(abs(v._monitor_t - 5.0) < 0.05, "scrubber seeks live sweep")
    v._on_live_toggled(False)


def test_layout_switcher_has_required_controls():
    print("\n[test_layout_switcher_has_required_controls]")
    v = _make_viewer()
    ls = v.layout_switcher
    _check(hasattr(ls, "_layout_btns") and len(ls._layout_btns) == 5,
           "5 layout buttons present")
    _check(hasattr(ls, "_live_btn"), "live toggle button present")
    _check(hasattr(ls, "_speed_btn"), "speed cycle button present")
    _check(hasattr(ls, "_patient_btn"), "patient drawer button present")
    _check(hasattr(ls, "_leads_btn"), "lead chooser button present")


def test_visible_leads_filter_in_subset_layouts():
    print("\n[test_visible_leads_filter_in_subset_layouts]")
    v = _make_viewer()
    v._on_layout_changed("grid_3x4")
    v._on_visible_leads_changed(["I", "II", "V1", "V5"])
    visible = v.grid_view.visible_cell_leads()
    _check(set(visible) == {"I", "II", "V1", "V5"},
           "3x4 honours visible_leads subset")


def test_focus_lead_aliases_single_lead():
    print("\n[test_focus_lead_aliases_single_lead]")
    v = _make_viewer()
    v._on_layout_changed("focus_1L")
    v._focus_lead = "V4"
    v._apply_layout()
    _check(v.single_lead is v.grid_view.cells["V4"],
           "single_lead alias points at focused cell")


def test_pause_button_only_active_in_live():
    print("\n[test_pause_button_only_active_in_live]")
    v = _make_viewer()
    _check(not v.pause_btn.isEnabled(), "pause disabled when not live")
    v._on_live_toggled(True)
    _check(v.pause_btn.isEnabled(), "pause enabled when live")
    v._on_live_toggled(False)
    _check(not v.pause_btn.isEnabled(), "pause disabled after live off")


def test_undo_redo_via_panel():
    print("\n[test_undo_redo_via_panel]")
    v = _make_viewer()
    v._marking_store.add(Marking(type="annotation", lead="II",
                                  t1=0.5, t2=1.5, label="A"))
    v._refresh_markings()
    _check(len(v._marking_store.get_all()) == 1, "1 marking added")
    v._undo()
    _check(len(v._marking_store.get_all()) == 0, "undo removed marking")
    v._redo()
    _check(len(v._marking_store.get_all()) == 1, "redo restored")


def test_no_legacy_attrs():
    print("\n[test_no_legacy_attrs]")
    v = _make_viewer()
    _check(not hasattr(v, "view_seg"), "view_seg removed")
    _check(not hasattr(v, "lead_sidebar"), "lead_sidebar removed")
    _check(not hasattr(v, "monitor_sidebar"), "monitor_sidebar removed")
    _check(not hasattr(v, "view_stack"), "view_stack removed")
    _check(not hasattr(v, "monitor_area"), "monitor_area removed")
    _check(not hasattr(v, "_monitor_strips"), "_monitor_strips removed")


def main():
    tests = [
        test_initial_state,
        test_layout_switching_preserves_cells,
        test_live_toggle_drives_all_visible_cells,
        test_selection_on_arbitrary_cell,
        test_annotation_creation_flow,
        test_markings_render_on_grid_cells,
        test_speed_cycling,
        test_scrubber_in_static_and_live,
        test_layout_switcher_has_required_controls,
        test_visible_leads_filter_in_subset_layouts,
        test_focus_lead_aliases_single_lead,
        test_pause_button_only_active_in_live,
        test_undo_redo_via_panel,
        test_no_legacy_attrs,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            _FAILED.append(f"{t.__name__} EXC: {e}")
            print(f"  EXC {t.__name__}: {e}")

    print(f"\n=== {len(_PASSED)} passed, {len(_FAILED)} failed ===")
    if _FAILED:
        print("FAILURES:")
        for f in _FAILED:
            print(f"  - {f}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
