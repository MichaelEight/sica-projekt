"""Standalone runner for test_resample.py (no pytest)."""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_resample as tr  # noqa: E402

TESTS = [name for name in dir(tr) if name.startswith("test_")]
passed, failed = 0, []

for name in TESTS:
    try:
        getattr(tr, name)()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed.append(name)

print(f"\n{passed}/{len(TESTS)} passed")
sys.exit(0 if not failed else 1)
