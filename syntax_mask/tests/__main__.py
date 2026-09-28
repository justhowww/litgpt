"""Run the tests without pytest: ``python -m syntax_mask.tests``."""

import sys
import time
import traceback

from . import test_syntax_mask as mod

failed = 0
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    started = time.perf_counter()
    try:
        getattr(mod, name)()
        print(f"PASS {name} ({time.perf_counter() - started:.1f}s)")
    except Exception:  # noqa: BLE001
        failed += 1
        print(f"FAIL {name}")
        traceback.print_exc()
sys.exit(1 if failed else 0)
