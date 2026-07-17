"""Randomized round-trip tests for scripts/byte/eval/cavlc_coeff_sanity.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "byte" / "eval" / "cavlc_coeff_sanity.py"
spec = importlib.util.spec_from_file_location("cavlc_coeff_sanity", _SCRIPT)
CS = importlib.util.module_from_spec(spec)
sys.modules["cavlc_coeff_sanity"] = CS
spec.loader.exec_module(CS)


def test_random_consistent_blocks_do_not_overrun():
    for i, (nc, max_coeff) in enumerate([(-1, 4), (0, 16), (1, 16), (2, 16), (4, 16), (8, 16)]):
        result = CS.run_trials(nc, max_coeff, trials=500, seed=1000 + i)
        assert not result["errors"], result


if __name__ == "__main__":
    test_random_consistent_blocks_do_not_overrun()
    print("all passed")
