"""Ground-truth sanity for the NAL-termination analyzer.

A real encoder NEVER closes a NAL before its syntax completes. So on ground truth every
boundary must classify as valid_start_code and zero as premature_start_code. If this
fails, the analyzer is wrong -- not the model. That makes this the decisive test of the
diagnostic itself, and it needs no GPU or model.

Stdlib-only (no torch), loaded by path like the other byte tests.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[2] / "scripts" / "byte" / "eval"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_FILES = ["baseline_qcif.h264", "baseline_qcif_lowqp.h264"]

# The fixtures are one-slice-per-frame (99 MBs per slice), unlike the slice-max-mbs=1
# corpus, so the automaton must be told how many MBs make a complete slice.
_FIXTURE_MAX_MBS = 99


def _load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "nal_termination", _EVAL_DIR / "nal_termination.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nal_termination"] = module
    spec.loader.exec_module(module)
    return module


NT = _load_analyzer()


@functools.lru_cache(maxsize=None)
def _analyze_fixture(name: str):
    """Ground truth is its own reference: gen == gt, whole stream, no prefix.

    Cached: the replay is O(N^2) per NAL (advance re-unescapes the open NAL each byte)
    and these fixtures are one-slice-per-frame, so a NAL is ~1.5 kB. The real
    slice-max-mbs=1 corpus has ~10-byte NALs, where that cost vanishes.
    """
    path = _FIXTURES / name
    if not path.exists():
        return None
    data = path.read_bytes()
    return NT.analyze_stream(data, data, 0, slice_max_mbs=_FIXTURE_MAX_MBS)


def test_ground_truth_has_no_premature_boundaries():
    for name in _FIXTURE_FILES:
        res = _analyze_fixture(name)
        if res is None:
            continue
        hist = res["classification_hist"]
        assert hist.get(NT.PREMATURE, 0) == 0, (
            f"{name}: analyzer reports {hist[NT.PREMATURE]} premature boundaries on "
            f"ENCODER output -- the analyzer is wrong. hist={hist}"
        )
        vcl_closed = hist.get(NT.VALID, 0)
        assert vcl_closed > 0, f"{name}: no VCL boundary was classified: {hist}"


def test_ground_truth_pairs_and_matches_itself():
    """gen == gt, so every NAL must align, delta_bytes == 0, TotalCoeff identical."""
    for name in _FIXTURE_FILES:
        res = _analyze_fixture(name)
        if res is None:
            continue
        for n in res["nals"]:
            if "delta_bytes" in n:
                assert n["delta_bytes"] == 0, f"{name}: {n}"
            if "total_coeff_match" in n:
                assert n["total_coeff_match"], f"{name}: {n}"


def test_no_parser_automaton_contradiction_on_ground_truth():
    """The two independent implementations must agree on GT: no NAL may be called
    'done' by the automaton yet desync in the recursive parser (or vice versa)."""
    for name in _FIXTURE_FILES:
        res = _analyze_fixture(name)
        if res is None:
            continue
        assert not res["parser_automaton_contradictions"], (
            f"{name}: automaton says done but parser desyncs at gen_nal_index "
            f"{res['parser_automaton_contradictions']}"
        )


if __name__ == "__main__":  # stdlib runner
    for _n in _FIXTURE_FILES:
        if not (_FIXTURES / _n).exists():
            print(f"skip {_n} (missing)")
            continue
        _r = _analyze_fixture(_n)
        print(f"{_n}: {_r['classification_hist']}  nals={_r['n_gen_nals']}")
    test_ground_truth_has_no_premature_boundaries()
    test_ground_truth_pairs_and_matches_itself()
    test_no_parser_automaton_contradiction_on_ground_truth()
    print("all passed")
