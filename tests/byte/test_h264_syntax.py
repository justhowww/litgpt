"""Invariant tests for the Constrained-Baseline CAVLC syntax parser.

The parser (litgpt/byte/h264_syntax.py) is stdlib-only, so these load it by file
path to stay independent of the torch/lightning training environment (the repo
conftest imports torch). Correctness is asserted via the two self-checking
invariants a desync would break, plus full byte coverage:

  A. exact consumption -- after the last macroblock the bit cursor sits on
     rbsp_stop_one_bit (zero gap).
  B. MB count          -- decoded MBs == PicWidthInMbs * PicHeightInMbs.
  C. coverage          -- the syntax spans tile the whole Annex-B file.

Fixtures are Constrained-Baseline/CAVLC clips under tests/byte/fixtures/,
generated with `ffmpeg -profile:v baseline -coder 0 -bf 0`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BYTE_DIR = Path(__file__).resolve().parents[2] / "litgpt" / "byte"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_FILES = ["baseline_qcif.h264", "baseline_qcif_lowqp.h264"]


def _load_module(name: str):
    if str(_BYTE_DIR) not in sys.path:
        sys.path.insert(0, str(_BYTE_DIR))
    spec = importlib.util.spec_from_file_location(name, _BYTE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # required so dataclass annotations resolve (py3.9)
    spec.loader.exec_module(module)
    return module


def _load_parser():
    return _load_module("h264_syntax")


H = _load_parser()


def _fixtures():
    found = [_FIXTURES / name for name in _FIXTURE_FILES if (_FIXTURES / name).exists()]
    if not found:
        pytest.skip("no baseline/CAVLC fixtures present (see module docstring to generate)")
    return found


def _parsed(path: Path):
    data = path.read_bytes()
    return data, H.parse_stream(data, parse_slice_data=True)


@pytest.mark.parametrize("name", _FIXTURE_FILES)
def test_invariants_per_fixture(name):
    path = _FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} missing")
    data, parsed = _parsed(path)
    sps = next(iter(parsed.sps.values()))
    assert sps.profile_idc == 66, "fixture must be (Constrained) Baseline"
    expect_mbs = sps.pic_width_in_mbs * sps.pic_height_in_mbs

    vcl = [p for p in parsed.nals if p.nal.nal_type in (1, 5)]
    assert vcl, "no VCL slices parsed"
    for p in vcl:
        assert p.status == H.ParseStatus.OK, f"nal {p.nal.index}: {p.status} ({p.reason})"
        # Invariant B
        assert p.mb_count == expect_mbs, f"nal {p.nal.index}: {p.mb_count} != {expect_mbs} MBs"
        # Invariant A: rbsp_trailing begins exactly at rbsp_stop_one_bit.
        rbsp, _, _ = H.unescape_rbsp(data, p.nal.payload_start + 1, p.nal.payload_end)
        stop = H.BitReader(rbsp).compute_stop_bit()
        trailing = [s for s in p.spans if s.category == H.Category.RBSP_TRAILING]
        assert trailing, f"nal {p.nal.index}: no rbsp_trailing span"
        assert trailing[0].bit_start == stop, (
            f"nal {p.nal.index}: cursor off rbsp_stop_one_bit by "
            f"{trailing[0].bit_start - stop} bits (desync)"
        )


@pytest.mark.parametrize("name", _FIXTURE_FILES)
def test_byte_coverage_tiles_file(name):
    path = _FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} missing")
    data, parsed = _parsed(path)
    spans = sorted(parsed.all_spans(), key=lambda s: s.byte_start)
    cursor = 0
    gaps = 0
    for span in spans:
        if span.byte_start > cursor:
            gaps += 1
        cursor = max(cursor, span.byte_end)
    assert gaps == 0, f"{gaps} coverage gaps"
    assert cursor == len(data), f"covered {cursor} of {len(data)} bytes"


def test_emulation_prevention_byte_map_monotonic():
    for path in _fixtures():
        data = path.read_bytes()
        for nal in H.iter_nals(data):
            rbsp, byte_map, epb = H.unescape_rbsp(data, nal.payload_start + 1, nal.payload_end)
            assert len(rbsp) == len(byte_map)
            assert all(byte_map[i] < byte_map[i + 1] for i in range(len(byte_map) - 1))
            # EPBs are the dropped bytes: total accounted == payload minus header.
            assert len(byte_map) + len(epb) == nal.payload_end - (nal.payload_start + 1)


def test_tables_validate_on_import():
    # Importing the tables module runs its prefix-free / coverage validator.
    module = _load_module("h264_cavlc_tables")
    assert module.COEFF_TOKEN_BINS[0][(0, 0)] == "1"


if __name__ == "__main__":
    # Local runner (no pytest/conftest, so no torch needed).
    paths = [p for p in (_FIXTURES / n for n in _FIXTURE_FILES) if p.exists()]
    assert paths, "no fixtures found"
    for path in paths:
        data, parsed = _parsed(path)
        sps = next(iter(parsed.sps.values()))
        expect = sps.pic_width_in_mbs * sps.pic_height_in_mbs
        vcl = [p for p in parsed.nals if p.nal.nal_type in (1, 5)]
        for p in vcl:
            rbsp, _, _ = H.unescape_rbsp(data, p.nal.payload_start + 1, p.nal.payload_end)
            stop = H.BitReader(rbsp).compute_stop_bit()
            trailing = [s for s in p.spans if s.category == H.Category.RBSP_TRAILING][0]
            assert p.status == H.ParseStatus.OK and p.mb_count == expect
            assert trailing.bit_start == stop
        print(f"{path.name}: {len(vcl)} VCL slices OK, {expect} MBs each, invariants A+B hold")
    print("all invariant checks passed")
