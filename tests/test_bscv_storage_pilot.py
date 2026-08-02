"""Stdlib-only tests for the BSCV storage pilot helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts/byte/reports/report_bscv_storage_pilot.py"


def _load():
    spec = importlib.util.spec_from_file_location("report_bscv_storage_pilot", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load()


def test_annexb_header_requires_sps_pps_then_idr():
    stream = (
        b"\x00\x00\x00\x01\x67\x01"
        b"\x00\x00\x00\x01\x68\x02"
        b"\x00\x00\x01\x06\x03"
        b"\x00\x00\x01\x65\x04"
    )
    types = P.annexb_nal_types(stream)
    assert types == [7, 8, 6, 5]
    assert P.header_is_self_contained(types)
    assert not P.header_is_self_contained([7, 8, 1])


def test_variant_command_adds_seek_and_exact_frame_limit():
    cfg = P.load_config(_REPO_ROOT / "preprocessing/h264_preprocess_config_bscv.json")
    cmd = P.variant_command(Path("in.mp4"), Path("out.h264"), cfg, 1.25, 32)
    assert cmd[cmd.index("-ss") + 1] == "1.250000"
    assert cmd[cmd.index("-frames:v") + 1] == "32"
    assert cmd.index("-frames:v") < cmd.index("-f")


def test_summary_projects_mean_bytes_and_paired_savings():
    base = {
        "error": None,
        "ffprobe_ok": True,
        "decode_ok": True,
        "self_contained_header": True,
        "x264_options_match": True,
        "frame_count": 16,
    }
    rows = {
        (0, "full"): {**base, "sample_id": 0, "variant": "full", "target_frames": None, "output_bytes": 1000},
        (0, "gop1"): {**base, "sample_id": 0, "variant": "gop1", "target_frames": 16, "output_bytes": 250},
    }
    summary = P.summarize(rows, [("full", None), ("gop1", 16)], 1_000_000)
    assert summary["variants"]["gop1"]["projected_decimal_tb"] == 0.00025
    assert summary["variants"]["gop1"]["paired_savings_mean"] == 0.75
