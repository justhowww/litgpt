from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/byte/eval/diagnose_bscv_concealment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("diagnose_bscv_concealment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = _load_module()


def test_frame_corruption_uses_fixed_hex_length_and_position():
    data = bytes(range(256)) * 20
    packets = [
        {"pos": 0, "size": 2500, "flags": "K__"},
        {"pos": 2500, "size": len(data) - 2500, "flags": "___"},
    ]

    corrupted, cuts = D.corrupt_frames(
        data,
        packets,
        corr_prob=1,
        corr_pos=0.4,
        corr_len_hex=2048,
        seed=42,
    )

    assert len(cuts) == 1
    assert cuts[0].deleted_hex_chars == 2048
    assert not cuts[0].fallback
    assert len(data) - len(corrupted) == 1024


def test_original_vcl_recognizes_three_and_four_byte_start_codes():
    stream = (
        b"\x00\x00\x00\x01\x67" + b"sps" * 5
        + b"\x00\x00\x01\x65" + b"I" * 200
        + b"\x00\x00\x00\x01\x41" + b"P" * 200
    )

    corrupted, cuts = D.corrupt_original_vcl(
        stream,
        corr_prob=1,
        corr_pos=0.4,
        corr_len_hex=100,
        seed=7,
    )

    assert len(cuts) == 1
    assert cuts[0].deleted_hex_chars == 100
    assert len(stream) - len(corrupted) == 50


def test_short_frame_uses_original_style_fallback():
    data = b"x" * 30
    packets = [{"pos": 0, "size": len(data), "flags": "K__"}]

    corrupted, cuts = D.corrupt_frames(
        data,
        packets,
        corr_prob=1,
        corr_pos=0.4,
        corr_len_hex=2048,
        seed=0,
    )

    assert len(cuts) == 1
    assert cuts[0].fallback
    assert 0 < len(corrupted) < len(data)
