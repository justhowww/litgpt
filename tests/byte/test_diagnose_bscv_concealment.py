from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/byte/eval/diagnose_bscv_concealment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("diagnose_bscv_concealment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_factorial_changes_only_cabac_and_slice_count():
    configs = D._factorial_configs(duration=5.0, slices=16)

    assert set(configs) == {
        "cavlc_01slice",
        "cavlc_16slice",
        "cabac_01slice",
        "cabac_16slice",
    }
    reference = configs["cavlc_01slice"][0]
    for name, (config, _) in configs.items():
        assert config.width == reference.width
        assert config.height == reference.height
        assert config.fps == reference.fps
        assert config.qp == reference.qp
        assert config.gop == reference.gop
        assert config.ffmpeg.disable_bframes == reference.ffmpeg.disable_bframes
        assert config.ffmpeg.refs == reference.ffmpeg.refs
        assert config.ffmpeg.x264_params["cabac"] == int(name.startswith("cabac"))
        assert config.ffmpeg.x264_params["slices"] == (16 if "16slice" in name else 1)
