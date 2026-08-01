"""Encoder-command tests for the H.264 preprocessing pipeline.

`preprocessing/parse_mp4_to_h264.py` is stdlib-only, so we load it by file path
to stay independent of the torch/lightning training environment (the repo
conftest imports torch). These tests pin the exact ffmpeg command produced for
both the default (one-slice-per-frame, QP28) config and the AVC-LM fall-back
config (per-MB slices via slice-max-mbs=1, QP37, 256x144, scale+pad, -t 5).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

_PREPROC_DIR = Path(__file__).resolve().parents[1] / "preprocessing"


def _load_module(name: str):
    if str(_PREPROC_DIR) not in sys.path:
        sys.path.insert(0, str(_PREPROC_DIR))
    spec = importlib.util.spec_from_file_location(name, _PREPROC_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # required so dataclass annotations resolve (py3.9)
    spec.loader.exec_module(module)
    return module


P = _load_module("parse_mp4_to_h264")

_DEFAULT_CONFIG = _PREPROC_DIR / "h264_preprocess_config.json"
_AVCLM_CONFIG = _PREPROC_DIR / "h264_preprocess_config_avclm.json"
_BSCV_CONFIG = _PREPROC_DIR / "h264_preprocess_config_bscv.json"


def _cmd(config_path: Path) -> list[str]:
    cfg = P.load_config(config_path)
    return P.build_encode_command(Path("IN.mp4"), Path("OUT.h264"), cfg)


def test_default_config_backward_compatible():
    """The QP28 one-slice-per-frame config is unchanged by the new fields."""
    cfg = P.load_config(_DEFAULT_CONFIG)
    assert cfg.clip_duration_sec is None
    assert cfg.ffmpeg.disable_subtitles is False
    assert cfg.ffmpeg.bitstream_filter is None
    assert cfg.keyint_min == cfg.gop == 16

    cmd = _cmd(_DEFAULT_CONFIG)
    joined = " ".join(cmd)
    # New stream-level options must NOT appear when unset.
    assert "-t" not in cmd
    assert "-sn" not in cmd
    assert "-bsf:v" not in cmd
    assert "qp=28" in joined
    assert "slices=1" in joined


def test_avclm_config_command():
    """AVC-LM fall-back config emits the exact per-MB-slice ffmpeg command."""
    cfg = P.load_config(_AVCLM_CONFIG)
    assert cfg.width == 256 and cfg.height == 144
    assert cfg.resize_mode == "scale_pad"
    assert cfg.fps == 3
    assert cfg.clip_duration_sec == 5
    assert cfg.qp == 37
    assert cfg.ffmpeg.refs == 3
    assert cfg.ffmpeg.disable_subtitles is True
    assert cfg.ffmpeg.bitstream_filter is None  # raw -f h264 is already Annex-B

    cmd = _cmd(_AVCLM_CONFIG)
    joined = " ".join(cmd)

    # Scale+pad filter with fps appended.
    vf_idx = cmd.index("-vf")
    assert cmd[vf_idx + 1] == (
        "scale=256:144:force_original_aspect_ratio=decrease,"
        "pad=256:144:(ow-iw)/2:(oh-ih)/2,fps=3"
    )
    # Clip duration and subtitle disable.
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "5"
    assert "-sn" in cmd
    # refs 3.
    assert cmd[cmd.index("-refs") + 1] == "3"
    # x264-params: per-MB slices, QP37, simplified opts.
    x264 = cmd[cmd.index("-x264-params") + 1]
    for token in [
        "slice-max-mbs=1",
        "trellis=0",
        "me=dia",
        "subme=0",
        "psy=0",
        "mixed-refs=0",
        "fast-pskip=0",
        "partitions=none",
        "qp=37",
    ]:
        assert token in x264, token
    # No bitstream filter for the raw Annex-B muxer.
    assert "-bsf:v" not in cmd
    assert joined.endswith("-preset veryfast -f h264 OUT.h264")


def test_build_video_filter_scale_pad():
    cfg = P.load_config(_AVCLM_CONFIG)
    assert P.build_video_filter(cfg) == (
        "scale=256:144:force_original_aspect_ratio=decrease,"
        "pad=256:144:(ow-iw)/2:(oh-ih)/2,fps=3"
    )


def test_bitstream_filter_emitted_when_set():
    """When bitstream_filter is set, -bsf:v is emitted before -f."""
    cfg = P.load_config(_AVCLM_CONFIG)
    cfg_ffmpeg = dataclasses.replace(cfg.ffmpeg, bitstream_filter="h264_mp4toannexb")
    cfg = dataclasses.replace(cfg, ffmpeg=cfg_ffmpeg)
    cmd = P.build_encode_command(Path("IN.mp4"), Path("OUT.h264"), cfg)
    assert "-bsf:v" in cmd
    assert cmd[cmd.index("-bsf:v") + 1] == "h264_mp4toannexb"
    # Emitted before the output muxer selection.
    assert cmd.index("-bsf:v") < cmd.index("-f")


def test_bscv_config_matches_recovered_x264_settings():
    """The BSCV preset pins settings recovered from two embedded x264 SEIs."""
    cfg = P.load_config(_BSCV_CONFIG)
    assert cfg.fps == 30
    assert cfg.rate_control == "qp" and cfg.qp == 1
    assert cfg.gop == 16 and cfg.keyint_min == 1
    assert cfg.ffmpeg.profile == "high"
    assert cfg.ffmpeg.refs == 3
    assert cfg.ffmpeg.scene_cut_threshold == 40
    assert cfg.ffmpeg.threads == 22

    cmd = _cmd(_BSCV_CONFIG)
    assert cmd[cmd.index("-g") + 1] == "16"
    assert cmd[cmd.index("-keyint_min") + 1] == "1"
    assert cmd[cmd.index("-refs") + 1] == "3"
    assert cmd[cmd.index("-sc_threshold") + 1] == "40"
    assert "-crf" not in cmd

    x264 = cmd[cmd.index("-x264-params") + 1]
    assert "slices=" not in x264
    for token in [
        "cabac=1",
        "bframes=3",
        "b-pyramid=2",
        "b-adapt=1",
        "8x8dct=1",
        "me=hex",
        "subme=7",
        "mbtree=0",
        "aq-mode=0",
        "qp=1",
    ]:
        assert token in x264, token

    settings = P.output_settings_to_manifest(cfg)
    assert settings["codec"] == "h264_high_cabac"
    assert settings["keyint_min"] == 1
