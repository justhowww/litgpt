from __future__ import annotations

import hashlib
import json

import pytest

from scripts.byte.eval.eval_fim_avclm import (
    _load_train_split,
    _verify_fixed_hole_replay,
)


def test_fixed_hole_replay_accepts_identical_target_and_boundaries():
    target = [1, 2, 3, 255]
    meta = {
        "frame_lo": 100,
        "frame_hi": 900,
        "fim_split": 240,
        "fim_gap": len(target),
    }
    expected = {
        **meta,
        "target_length": len(target),
        "target_sha256": hashlib.sha256(bytes(target)).hexdigest(),
    }
    _verify_fixed_hole_replay(expected, meta, target, ("clip.h264", 0, 10))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fim_split", 241),
        ("fim_gap", 5),
        ("target_sha256", "not-the-training-target"),
    ],
)
def test_fixed_hole_replay_rejects_any_difference(field, value):
    target = [1, 2, 3, 255]
    meta = {
        "frame_lo": 100,
        "frame_hi": 900,
        "fim_split": 240,
        "fim_gap": len(target),
    }
    expected = {
        **meta,
        "target_length": len(target),
        "target_sha256": hashlib.sha256(bytes(target)).hexdigest(),
        field: value,
    }
    with pytest.raises(RuntimeError, match="Fixed-hole replay mismatch"):
        _verify_fixed_hole_replay(expected, meta, target, ("clip.h264", 0, 10))


def test_train_split_groups_multiple_holes_per_window(tmp_path):
    path = tmp_path / "train_split.json"
    holes = [
        {
            "hole_id": hole_id,
            "h264_path": "clip.h264",
            "start_nal": 0,
            "end_nal": 10,
            "frame_lo": 100,
            "frame_hi": 900,
            "fim_split": 240 + hole_id,
            "fim_gap": 4,
        }
        for hole_id in range(4)
    ]
    path.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "h264_path": "clip.h264",
                        "start_nal": 0,
                        "end_nal": 10,
                    }
                ],
                "fixed_fim_holes": list(reversed(holes)),
            }
        ),
        encoding="utf-8",
    )

    windows, _, grouped = _load_train_split(path)
    key = ("clip.h264", 0, 10)
    assert key in windows
    assert [hole["hole_id"] for hole in grouped[key]] == list(range(4))
