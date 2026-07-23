from __future__ import annotations

import hashlib

import pytest

from scripts.byte.eval.eval_fim_avclm import _verify_fixed_hole_replay


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
