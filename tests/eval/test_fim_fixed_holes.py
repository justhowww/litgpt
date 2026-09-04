from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.byte.eval.eval_fim_avclm import (
    _corruption_frame_type,
    _corrupt_gen_frame_hole_spec,
    _load_train_split,
    _verify_fixed_hole_replay,
    summarize,
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


def test_corrupt_gen_frame_hole_is_exact_positioned_and_reproducible(tmp_path):
    stream = tmp_path / "clip.h264"
    stream.write_bytes(b"x" * 4096)

    class FakeDataset:
        fim_min_gap = 1024
        fim_max_gap = 1024
        frame_guard_bytes = 4
        samples = [SimpleNamespace(h264_path=Path(stream))]

        @staticmethod
        def _fim_candidates(sample, data):
            assert sample.h264_path == stream
            assert len(data) == 4096
            return [(0, 1300)]

    first = _corrupt_gen_frame_hole_spec(
        FakeDataset(), 0, corr_pos=0.4, eligibility_bytes=1024, seed=42
    )
    second = _corrupt_gen_frame_hole_spec(
        FakeDataset(), 0, corr_pos=0.4, eligibility_bytes=1024, seed=42
    )

    # 1,300-byte frame - 4-byte guard - 1,024-byte hole = 272-byte
    # placement range. floor(272 * 0.4) = 108.
    assert first == (0, 1300, 112, 1024)
    assert second == first


def test_corruption_frame_type_reads_idr_and_p_slice_type(tmp_path):
    path = tmp_path / "clip.h264"
    # first_mb_in_slice=0 and slice_type=P(0) are each encoded as ue(v) bit 1.
    path.write_bytes(b"\x00\x00\x01\x65\xc0\x00\x00\x01\x41\xc0")
    dataset = SimpleNamespace(
        samples=[SimpleNamespace(h264_path=path, start_nal=0, end_nal=2)],
        nal_index={
            str(path): [
                SimpleNamespace(
                    start=0, end=5, start_code_len=3, nal_type=5
                ),
                SimpleNamespace(
                    start=5, end=10, start_code_len=3, nal_type=1
                ),
            ]
        },
    )

    assert _corruption_frame_type(dataset, 0, 0) == "idr"
    assert _corruption_frame_type(dataset, 0, 5) == "p"


def test_summary_reports_corruption_baseline_and_repair_lift():
    summary = summarize(
        [
            {
                "completed_bytes": 100,
                "target_bytes": 100,
                "completed_frames": 1,
                "stop_reason": "eos",
                "strict_valid": True,
                "cont_psnr_mean": 30.0,
                "cont_ssim_mean": 0.9,
                "corrupted_concealed_psnr": 20.0,
                "corrupted_concealed_ssim": 0.7,
                "repair_psnr_lift_db": 10.0,
                "repair_ssim_lift": 0.2,
                "corrupted_concealed_target_frame_available": True,
                "corrupted_concealed_decode_status": "decoded",
                "corruption_frame_type": "p",
            }
        ],
        stop_mode="learned_eos",
    )

    assert summary["corrupted_concealed_psnr_mean"] == 20.0
    assert summary["repaired_psnr_mean"] == 30.0
    assert summary["repair_psnr_lift_db_mean"] == 10.0
    assert summary["corruption_by_frame_type"]["p"]["count"] == 1
    assert (
        summary["corruption_by_frame_type"]["p"][
            "corrupted_concealed_psnr_mean"
        ]
        == 20.0
    )
    assert summary["repair_quality_paired_count"] == 1
