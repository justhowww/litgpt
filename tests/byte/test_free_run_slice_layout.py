"""Focused rollout semantics for fixed-MB versus one-picture slices."""

from __future__ import annotations

from pathlib import Path

import pytest

from litgpt.byte import h264_syntax as HS
from litgpt.byte.free_run_eval import _target_closed_vcl_nals
from scripts.byte.eval.eval_ar_continuation import audit_gt_continuation_mask


def test_frame_layout_stops_after_requested_complete_vcl_nals():
    assert _target_closed_vcl_nals(2, "frame") == 2


def test_macroblock_layout_preserves_next_frame_boundary_lookahead():
    assert _target_closed_vcl_nals(2, "macroblock") == 3


def test_rollout_layout_validation_is_explicit():
    with pytest.raises(ValueError, match="unknown slice layout"):
        _target_closed_vcl_nals(2, "automatic")


def test_exact_prefix_gt_mask_preflight_accepts_full_picture_fixture():
    fixture = Path(__file__).resolve().parent / "fixtures" / "baseline_qcif.h264"
    if not fixture.exists():
        pytest.skip("fixture missing")
    data = fixture.read_bytes()
    vcl = [nal for nal in HS.iter_nals(data) if nal.nal_type in HS.VCL_NAL_TYPES]
    assert len(vcl) >= 2
    target = vcl[1]
    result = audit_gt_continuation_mask(
        data[: target.start_code_start],
        data[target.start_code_start : target.payload_end],
        "frame",
    )
    assert result["ok"], result
    assert result["completed_mbs"] == result["slice_max_mbs"] == 99
