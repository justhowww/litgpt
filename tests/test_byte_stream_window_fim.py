"""Tests for ByteStreamWindowDataset's masked-span FIM mode (p_fim > 0).

The fixtures imitate the AVC-LM corpus shape that phase 2 actually trains on:
slice-max-mbs=1, so one VCL NAL is ONE MACROBLOCK and a frame is a RUN of NALs.
Only the run's first NAL carries first_mb_in_slice == 0. That distinction is the
whole point -- a hole sized like a real corruption spans many NALs, and "frame"
cannot mean "NAL" here.

first_mb_in_slice is ue(v) at the top of the slice payload, so payload byte 0x80
(binary 1000_0000) decodes to 0 -> frame start, and 0x42 (0100_0010) decodes to
1 -> a continuation macroblock.
"""

from __future__ import annotations

import pytest
import torch

from litgpt.byte.data import (
    FIM_BEGIN_ID,
    FIM_END_ID,
    FIM_HOLE_ID,
    IGNORE_INDEX,
    REGION_BRIDGE,
    REGION_ORPHAN,
    REGION_PREFIX,
    SEQ_EOS_ID,
    ByteStreamWindowDataset,
    parse_annexb_nals,
)

MBS_PER_FRAME = 20
MB_PAYLOAD = 8
MB_NAL_BYTES = 3 + 1 + MB_PAYLOAD  # start code + header + payload
FRAME_BYTES = MBS_PER_FRAME * MB_NAL_BYTES  # 240


def _mb_nal(nal_type: int, frame_start: bool, four_byte: bool = False) -> bytes:
    start_code = b"\x00\x00\x00\x01" if four_byte else b"\x00\x00\x01"
    header = bytes([nal_type & 0x1F])
    lead = 0x80 if frame_start else 0x42  # ue(v) -> 0 (frame start) or 1
    return start_code + header + bytes([lead]) + bytes([0x42] * (MB_PAYLOAD - 1))


def _frame(nal_type: int) -> bytes:
    return _mb_nal(nal_type, frame_start=True) + b"".join(
        _mb_nal(nal_type, frame_start=False) for _ in range(MBS_PER_FRAME - 1)
    )


def _stream(num_p_frames: int = 3) -> bytes:
    sps = _mb_nal(7, frame_start=False, four_byte=True)
    pps = _mb_nal(8, frame_start=False)
    return sps + pps + _frame(5) + b"".join(_frame(1) for _ in range(num_p_frames))


def _dataset(tmp_path, **kwargs):
    data = _stream()
    path = tmp_path / "clip.h264"
    path.write_bytes(data)
    nal_index = {str(path): parse_annexb_nals(data)}
    rows = [{"h264_path": str(path), "status": "ok"}]
    params = dict(
        max_seq_length=4096,
        min_frames=2,
        p_fim=1.0,
        fim_format="psm",
        fim_min_gap=16,
        fim_max_gap=64,
        frame_guard_bytes=16,
        nal_index=nal_index,
    )
    params.update(kwargs)
    return ByteStreamWindowDataset(rows, **params), data


def _meta(item):
    return item["sample_meta"]


def test_frame_bounds_track_first_mb_not_nal_boundaries(tmp_path):
    ds, _ = _dataset(tmp_path)
    bounds = ds._frame_bounds(ds.samples[0], ds.samples[0].h264_path.read_bytes())
    # 4 frames (1 IDR + 3 P), each a run of MBS_PER_FRAME NALs -- NOT 80 frames.
    assert len(bounds) == 4
    head = 4 + 1 + MB_PAYLOAD + 3 + 1 + MB_PAYLOAD  # SPS + PPS
    assert bounds == [
        (head + i * FRAME_BYTES, head + (i + 1) * FRAME_BYTES) for i in range(4)
    ]


def test_psm_layout_and_labels_only_on_missing_span(tmp_path):
    ds, _ = _dataset(tmp_path)
    item = ds[0]
    assert _meta(item)["task"] == "fim"

    ids, labels = item["input_ids"], item["labels"]
    gap = _meta(item)["fim_gap"]
    begin = (ids == FIM_BEGIN_ID).nonzero().flatten()
    hole = (ids == FIM_HOLE_ID).nonzero().flatten()
    end = (ids == FIM_END_ID).nonzero().flatten()
    assert len(begin) == len(hole) == len(end) == 1
    assert begin.item() < hole.item() < end.item()

    # Exactly the missing span is supervised, and it is the tail of the sample.
    supervised = (labels != IGNORE_INDEX).nonzero().flatten()
    assert len(supervised) == gap
    assert supervised[-1].item() == len(labels) - 1
    assert torch.equal(supervised, torch.arange(len(labels) - gap, len(labels)))

    # Teacher forcing: FIM_END then missing[:-1] predicts missing.
    assert ids[end.item()] == FIM_END_ID
    assert torch.equal(ids[end.item() + 1 :], labels[-gap:][:-1])


def test_missing_span_is_excised_from_the_hole_frame(tmp_path):
    ds, data = _dataset(tmp_path)
    item = ds[0]
    m = _meta(item)
    window = torch.tensor(list(data), dtype=torch.long)
    split, gap = m["fim_split"], m["fim_gap"]

    # The supervised span is exactly window[split:split+gap] ...
    assert torch.equal(item["labels"][-gap:], window[split : split + gap])
    # ... and it lies inside the chosen frame, past the head guard.
    assert m["frame_lo"] + ds.frame_guard_bytes <= split
    assert split + gap <= m["frame_hi"]


def test_causality_no_bytes_after_the_hole_frame(tmp_path):
    # The window holds 4 frames; whichever one hosts the hole, everything after it
    # must be dropped -- real-time repair never sees future frames.
    for idx in range(8):
        ds, data = _dataset(tmp_path, seed=idx)
        item = ds[0]
        m = _meta(item)
        expected = m["frame_hi"] + ds._fim_overhead()
        assert len(item["input_ids"]) == expected
        assert m["frame_hi"] <= len(data)


def test_guard_keeps_parameter_sets_and_frame_head_out_of_the_hole(tmp_path):
    for idx in range(16):
        ds, _ = _dataset(tmp_path, seed=idx)
        item = ds[0]
        m = _meta(item)
        # The hole never reaches back into SPS/PPS or the frame's first NAL, which
        # carries the first_mb_in_slice == 0 that makes it a frame boundary.
        assert m["fim_split"] >= m["frame_lo"] + ds.frame_guard_bytes


def test_regions_mark_prefix_orphan_and_bridge(tmp_path):
    ds, _ = _dataset(tmp_path)
    item = ds[0]
    regions, ids = item["region_ids"], item["input_ids"]
    gap = _meta(item)["fim_gap"]
    assert regions[(ids == FIM_BEGIN_ID).nonzero().item()] == REGION_PREFIX
    assert regions[(ids == FIM_HOLE_ID).nonzero().item()] == REGION_ORPHAN
    assert (regions[-gap:] == REGION_BRIDGE).all()


def test_sample_never_exceeds_max_seq_length(tmp_path):
    # collate's pad_and_truncate cuts from the RIGHT, so an oversized sample would
    # amputate missing_tail and silently destroy every label in it.
    budget = 2 * FRAME_BYTES + 40
    for idx in range(16):
        ds, _ = _dataset(tmp_path, seed=idx, max_seq_length=budget)
        item = ds[idx % len(ds)]
        assert len(item["input_ids"]) <= budget
        assert (item["labels"] != IGNORE_INDEX).any()


def test_use_eos_appends_terminator_to_the_span(tmp_path):
    ds, _ = _dataset(tmp_path, use_eos=True)
    item = ds[0]
    assert item["labels"][-1].item() == SEQ_EOS_ID
    # PSM's net overhead is 2 (3 markers, minus the byte the teacher-forcing shift
    # gives back), + 1 for EOS.
    assert len(item["input_ids"]) == _meta(item)["frame_hi"] + 3


def test_fim_overhead_matches_the_realized_sample_length(tmp_path):
    # Pins the arithmetic _fim_candidates budgets against. If these disagree, the
    # budget check is wrong and oversized samples get truncated from the right.
    for fmt, eos in [("psm", False), ("psm", True), ("bridge", False), ("bridge", True)]:
        ds, _ = _dataset(tmp_path, fim_format=fmt, use_eos=eos)
        item = ds[0]
        assert len(item["input_ids"]) == _meta(item)["frame_hi"] + ds._fim_overhead()


def test_p_fim_zero_keeps_the_ar_objective(tmp_path):
    ds, data = _dataset(tmp_path, p_fim=0.0)
    item = ds[0]
    assert _meta(item)["task"] == "ar"
    assert _meta(item)["fim_format"] == "stream"
    assert len(item["input_ids"]) == len(data)
    assert (item["labels"] != IGNORE_INDEX).all()


def test_unreachable_fim_raises_instead_of_silently_training_ar(tmp_path):
    # The trap this guard exists for: ByteSliceDataset on this corpus computes
    # max_fim_gap == 1 against fim_min_gap == 64 and quietly returns AR items for
    # every sample, while the run logs p_fim > 0 and reports FIM metrics.
    with pytest.raises(ValueError, match="silently fall back to AR"):
        _dataset(tmp_path, fim_min_gap=4096, fim_max_gap=8192)


def test_hole_spans_many_nals_on_a_per_mb_corpus(tmp_path):
    # Documents the (b)-corpus caveat concretely: a corruption-sized hole cannot be
    # sub-NAL here, so BSCV's "excise inside one slice payload, NAL header intact"
    # artifact is not reproducible -- the hole necessarily eats whole macroblocks.
    ds, _ = _dataset(tmp_path, fim_min_gap=60, fim_max_gap=64)
    gap = _meta(ds[0])["fim_gap"]
    assert gap // MB_NAL_BYTES >= 5
