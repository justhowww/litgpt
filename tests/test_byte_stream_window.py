"""Tests for ByteStreamWindowDataset (multi-frame AR / H0 objective)."""

from __future__ import annotations

import torch

from litgpt.byte.data import (
    IGNORE_INDEX,
    PARAMETER_SET_NAL_TYPES,
    REGION_META,
    REGION_TARGET,
    SLICE_BOS_ID,
    VCL_NAL_TYPES,
    ByteStreamWindowDataset,
    parse_annexb_nals,
)


def _nal(nal_type: int, payload_len: int, four_byte: bool = False) -> bytes:
    start_code = b"\x00\x00\x00\x01" if four_byte else b"\x00\x00\x01"
    header = bytes([nal_type & 0x1F])  # forbidden_zero/ref_idc 0 is fine for tests
    return start_code + header + bytes([0x42] * payload_len)


def _make_stream(specs: list[tuple[int, int]]) -> bytes:
    # First NAL uses a 4-byte start code (common at stream start), rest 3-byte.
    return b"".join(
        _nal(t, n, four_byte=(i == 0)) for i, (t, n) in enumerate(specs)
    )


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    nals = parse_annexb_nals(data)
    return path, {str(path): nals}


def _expected_window(data, nals, start, end):
    chunks, regions, offsets = [], [], []
    for nal in nals[start:end]:
        length = nal.end - nal.start
        chunks.append(torch.tensor(list(data[nal.start : nal.end]), dtype=torch.long))
        region = REGION_META if nal.nal_type in PARAMETER_SET_NAL_TYPES else REGION_TARGET
        regions.append(torch.full((length,), region, dtype=torch.long))
        offsets.append(torch.arange(length, dtype=torch.long))
    return torch.cat(chunks), torch.cat(regions), torch.cat(offsets)


def test_single_window_ar_layout(tmp_path):
    # SPS, PPS, IDR, P, P -> one window of 5 NALs, 3 VCL frames.
    specs = [(7, 8), (8, 4), (5, 40), (1, 20), (1, 16)]
    data = _make_stream(specs)
    path, nal_index = _write(tmp_path, "clip.h264", data)
    rows = [{"h264_path": str(path), "status": "ok"}]

    ds = ByteStreamWindowDataset(rows, max_seq_length=4096, min_frames=2, nal_index=nal_index)
    assert len(ds) == 1
    s = ds.samples[0]
    assert (s.start_nal, s.end_nal, s.num_frames) == (0, 5, 3)

    item = ds[0]
    nals = nal_index[str(path)]
    window, raw_region, raw_offset = _expected_window(data, nals, 0, 5)

    # AR teacher forcing: input = [BOS, window[:-1]], labels = window.
    assert item["input_ids"][0].item() == SLICE_BOS_ID
    assert torch.equal(item["input_ids"][1:], window[:-1])
    assert torch.equal(item["labels"], window)
    # Every window byte is supervised (no IGNORE_INDEX in labels).
    assert (item["labels"] != IGNORE_INDEX).all()

    # region/offset align to input positions: [BOS] + raw[:-1].
    assert item["region_ids"][0].item() == REGION_TARGET
    assert torch.equal(item["region_ids"][1:], raw_region[:-1])
    assert item["offset_ids"][0].item() == 0
    assert torch.equal(item["offset_ids"][1:], raw_offset[:-1])

    # lengths all equal.
    n = item["input_ids"].numel()
    assert item["labels"].numel() == n
    assert item["region_ids"].numel() == n
    assert item["offset_ids"].numel() == n


def test_offset_resets_per_nal_and_regions(tmp_path):
    specs = [(7, 8), (8, 4), (5, 40), (1, 20), (1, 16)]
    data = _make_stream(specs)
    path, nal_index = _write(tmp_path, "clip.h264", data)
    rows = [{"h264_path": str(path), "status": "ok"}]
    ds = ByteStreamWindowDataset(rows, max_seq_length=4096, min_frames=2, nal_index=nal_index)
    item = ds[0]
    nals = nal_index[str(path)]

    # Reconstruct per-NAL offset: each NAL's bytes must run 0..len-1 in raw_offset.
    _, raw_region, raw_offset = _expected_window(data, nals, 0, 5)
    cursor = 0
    for nal in nals:
        length = nal.end - nal.start
        seg = raw_offset[cursor : cursor + length]
        assert torch.equal(seg, torch.arange(length))
        expected_region = REGION_META if nal.nal_type in PARAMETER_SET_NAL_TYPES else REGION_TARGET
        assert (raw_region[cursor : cursor + length] == expected_region).all()
        cursor += length
    # Parameter sets (SPS/PPS) are META, VCL frames are TARGET.
    assert raw_region[0].item() == REGION_META  # SPS
    # Max offset stays within max_seq_length so the model's offset embedding covers it.
    assert int(item["offset_ids"].max()) < 4096


def test_budget_truncates_and_min_frames_skips(tmp_path):
    # Big IDR + small P. Budget fits IDR + one P only (2 VCL); the second P drops.
    specs = [(7, 8), (8, 4), (5, 300), (1, 50), (1, 50)]
    data = _make_stream(specs)
    path, nal_index = _write(tmp_path, "clip.h264", data)
    nals = nal_index[str(path)]
    # Budget = SPS+PPS+IDR+first P, just under the second P.
    budget = sum(n.end - n.start for n in nals[:4])
    rows = [{"h264_path": str(path), "status": "ok"}]
    ds = ByteStreamWindowDataset(rows, max_seq_length=budget, min_frames=2, nal_index=nal_index)
    assert len(ds) == 1
    assert ds.samples[0].end_nal == 4  # second P excluded by budget
    assert ds[0]["input_ids"].numel() <= budget

    # With min_frames=4 the (only) window has 2 VCL frames -> skipped -> empty -> error.
    import pytest

    with pytest.raises(ValueError):
        ByteStreamWindowDataset(rows, max_seq_length=budget, min_frames=4, nal_index=nal_index)


def test_multi_gop_two_nonoverlapping_windows(tmp_path):
    # Two GOPs; budget comfortably fits each but a single window can't span both
    # because the budget is set below the full-stream length.
    specs = [
        (7, 8), (8, 4), (5, 40), (1, 20), (1, 20),  # GOP 1
        (7, 8), (8, 4), (5, 40), (1, 20), (1, 20),  # GOP 2
    ]
    data = _make_stream(specs)
    path, nal_index = _write(tmp_path, "clip.h264", data)
    nals = nal_index[str(path)]
    gop1_len = sum(n.end - n.start for n in nals[:5])
    rows = [{"h264_path": str(path), "status": "ok"}]
    # Budget fits exactly one GOP, not two.
    ds = ByteStreamWindowDataset(rows, max_seq_length=gop1_len, min_frames=2, nal_index=nal_index)
    assert len(ds) == 2
    a, b = ds.samples
    assert (a.start_nal, a.end_nal) == (0, 5)
    assert (b.start_nal, b.end_nal) == (5, 10)
    # Non-overlapping.
    assert a.end_nal <= b.start_nal
