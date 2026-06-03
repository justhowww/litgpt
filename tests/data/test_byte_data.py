import json

import torch

from litgpt.data.byte_data import (
    IGNORE_INDEX,
    PAD_ID,
    REGION_BRIDGE,
    REGION_META,
    REGION_ORPHAN,
    REGION_PAD,
    REGION_PREFIX,
    REGION_REF,
    REGION_TARGET,
    SLICE_BOS_ID,
    SPAN_BOS_ID,
    ByteDataConfig,
    ByteDataModule,
    ByteSliceDataset,
    collate_byte_samples,
    load_manifest_rows,
    parse_annexb_nals,
)


def nal(
    nal_header: int, payload: bytes, start_code: bytes = b"\x00\x00\x00\x01"
) -> bytes:
    return start_code + bytes([nal_header]) + payload


def synthetic_stream() -> tuple[bytes, dict[str, bytes]]:
    parts = {
        "sps": nal(0x67, b"sps"),
        "pps": nal(0x68, b"pps"),
        "idr": nal(0x65, b"I" * 12),
        "p1": nal(0x41, bytes(range(32, 72))),
        "p2": nal(0x41, bytes(range(72, 112))),
    }
    return b"".join(parts.values()), parts


def write_manifest(tmp_path, stream: bytes, relative_path: bool = False):
    h264_path = tmp_path / "h264" / "clip.h264"
    h264_path.parent.mkdir()
    h264_path.write_bytes(stream)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_h264_path = h264_path.relative_to(tmp_path) if relative_path else h264_path
    manifest_path.write_text(
        json.dumps({"status": "ok", "h264_path": str(manifest_h264_path)}) + "\n"
    )
    return manifest_path, h264_path


def make_dataset(tmp_path, **kwargs) -> ByteSliceDataset:
    stream, _ = synthetic_stream()
    manifest_path, _ = write_manifest(tmp_path, stream)
    rows = [json.loads(manifest_path.read_text())]
    return ByteSliceDataset(
        rows, max_seq_length=kwargs.pop("max_seq_length", 256), **kwargs
    )


def test_parse_annexb_nals():
    """Parses Annex-B start codes and preserves exact NAL byte ranges."""
    stream, parts = synthetic_stream()
    nals = parse_annexb_nals(stream)

    assert [n.nal_type for n in nals] == [
        7,
        8,
        5,
        1,
        1,
    ]  # SPS/PPS/IDR/P/P NAL types are parsed in stream order.
    assert (
        stream[nals[0].start : nals[0].end] == parts["sps"]
    )  # NAL start/end offsets preserve the first NAL exactly.
    assert (
        stream[nals[-1].start : nals[-1].end] == parts["p2"]
    )  # NAL start/end offsets preserve the final NAL exactly.


def test_ar_sample_layout_includes_metadata_and_reference(tmp_path):
    """Builds AR samples as [B_meta, B_ref, SLICE_BOS, B_t[:-1]] -> B_t."""
    ds = make_dataset(tmp_path, p_fim=0.0, num_ref_slices=1)
    sample = ds[0]
    labels = sample["labels"]
    input_ids = sample["input_ids"]
    region_ids = sample["region_ids"]
    supervised = labels != IGNORE_INDEX

    assert (
        input_ids.shape == labels.shape == region_ids.shape
    )  # Per-token tensors stay aligned.
    assert int(supervised.sum()) > 0  # AR sample has target bytes under loss.
    assert (
        input_ids[supervised.nonzero()[0].item() - 1].item() != SPAN_BOS_ID
    )  # AR mode does not use the FIM bridge marker.
    assert (
        input_ids[supervised][0].item() == SLICE_BOS_ID
    )  # AR target generation starts from the slice BOS token.
    assert (
        input_ids[supervised][1:].tolist() == labels[supervised][:-1].tolist()
    )  # Teacher forcing shifts B_t by one token.
    assert (
        REGION_META in region_ids.tolist()
    )  # SPS/PPS metadata is included as conditioning.
    assert (
        REGION_REF in region_ids.tolist()
    )  # Previous reference slice is included as conditioning.
    assert (
        REGION_TARGET in region_ids.tolist()
    )  # Target slice positions are marked separately from conditioning.
    assert (
        sample["sample_meta"]["task"] == "ar"
    )  # Sample metadata records the selected AR task.
    assert (
        sample["sample_meta"]["num_meta_nals"] == 2
    )  # Both latest SPS and PPS are attached.
    assert (
        sample["sample_meta"]["num_ref_slices"] == 1
    )  # Requested one reference slice is attached.


def test_metadata_can_be_disabled(tmp_path):
    """Allows ablations where SPS/PPS metadata is not fed as conditioning."""
    ds = make_dataset(tmp_path, p_fim=0.0, include_parameter_sets=False)
    sample = ds[0]

    assert (
        REGION_META not in sample["region_ids"].tolist()
    )  # Metadata region disappears when parameter sets are disabled.
    assert (
        sample["sample_meta"]["num_meta_nals"] == 0
    )  # Metadata count reflects the ablation setting.


def test_relative_h264_paths_resolve_from_manifest_dir(tmp_path):
    """Keeps h264/ + manifest.jsonl movable as one corpus directory."""
    stream, _ = synthetic_stream()
    manifest_path, h264_path = write_manifest(tmp_path, stream, relative_path=True)

    rows = load_manifest_rows(manifest_path)

    assert (
        rows[0]["h264_path"] == str(h264_path)
    )  # Relative h264_path resolves against manifest.jsonl's directory.
    assert ByteSliceDataset(
        rows, max_seq_length=256
    )  # Resolved path can be opened and indexed by the dataset.


def test_old_cwd_relative_h264_paths_use_manifest_h264_sibling(tmp_path):
    """Supports old manifests written with cwd-relative output paths."""
    stream, _ = synthetic_stream()
    manifest_path, h264_path = write_manifest(tmp_path, stream)
    stale_path = "../SHELL.metzler-prj/OpenVid-1M/h264/h264/clip.h264"
    manifest_path.write_text(
        json.dumps({"status": "ok", "h264_path": stale_path}) + "\n"
    )

    rows = load_manifest_rows(manifest_path)

    assert (
        rows[0]["h264_path"] == str(h264_path)
    )  # Old cwd-relative path is recovered using manifest_dir/h264.
    assert ByteSliceDataset(
        rows, max_seq_length=256
    )  # Recovered old-manifest path can be opened and indexed.


def test_fim_sample_supervises_only_missing_span(tmp_path):
    """Builds FIM samples that place loss only on the generated missing span."""
    ds = make_dataset(
        tmp_path,
        p_fim=1.0,
        num_ref_slices=1,
        fim_min_gap=4,
        fim_max_gap=4,
        slice_header_guard_bytes=2,
    )
    sample = ds[0]
    labels = sample["labels"]
    region_ids = sample["region_ids"]
    supervised = labels != IGNORE_INDEX

    assert (
        sample["sample_meta"]["task"] == "fim"
    )  # Sample metadata records the selected FIM task.
    assert (
        int(supervised.sum()) == 4
    )  # Fixed FIM gap length creates exactly four supervised bytes.
    assert (
        REGION_PREFIX in region_ids.tolist()
    )  # Received bytes before the gap are present.
    assert (
        REGION_ORPHAN in region_ids.tolist()
    )  # Received bytes after the gap are present.
    assert (
        REGION_BRIDGE in region_ids.tolist()
    )  # Generated missing-span input positions are marked.
    assert torch.all(
        labels[~supervised] == IGNORE_INDEX
    )  # Conditioning tokens are excluded from loss.


def test_reference_slices_are_dropped_whole_when_context_is_tight(tmp_path):
    """Drops oversized reference slices as whole units instead of truncating them."""
    ds = make_dataset(tmp_path, p_fim=0.0, num_ref_slices=1, max_seq_length=64)
    sample = ds[0]

    assert (
        sample["sample_meta"]["dropped_ref_slices"] == 1
    )  # Tight context records the dropped reference slice.
    assert (
        REGION_REF not in sample["region_ids"].tolist()
    )  # Reference is dropped when context length is limited.
    assert (
        int((sample["labels"] != IGNORE_INDEX).sum()) > 0
    )  # Target supervision is preserved after dropping refs.


def test_collate_pads_with_byte_constants(tmp_path):
    """Pads batched tensors with byte-data constants understood by the model/loss."""
    ds = make_dataset(tmp_path, p_fim=0.0, num_ref_slices=1)
    sample_a = ds[0]
    sample_b = {
        **sample_a,
        "input_ids": sample_a["input_ids"][:-5],
        "labels": sample_a["labels"][:-5],
        "region_ids": sample_a["region_ids"][:-5],
        "offset_ids": sample_a["offset_ids"][:-5],
    }

    batch = collate_byte_samples([sample_a, sample_b], max_seq_length=256)

    assert (
        batch["input_ids"].shape == batch["labels"].shape == batch["region_ids"].shape
    )  # Collated tensors remain aligned.
    assert (
        batch["input_ids"][1, -1].item() == PAD_ID
    )  # Shorter input sequence is padded with PAD_ID.
    assert (
        batch["labels"][1, -1].item() == IGNORE_INDEX
    )  # Padding positions are excluded from loss.
    assert (
        batch["region_ids"][1, -1].item() == REGION_PAD
    )  # Padding positions get the pad region id.


def test_byte_data_module_smoke(tmp_path):
    """Checks the LitGPT DataModule wrapper can produce a train batch."""
    stream, _ = synthetic_stream()
    manifest_path, _ = write_manifest(tmp_path, stream)
    config = ByteDataConfig(num_workers=0, val_fraction=0.5)
    dm = ByteDataModule(manifest_path=manifest_path, config=config)
    dm.connect(batch_size=1, max_seq_length=256)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    assert "input_ids" in batch  # DataModule train loader emits model inputs.
    assert "labels" in batch  # DataModule train loader emits LM labels.
    assert (
        batch["input_ids"].shape == batch["labels"].shape
    )  # Batch input/label tensors are token-aligned.
