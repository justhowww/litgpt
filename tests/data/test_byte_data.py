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
    parse_annexb_nals,
)


def nal(nal_header: int, payload: bytes, start_code: bytes = b"\x00\x00\x00\x01") -> bytes:
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


def write_manifest(tmp_path, stream: bytes):
    h264_path = tmp_path / "h264" / "clip.h264"
    h264_path.parent.mkdir()
    h264_path.write_bytes(stream)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps({"status": "ok", "h264_path": str(h264_path)}) + "\n")
    return manifest_path, h264_path


def make_dataset(tmp_path, **kwargs) -> ByteSliceDataset:
    stream, _ = synthetic_stream()
    manifest_path, _ = write_manifest(tmp_path, stream)
    rows = [json.loads(manifest_path.read_text())]
    return ByteSliceDataset(rows, max_seq_length=kwargs.pop("max_seq_length", 256), **kwargs)


def test_parse_annexb_nals():
    """Parses Annex-B start codes and preserves exact NAL byte ranges."""
    stream, parts = synthetic_stream()
    nals = parse_annexb_nals(stream)

    assert [n.nal_type for n in nals] == [7, 8, 5, 1, 1]
    assert stream[nals[0].start : nals[0].end] == parts["sps"]
    assert stream[nals[-1].start : nals[-1].end] == parts["p2"]


def test_ar_sample_layout_includes_metadata_and_reference(tmp_path):
    """Builds AR samples as [B_meta, B_ref, SLICE_BOS, B_t[:-1]] -> B_t."""
    ds = make_dataset(tmp_path, p_fim=0.0, num_ref_slices=1)
    sample = ds[0]
    labels = sample["labels"]
    input_ids = sample["input_ids"]
    region_ids = sample["region_ids"]
    supervised = labels != IGNORE_INDEX

    assert input_ids.shape == labels.shape == region_ids.shape
    assert int(supervised.sum()) > 0
    assert input_ids[supervised.nonzero()[0].item() - 1].item() != SPAN_BOS_ID
    assert input_ids[supervised][0].item() == SLICE_BOS_ID
    assert input_ids[supervised][1:].tolist() == labels[supervised][:-1].tolist()
    assert REGION_META in region_ids.tolist()
    assert REGION_REF in region_ids.tolist()
    assert REGION_TARGET in region_ids.tolist()
    assert sample["sample_meta"]["task"] == "ar"
    assert sample["sample_meta"]["num_meta_nals"] == 2
    assert sample["sample_meta"]["num_ref_slices"] == 1


def test_metadata_can_be_disabled(tmp_path):
    """Allows ablations where SPS/PPS metadata is not fed as conditioning."""
    ds = make_dataset(tmp_path, p_fim=0.0, include_parameter_sets=False)
    sample = ds[0]

    assert REGION_META not in sample["region_ids"].tolist()
    assert sample["sample_meta"]["num_meta_nals"] == 0


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

    assert sample["sample_meta"]["task"] == "fim"
    assert int(supervised.sum()) == 4
    assert REGION_PREFIX in region_ids.tolist()
    assert REGION_ORPHAN in region_ids.tolist()
    assert REGION_BRIDGE in region_ids.tolist()
    assert torch.all(labels[~supervised] == IGNORE_INDEX)


def test_reference_slices_are_dropped_whole_when_context_is_tight(tmp_path):
    """Drops oversized reference slices as whole units instead of truncating them."""
    ds = make_dataset(tmp_path, p_fim=0.0, num_ref_slices=1, max_seq_length=64)
    sample = ds[0]

    assert sample["sample_meta"]["dropped_ref_slices"] == 1
    assert REGION_REF not in sample["region_ids"].tolist()
    assert int((sample["labels"] != IGNORE_INDEX).sum()) > 0


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

    assert batch["input_ids"].shape == batch["labels"].shape == batch["region_ids"].shape
    assert batch["input_ids"][1, -1].item() == PAD_ID
    assert batch["labels"][1, -1].item() == IGNORE_INDEX
    assert batch["region_ids"][1, -1].item() == REGION_PAD


def test_byte_data_module_smoke(tmp_path):
    """Checks the LitGPT DataModule wrapper can produce a train batch."""
    stream, _ = synthetic_stream()
    manifest_path, _ = write_manifest(tmp_path, stream)
    config = ByteDataConfig(num_workers=0, val_fraction=0.5)
    dm = ByteDataModule(manifest_path=manifest_path, config=config)
    dm.connect(batch_size=1, max_seq_length=256)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == batch["labels"].shape
