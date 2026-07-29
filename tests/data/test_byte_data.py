import json
from pathlib import Path

import pytest
import torch

from litgpt.data.byte_data import (
    FIM_BEGIN_ID,
    FIM_END_ID,
    FIM_HOLE_ID,
    IGNORE_INDEX,
    PAD_ID,
    REGION_BRIDGE,
    REGION_META,
    REGION_ORPHAN,
    REGION_PAD,
    REGION_PREFIX,
    REGION_REF,
    REGION_TARGET,
    SEQ_EOS_ID,
    SLICE_BOS_ID,
    SPAN_BOS_ID,
    VOCAB_SIZE,
    ByteDataConfig,
    ByteDataModule,
    ByteSliceDataset,
    collate_byte_samples,
    default_nal_index_path,
    load_nal_index,
    load_manifest_rows,
    parse_annexb_nals,
    vocab_size_for_fim_format,
)
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.pretrain import get_model_inputs_and_targets
from scripts.build_byte_nal_index import build_index


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


def make_two_video_dataset(tmp_path, reference_mode: str) -> ByteSliceDataset:
    rows = []
    for name, delta in (("a", 0), ("b", 10)):
        stream, _ = synthetic_stream()
        stream = stream.replace(bytes(range(32, 72)), bytes(range(32 + delta, 72 + delta)))
        h264_path = tmp_path / "h264" / f"{name}.h264"
        h264_path.parent.mkdir(parents=True, exist_ok=True)
        h264_path.write_bytes(stream)
        rows.append({"status": "ok", "h264_path": str(h264_path)})
    return ByteSliceDataset(
        rows,
        max_seq_length=256,
        p_fim=0.0,
        num_ref_slices=1,
        reference_mode=reference_mode,
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


def test_parse_annexb_nals_mixed_start_code_lengths():
    """Distinguishes adjacent three-byte and four-byte Annex-B start codes."""
    first = nal(0x67, b"sps", start_code=b"\x00\x00\x01")
    second = nal(0x68, b"pps", start_code=b"\x00\x00\x00\x01")
    stream = first + second

    nals = parse_annexb_nals(stream)

    assert len(nals) == 2  # Both NAL units are discovered.
    assert nals[0].start_code_len == 3  # Three-byte start code is preserved.
    assert nals[1].start_code_len == 4  # Four-byte start code is preserved.
    assert stream[nals[0].start : nals[0].end] == first  # First range is exact.
    assert stream[nals[1].start : nals[1].end] == second  # Second range is exact.


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
    ds = make_dataset(tmp_path, p_fim=0.0, condition_on_sps_pps=False)
    sample = ds[0]

    assert (
        REGION_META not in sample["region_ids"].tolist()
    )  # Metadata region disappears when parameter sets are disabled.
    assert (
        sample["sample_meta"]["num_meta_nals"] == 0
    )  # Metadata count reflects the ablation setting.


def test_reference_ablation_modes_preserve_target_supervision(tmp_path):
    """Changes only reference conditioning while preserving the target labels."""
    normal = make_two_video_dataset(tmp_path / "normal", "normal")[0]
    no_ref = make_two_video_dataset(tmp_path / "no_ref", "no_ref")[0]
    zero_ref = make_two_video_dataset(tmp_path / "zero_ref", "zero_ref")[0]
    shuffled = make_two_video_dataset(tmp_path / "shuffled", "shuffled_ref")[0]

    supervised = normal["labels"] != IGNORE_INDEX
    expected_labels = normal["labels"][supervised]
    assert torch.equal(
        no_ref["labels"][no_ref["labels"] != IGNORE_INDEX], expected_labels
    )  # Removing references does not change the supervised target bytes.
    assert torch.equal(
        zero_ref["labels"][zero_ref["labels"] != IGNORE_INDEX], expected_labels
    )  # Zeroing references does not change the supervised target bytes.
    assert torch.equal(
        shuffled["labels"][shuffled["labels"] != IGNORE_INDEX], expected_labels
    )  # Shuffling references does not change the supervised target bytes.
    assert (
        REGION_REF not in no_ref["region_ids"].tolist()
    )  # no_ref removes the reference region entirely.
    assert torch.all(
        zero_ref["input_ids"][zero_ref["region_ids"] == REGION_REF] == 0
    )  # zero_ref preserves reference positions but replaces every byte with zero.
    assert (
        shuffled["sample_meta"]["reference_source_path"]
        != shuffled["sample_meta"]["h264_path"]
    )  # shuffled_ref selects conditioning from a different video.


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


def test_manifest_row_limit_stops_debug_corpus_loading(tmp_path):
    """Limits corpus indexing for fast smoke/debug runs."""
    stream, _ = synthetic_stream()
    manifest_path, h264_path = write_manifest(tmp_path, stream)
    row = json.dumps({"status": "ok", "h264_path": str(h264_path)})
    manifest_path.write_text("\n".join([row, row, row]) + "\n")

    rows = load_manifest_rows(manifest_path, max_rows=2)

    assert len(rows) == 2  # Loader stops after the requested number of usable rows.


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


def test_fim_full_loss_scope_supervises_reordered_next_tokens(tmp_path):
    ds = make_dataset(
        tmp_path,
        p_fim=1.0,
        fim_format="psm",
        fim_loss_scope="full",
        use_eos=True,
        num_ref_slices=1,
        fim_min_gap=4,
        fim_max_gap=4,
        slice_header_guard_bytes=2,
    )
    sample = ds[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    assert (labels != IGNORE_INDEX).all()
    assert torch.equal(labels[:-1], input_ids[1:])
    assert labels[-1].item() == SEQ_EOS_ID
    assert sample["sample_meta"]["fim_loss_scope"] == "full"


def test_psm_fim_layout_uses_explicit_prefix_suffix_middle_markers(tmp_path):
    """Builds DeepSeek-style Prefix-Suffix-Middle ordering with masked loss."""
    ds = make_dataset(
        tmp_path,
        p_fim=1.0,
        fim_format="psm",
        num_ref_slices=1,
        fim_min_gap=4,
        fim_max_gap=4,
        slice_header_guard_bytes=2,
    )
    sample = ds[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    supervised = labels != IGNORE_INDEX

    begin_pos = input_ids.tolist().index(FIM_BEGIN_ID)
    hole_pos = input_ids.tolist().index(FIM_HOLE_ID)
    end_pos = input_ids.tolist().index(FIM_END_ID)

    assert (
        begin_pos < hole_pos < end_pos
    )  # PSM markers appear in causal generation order.
    assert (
        input_ids[supervised][0].item() == FIM_END_ID
    )  # FIM_END predicts the first missing byte.
    assert (
        input_ids[supervised][1:].tolist() == labels[supervised][:-1].tolist()
    )  # Missing bytes use the same one-token teacher-forcing shift.
    assert int(supervised.sum()) == 4  # Marker changes do not change gap supervision.
    assert (
        sample["sample_meta"]["fim_format"] == "psm"
    )  # Sample metadata records the selected FIM representation.
    assert (
        vocab_size_for_fim_format("bridge") == VOCAB_SIZE
    )  # Existing checkpoints remain shape-compatible.
    assert (
        vocab_size_for_fim_format("psm") == 262
    )  # PSM enables all three additional markers.


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


def test_short_slice_falls_back_to_ar_instead_of_clamping_fim_gap(tmp_path):
    """Preserves the configured FIM minimum instead of silently sampling a smaller gap."""
    ds = make_dataset(
        tmp_path,
        p_fim=1.0,
        fim_min_gap=1024,
        fim_max_gap=8192,
    )

    sample = ds[0]

    assert sample["sample_meta"]["task"] == "ar"  # Ineligible short slices remain valid AR examples.


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


def test_data_module_uses_persistent_nal_index_without_reading_streams(
    tmp_path, monkeypatch
):
    """Loads NAL offsets from SQLite instead of rescanning every H.264 file."""
    stream, _ = synthetic_stream()
    manifest_path, _ = write_manifest(tmp_path, stream)
    index_path = default_nal_index_path(manifest_path)
    build_index(manifest_path, index_path, workers=1, rebuild=True)

    def fail_read_bytes(self):
        raise AssertionError(f"Dataset setup rescanned H.264 bytes from {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    config = ByteDataConfig(num_workers=0, val_fraction=0.5)
    dm = ByteDataModule(
        manifest_path=manifest_path,
        config=config,
        nal_index_path=index_path,
    )
    dm.connect(batch_size=1, max_seq_length=256)

    dm.setup()

    assert dm.train_dataset is not None  # Cached offsets produce the train split.
    assert dm.val_dataset is not None  # Cached offsets produce the validation split.


def test_stale_nal_index_is_rejected_when_manifest_changes(tmp_path):
    """Requires rebuilding the cache after the corpus manifest is modified."""
    stream, _ = synthetic_stream()
    manifest_path, _ = write_manifest(tmp_path, stream)
    index_path = default_nal_index_path(manifest_path)
    build_index(manifest_path, index_path, workers=1, rebuild=True)
    rows = load_manifest_rows(manifest_path)
    manifest_path.write_text(manifest_path.read_text() + "\n")

    with pytest.raises(RuntimeError, match="Stale NAL index"):
        load_nal_index(index_path, manifest_path, rows)


def test_pretrain_batch_helper_preserves_byte_labels(tmp_path):
    """Passes byte labels and auxiliary ids to the model without shifting."""
    ds = make_dataset(tmp_path, p_fim=0.0)
    batch = collate_byte_samples([ds[0]], max_seq_length=256)

    model_inputs, targets = get_model_inputs_and_targets(batch, max_seq_length=256)

    assert torch.equal(
        model_inputs["idx"], batch["input_ids"]
    )  # Byte input_ids are passed to GPT unchanged.
    assert torch.equal(
        targets, batch["labels"]
    )  # Dataset-provided masked labels are not shifted a second time.
    assert torch.equal(
        model_inputs["region_ids"], batch["region_ids"]
    )  # Region ids are forwarded to the model.
    assert torch.equal(
        model_inputs["offset_ids"], batch["offset_ids"]
    )  # Original byte offsets are forwarded to the model.


def test_gpt_accepts_region_and_offset_embeddings():
    """Adds optional region/offset embeddings without changing output shape."""
    config = Config(
        block_size=16,
        n_layer=1,
        n_embd=16,
        n_head=4,
        vocab_size=VOCAB_SIZE,
        padding_multiple=8,
        use_region_id=True,
        use_offset_id=True,
    )
    model = GPT(config)
    input_ids = torch.tensor([[SLICE_BOS_ID, 0, 1, 2]])
    region_ids = torch.tensor([[REGION_TARGET, REGION_TARGET, REGION_TARGET, REGION_TARGET]])
    offset_ids = torch.tensor([[0, 1, 2, 3]])

    logits = model(input_ids, region_ids=region_ids, offset_ids=offset_ids)

    assert logits.shape == (
        1,
        4,
        config.padded_vocab_size,
    )  # Auxiliary embeddings preserve normal LM-logit dimensions.
