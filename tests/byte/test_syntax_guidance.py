"""Training-time syntax guidance: mask alignment and the -log p(legal) term."""

from __future__ import annotations

import math

import pytest
import torch

from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    EOS_VOCAB_SIZE,
    IGNORE_INDEX,
    SEQ_EOS_ID,
    ByteStreamWindowDataset,
    collate_byte_samples,
    parse_annexb_nals,
)
from litgpt.byte.training import (
    byte_training_loss_terms,
    syntax_legality_terms,
    unpack_syntax_masks,
)
from tests.test_byte_stream_window_fim import _stream

CODE_BASE = 200  # offset-encoding bits live at byte values 200..212 (unused by _stream)


def _mask_row(byte: int, offset: int) -> bytes:
    """Legal set {GT byte} plus a binary encoding of the file offset."""
    value = 1 << byte
    for k in range(13):
        if offset >> k & 1:
            value |= 1 << (CODE_BASE + k)
    return value.to_bytes(32, "little")


def _write_clip(tmp_path, data: bytes):
    path = tmp_path / "clip.h264"
    path.write_bytes(data)
    masks = b"".join(_mask_row(b, i) for i, b in enumerate(data))
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "clip.h264.masks").write_bytes(masks)
    return path, masks


def _dataset(tmp_path, **kwargs):
    data = _stream()
    path, masks = _write_clip(tmp_path, data)
    params = dict(
        max_seq_length=4096,
        min_frames=2,
        p_fim=1.0,
        fim_format="psm",
        fim_loss_scope="full",
        use_eos=True,
        fim_min_gap=16,
        fim_max_gap=64,
        frame_guard_bytes=16,
        nal_index={str(path): parse_annexb_nals(data)},
        syntax_mask_dir=tmp_path / "masks",
    )
    params.update(kwargs)
    dataset = ByteStreamWindowDataset(
        [{"h264_path": str(path), "status": "ok"}], **params
    )
    return dataset, data, masks


def _constrained_rows(item):
    rows = item["syntax_masks"]
    return [(t, bytes(rows[t].tolist())) for t in range(rows.size(0)) if rows[t].any()]


def _expected_rows(masks: bytes, offsets):
    return [masks[32 * o : 32 * (o + 1)] for o in offsets]


def test_psm_full_scope_constrains_context_prefix_and_middle_in_file_order(tmp_path):
    dataset, _, masks = _dataset(tmp_path)
    item = dataset[0]
    meta = item["sample_meta"]
    split, gap = meta["fim_split"], meta["fim_gap"]
    got = _constrained_rows(item)
    # context[1:] + prefix = file offsets 1..split-1, then the middle; the orphan,
    # markers and EOS are never constrained.
    assert [row for _, row in got] == _expected_rows(masks, range(1, split + gap))
    labels = item["labels"]
    for t, _ in got:
        assert 0 <= int(labels[t]) < BYTE_VOCAB_SIZE
    assert int(labels[-1]) == SEQ_EOS_ID and not item["syntax_masks"][-1].any()


def test_psm_span_scope_constrains_only_the_middle(tmp_path):
    dataset, _, masks = _dataset(tmp_path, fim_loss_scope="span")
    item = dataset[0]
    split, gap = item["sample_meta"]["fim_split"], item["sample_meta"]["fim_gap"]
    assert [row for _, row in _constrained_rows(item)] == _expected_rows(
        masks, range(split, split + gap)
    )


def test_bridge_format_uses_contiguous_context_and_prefix(tmp_path):
    dataset, _, masks = _dataset(tmp_path, fim_format="bridge")
    item = dataset[0]
    split, gap = item["sample_meta"]["fim_split"], item["sample_meta"]["fim_gap"]
    assert [row for _, row in _constrained_rows(item)] == _expected_rows(
        masks, range(1, split + gap)
    )


def test_ar_item_constrains_every_window_byte_but_not_eos(tmp_path):
    dataset, data, masks = _dataset(tmp_path, p_fim=0.0)
    item = dataset[0]
    window_len = item["labels"].numel() - 1
    assert [row for _, row in _constrained_rows(item)] == _expected_rows(
        masks, range(window_len)
    )
    assert not item["syntax_masks"][-1].any()


def test_patched_collation_keeps_masks_aligned_with_labels(tmp_path):
    dataset, _, _ = _dataset(tmp_path)
    items = [dataset[0], dataset[0]]
    flat = sum(int(item["syntax_masks"].any(dim=-1).sum()) for item in items)
    batch = collate_byte_samples(items, max_seq_length=4096, byte_patch_size=4)
    masks, labels = batch["syntax_masks"], batch["labels"]
    assert masks.shape == (*labels.shape, 32)
    legal = unpack_syntax_masks(masks)
    constrained = legal.any(dim=-1)
    assert int(constrained.sum()) == flat
    targets = labels[constrained]
    assert bool(legal[constrained].gather(1, targets.unsqueeze(1)).all())


def test_gt_byte_outside_mask_is_rejected(tmp_path):
    dataset, data, masks = _dataset(tmp_path)
    path = tmp_path / "masks" / "clip.h264.masks"
    broken = bytearray(masks)
    broken[32 * 40 : 32 * 41] = _mask_row((data[40] + 1) % 256, 40)
    path.write_bytes(bytes(broken))
    with pytest.raises(RuntimeError, match="GT byte illegal"):
        for index in range(len(dataset)):
            for _ in range(8):  # holes are resampled; offset 40 is in the context
                dataset[index]


def test_mask_file_size_mismatch_is_rejected(tmp_path):
    dataset, _, masks = _dataset(tmp_path)
    (tmp_path / "masks" / "clip.h264.masks").write_bytes(masks[:-32])
    with pytest.raises(ValueError, match="expected"):
        dataset[0]


def _mask_bits(*legal_bytes: int) -> torch.Tensor:
    value = sum(1 << b for b in legal_bytes)
    return torch.tensor(list(value.to_bytes(32, "little")), dtype=torch.uint8)


def test_syntax_term_is_minus_log_legal_mass_with_eos_illegal():
    logits = torch.full((1, 3, EOS_VOCAB_SIZE), -30.0)
    logits[0, 0, 5] = math.log(2)   # legal byte
    logits[0, 0, 9] = math.log(3)   # illegal byte
    logits[0, 0, SEQ_EOS_ID] = math.log(5)  # EOS is illegal mid-span
    logits[0, 1, 7] = 0.0           # second row unconstrained (all-zero mask)
    logits[0, 2, 1] = 0.0           # ignored target
    targets = torch.tensor([[5, 7, IGNORE_INDEX]])
    masks = torch.zeros((1, 3, 32), dtype=torch.uint8)
    masks[0, 0] = _mask_bits(5)
    masks[0, 2] = _mask_bits(1)
    terms = byte_training_loss_terms(
        logits, targets, syntax_masks=masks, syntax_loss_weight=0.5
    )
    total = 2 + 3 + 5 + (EOS_VOCAB_SIZE - 3) * math.exp(-30.0)
    # Non-EOS control ids are neutral (neither legal byte nor illegal).
    legal = 2 + (EOS_VOCAB_SIZE - BYTE_VOCAB_SIZE - 1) * math.exp(-30.0)
    assert terms["syntax_positions"].item() == 1
    assert math.isclose(terms["syntax_ce"].item(), math.log(total / legal), rel_tol=1e-5)
    assert math.isclose(terms["syntax_illegal_mass"].item(), 1 - legal / total, rel_tol=1e-5)
    assert math.isclose(terms["syntax_eos_mass"].item(), 5 / total, rel_tol=1e-5)
    assert math.isclose(
        terms["objective"].item(),
        terms["full_ce"].item() + 0.5 * terms["syntax_ce"].item(),
        rel_tol=1e-6,
    )


def test_syntax_term_gradient_is_finite_without_control_tokens():
    # Byte-only vocabulary: unconstrained rows have no neutral ids at all.
    logits = torch.randn(1, 4, BYTE_VOCAB_SIZE, requires_grad=True)
    targets = torch.tensor([[3, 4, 5, IGNORE_INDEX]])
    masks = torch.zeros((1, 4, 32), dtype=torch.uint8)
    masks[0, 0] = _mask_bits(3, 8)
    flat = logits.reshape(-1, BYTE_VOCAB_SIZE)
    terms = syntax_legality_terms(
        flat,
        torch.logsumexp(flat, dim=-1),
        targets.reshape(-1),
        targets.reshape(-1) != IGNORE_INDEX,
        masks,
        targets.shape,
    )
    terms["syntax_ce"].backward()
    assert torch.isfinite(logits.grad).all()
    # Only the constrained row receives gradient, and legal logits are pushed up.
    assert logits.grad[0, 1:].abs().sum() == 0
    assert logits.grad[0, 0, 3] < 0 and logits.grad[0, 0, 100] > 0


def test_missing_masks_with_positive_weight_fails_loudly():
    logits = torch.zeros(1, 1, EOS_VOCAB_SIZE)
    with pytest.raises(ValueError, match="requires syntax_masks"):
        byte_training_loss_terms(logits, torch.tensor([[1]]), syntax_loss_weight=1.0)
