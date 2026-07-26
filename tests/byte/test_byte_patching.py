from __future__ import annotations

import torch

from litgpt.byte.data import (
    FIM_END_ID,
    IGNORE_INDEX,
    PAD_ID,
    REGION_BRIDGE,
    REGION_PAD,
    REGION_TARGET,
    SLICE_BOS_ID,
    VOCAB_SIZE,
    patch_byte_sample,
)
from litgpt.byte.training import byte_training_loss
from litgpt.byte.training import get_model_inputs_and_targets
from litgpt.config import Config
from litgpt.model import GPT


def _sample(input_ids: list[int], labels: list[int], region: int) -> dict:
    length = len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "region_ids": torch.full((length,), region, dtype=torch.long),
        "offset_ids": torch.arange(length),
        "token_counts": {
            "raw": length,
            "raw_plus_prompt_template": length,
        },
        "sample_meta": {"task": "ar"},
    }


def test_ar_patch_layout_predicts_next_patch_without_byte_leakage():
    sample = _sample(
        [SLICE_BOS_ID, 10, 11, 12, 13, 14],
        [10, 11, 12, 13, 14, 15],
        REGION_TARGET,
    )

    patched = patch_byte_sample(sample, patch_size=4)

    assert patched["input_ids"].tolist() == [
        [PAD_ID, PAD_ID, PAD_ID, SLICE_BOS_ID],
        [10, 11, 12, 13],
    ]
    assert patched["labels"].tolist() == [
        [10, 11, 12, 13],
        [14, 15, IGNORE_INDEX, IGNORE_INDEX],
    ]
    assert patched["labels"][patched["labels"] != IGNORE_INDEX].tolist() == [
        10,
        11,
        12,
        13,
        14,
        15,
    ]
    # Completed target bytes reuse the input-side attributes from the next
    # positions of the original shifted byte sample.
    assert patched["offset_ids"].tolist() == [
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ]


def test_fim_prompt_ends_before_any_missing_byte_enters_patch_input():
    sample = _sample(
        [90, 91, FIM_END_ID, 10, 11],
        [IGNORE_INDEX, IGNORE_INDEX, 10, 11, 12],
        REGION_BRIDGE,
    )
    sample["sample_meta"]["task"] = "fim"

    patched = patch_byte_sample(sample, patch_size=4)

    assert patched["input_ids"].tolist() == [
        [PAD_ID, 90, 91, FIM_END_ID],
    ]
    assert patched["labels"].tolist() == [
        [10, 11, 12, IGNORE_INDEX],
    ]
    assert patched["target_region_ids"].tolist() == [
        [REGION_BRIDGE, REGION_BRIDGE, REGION_BRIDGE, REGION_PAD],
    ]


def test_patch_decoder_is_causal_within_each_transformer_position():
    torch.manual_seed(7)
    config = Config(
        block_size=4,
        n_layer=1,
        n_embd=16,
        n_head=4,
        vocab_size=VOCAB_SIZE,
        padding_multiple=8,
        byte_patch_size=4,
    )
    model = GPT(config).eval()
    input_ids = torch.tensor([[[PAD_ID, PAD_ID, PAD_ID, SLICE_BOS_ID]]])
    targets_a = torch.tensor([[[10, 11, 12, 13]]])
    targets_b = targets_a.clone()
    targets_b[..., 2] = 99

    logits_a = model(input_ids, patch_targets=targets_a)
    logits_b = model(input_ids, patch_targets=targets_b)

    assert logits_a.shape == (1, 1, 4, config.padded_vocab_size)
    # Changing target byte 2 cannot alter logits for bytes 0, 1, or 2.
    assert torch.equal(logits_a[..., :3, :], logits_b[..., :3, :])
    # It is allowed to alter byte 3, which causally consumes target byte 2.
    assert not torch.equal(logits_a[..., 3, :], logits_b[..., 3, :])


def test_byte_loss_accepts_patch_shaped_logits_and_targets():
    logits = torch.randn(2, 3, 4, VOCAB_SIZE)
    targets = torch.randint(0, 256, (2, 3, 4))
    targets[1, -1, -2:] = IGNORE_INDEX

    loss = byte_training_loss(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_training_adapter_passes_patch_targets_without_reshifting():
    sample = _sample(
        [SLICE_BOS_ID, 10, 11, 12, 13],
        [10, 11, 12, 13, 14],
        REGION_TARGET,
    )
    patched = patch_byte_sample(sample, patch_size=4)
    batch = {
        key: value.unsqueeze(0)
        for key, value in patched.items()
        if key
        in {
            "input_ids",
            "labels",
            "region_ids",
            "offset_ids",
            "target_region_ids",
        }
    }

    model_inputs, targets = get_model_inputs_and_targets(batch, max_seq_length=8)

    assert model_inputs["idx"].shape == (1, 2, 4)
    assert torch.equal(model_inputs["patch_targets"], targets)
    assert torch.equal(targets, batch["labels"])
