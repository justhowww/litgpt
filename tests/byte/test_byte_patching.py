from __future__ import annotations

import torch

from litgpt.byte.data import (
    FIM_END_ID,
    IGNORE_INDEX,
    PAD_ID,
    REGION_BRIDGE,
    REGION_PAD,
    REGION_TARGET,
    SEQ_EOS_ID,
    SLICE_BOS_ID,
    VOCAB_SIZE,
    patch_byte_sample,
)
from litgpt.byte.training import byte_training_loss
from litgpt.byte.training import get_model_inputs_and_targets
from litgpt.byte.megabyte_inference import (
    MegabyteInference,
    megabyte_max_new_bytes,
    megabyte_prompt_patches,
)
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


def test_ar_patch_layout_preserves_eos_as_final_local_target():
    sample = _sample(
        [SLICE_BOS_ID, 10, 11, 12, 13, 14, 15, 16],
        [10, 11, 12, 13, 14, 15, 16, SEQ_EOS_ID],
        REGION_TARGET,
    )

    patched = patch_byte_sample(sample, patch_size=8)

    assert patched["input_ids"].tolist() == [
        [PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, SLICE_BOS_ID],
    ]
    assert patched["labels"].tolist() == [
        [10, 11, 12, 13, 14, 15, 16, SEQ_EOS_ID],
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


def _patch_config(**kwargs) -> Config:
    return Config(
        block_size=4,
        n_layer=1,
        n_embd=16,
        n_head=4,
        vocab_size=VOCAB_SIZE,
        padding_multiple=8,
        byte_patch_size=4,
        megabyte_local_n_layer=1,
        megabyte_local_n_embd=16,
        megabyte_local_n_head=4,
        **kwargs,
    )


def test_megabyte_uses_lossless_global_patch_embedding():
    torch.manual_seed(7)
    model = GPT(_patch_config()).eval()
    input_ids = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])

    embedded = model.megabyte_global_wte(input_ids)
    global_inputs = model._megabyte_global_embed(input_ids, None, None)

    assert model.megabyte_global_wte.embedding_dim == 4
    assert global_inputs.shape == (1, 2, 16)
    assert torch.equal(global_inputs, embedded.reshape(1, 2, 16))
    assert not hasattr(model, "patch_input_proj")


def test_megabyte_patch_embedding_accepts_region_and_raw_byte_offsets():
    torch.manual_seed(8)
    model = GPT(
        _patch_config(
            use_region_id=True,
            use_offset_id=True,
            offset_vocab_size=16,
        )
    ).eval()
    input_ids = torch.tensor([[[1, 2, 3, 4]]])
    region_ids = torch.tensor([[[REGION_TARGET] * 4]])
    offset_ids = torch.tensor([[[4, 5, 6, 7]]])
    targets = torch.tensor([[[10, 11, 12, 13]]])

    logits = model(
        input_ids,
        region_ids=region_ids,
        offset_ids=offset_ids,
        patch_targets=targets,
    )

    assert logits.shape == (1, 1, 4, model.config.padded_vocab_size)


def test_megabyte_local_transformer_is_causal_within_each_patch():
    torch.manual_seed(7)
    config = _patch_config()
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


def test_megabyte_incremental_local_logits_match_teacher_forcing():
    torch.manual_seed(9)
    model = GPT(_patch_config()).eval()
    input_ids = torch.tensor([[[PAD_ID, PAD_ID, PAD_ID, SLICE_BOS_ID]]])
    targets = torch.tensor([[[10, 11, 12, 13]]])

    teacher_logits = model(input_ids, patch_targets=targets)
    global_output = model.megabyte_global_forward(input_ids)[:, 0]
    incremental = [
        model.megabyte_local_next_logits(
            global_output, targets[:, 0, :byte_index]
        )
        for byte_index in range(4)
    ]

    assert torch.allclose(
        torch.stack(incremental, dim=1),
        teacher_logits[:, 0],
        atol=1e-6,
        rtol=1e-5,
    )


def test_megabyte_inference_feeds_completed_patch_to_global_cache():
    torch.manual_seed(10)
    model = GPT(_patch_config()).eval()
    prompt = torch.tensor([[SLICE_BOS_ID, 1, 2, 3, 4]])
    regions = torch.full_like(prompt, REGION_TARGET)
    offsets = torch.arange(prompt.size(1)).unsqueeze(0)
    patched_prompt, _, _ = megabyte_prompt_patches(
        prompt, regions, offsets, patch_size=4
    )
    generated_patch = [10, 11, 12, 13]

    model.set_kv_cache(batch_size=1, max_seq_length=model.max_seq_length)
    try:
        model.megabyte_global_forward(
            patched_prompt,
            input_pos=torch.arange(patched_prompt.size(1)),
            input_pos_maxp1=patched_prompt.size(1),
        )
        expected_next_global = model.megabyte_global_forward(
            torch.tensor([[generated_patch]]),
            input_pos=torch.tensor([patched_prompt.size(1)]),
            input_pos_maxp1=patched_prompt.size(1) + 1,
        )[:, -1]
        expected_logits = model.megabyte_local_next_logits(
            expected_next_global, torch.empty((1, 0), dtype=torch.long)
        )
    finally:
        model.clear_kv_cache()

    with MegabyteInference(model, prompt, regions, offsets, torch.device("cpu")) as state:
        for byte_index, token in enumerate(generated_patch):
            state.next_logits()
            state.append(token, REGION_TARGET, byte_index)
        actual_logits = state.next_logits()

    assert torch.allclose(actual_logits, expected_logits, atol=1e-6, rtol=1e-5)
    assert megabyte_max_new_bytes(model, prompt.size(1)) == 12


def test_megabyte_start_code_metadata_can_cross_patch_boundary():
    torch.manual_seed(12)
    model = GPT(
        _patch_config(
            use_region_id=True,
            use_offset_id=True,
            offset_vocab_size=32,
        )
    ).eval()
    prompt = torch.tensor([[SLICE_BOS_ID]])
    regions = torch.full_like(prompt, REGION_TARGET)
    offsets = torch.zeros_like(prompt)

    with MegabyteInference(model, prompt, regions, offsets, torch.device("cpu")) as state:
        for byte_index, token in enumerate((9, 0, 0, 0)):
            state.next_logits()
            state.append(token, REGION_TARGET, byte_index)
        state.next_logits()  # commits the first generated patch
        state.append(1, REGION_TARGET, 4)
        state.rewrite_recent_metadata(
            4,
            region_ids=[REGION_TARGET] * 4,
            offset_ids=[0, 1, 2, 3],
        )
        state.next_logits()  # refreshes the corrected prior-patch KV entry

        assert state._completed[-1].offset_ids == [0, 0, 1, 2]
        assert state._current_offsets == [3]


def test_megabyte_global_and_local_towers_receive_gradients():
    torch.manual_seed(11)
    model = GPT(_patch_config()).train()
    input_ids = torch.tensor(
        [[[PAD_ID, PAD_ID, PAD_ID, SLICE_BOS_ID], [10, 11, 12, 13]]]
    )
    targets = torch.tensor([[[10, 11, 12, 13], [14, 15, 16, 17]]])

    logits = model(input_ids, patch_targets=targets)
    byte_training_loss(logits, targets).backward()

    assert model.megabyte_global_wte.weight.grad is not None
    assert model.transformer.h[0].attn.qkv.weight.grad is not None
    assert model.megabyte_global_to_local.weight.grad is not None
    assert model.megabyte_local.h[0].attn.qkv.weight.grad is not None
    assert model.megabyte_local.wte.weight.grad is not None


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
