from __future__ import annotations

import pytest

from litgpt.byte.model_config import build_byte_model_config
from litgpt.model import GPT


def _config(architecture: str):
    return build_byte_model_config(
        architecture=architecture,
        name=f"test-{architecture}",
        block_size=4,
        n_layer=1,
        n_embd=16,
        n_head=4,
        vocab_size=264,
        padding_multiple=8,
        use_region_id=False,
        use_offset_id=False,
        offset_vocab_size=None,
        byte_patch_size=4,
        megabyte_local_n_layer=1,
        megabyte_local_n_embd=16,
        megabyte_local_n_head=4,
    )


def test_pythia_architecture_preserves_existing_global_and_local_blocks():
    config = _config("pythia")
    model = GPT(config)

    assert config.byte_model_architecture == "pythia"
    assert config.norm_class_name == "LayerNorm"
    assert config.mlp_class_name == "GptNeoxMLP"
    assert config.parallel_residual
    assert config.rotary_percentage == 0.25
    assert config.n_query_groups == config.n_head
    assert model.megabyte_local_config.norm_class_name == "LayerNorm"
    assert model.megabyte_local_config.mlp_class_name == "GptNeoxMLP"
    assert (
        model.megabyte_local_config.n_query_groups
        == config.megabyte_local_n_head
    )
    assert model.megabyte_local_config.intermediate_size == 64


def test_qwen3_architecture_applies_to_global_and_local_blocks():
    config = _config("qwen3")
    model = GPT(config)

    assert config.byte_model_architecture == "qwen3"
    assert config.norm_class_name == "RMSNorm"
    assert config.mlp_class_name == "LLaMAMLP"
    assert not config.parallel_residual
    assert config.norm_qk
    assert config.rotary_percentage == 1.0
    assert config.n_query_groups == 1
    assert config.intermediate_size == 48

    local = model.megabyte_local_config
    assert local.norm_class_name == "RMSNorm"
    assert local.mlp_class_name == "LLaMAMLP"
    assert not local.parallel_residual
    assert local.norm_qk
    assert local.rotary_percentage == 1.0
    assert local.n_query_groups == 1
    assert local.intermediate_size == 48


def test_qwen3_requires_head_counts_divisible_by_four():
    with pytest.raises(ValueError, match="global attention.*divisible by 4"):
        build_byte_model_config(
            architecture="qwen3",
            name="bad-qwen3",
            block_size=4,
            n_layer=1,
            n_embd=18,
            n_head=6,
            vocab_size=264,
            padding_multiple=8,
            use_region_id=False,
            use_offset_id=False,
            offset_vocab_size=None,
            byte_patch_size=2,
            megabyte_local_n_layer=1,
            megabyte_local_n_embd=16,
            megabyte_local_n_head=4,
        )
