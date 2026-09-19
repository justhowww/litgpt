"""Architecture presets for byte-domain AR/FIM pretraining."""

from __future__ import annotations

from typing import Any

from litgpt.config import Config


BYTE_MODEL_ARCHITECTURES = ("pythia", "qwen3")


def _qwen3_kv_heads(query_heads: int, *, component: str) -> int:
    """Use Qwen3's 4:1 query-to-KV-head ratio for the project model."""
    if query_heads % 4:
        raise ValueError(
            f"Qwen3 {component} attention requires a head count divisible by 4; "
            f"got {query_heads}"
        )
    return query_heads // 4


def build_byte_model_config(
    *,
    architecture: str,
    name: str,
    block_size: int,
    n_layer: int,
    n_embd: int,
    n_head: int,
    vocab_size: int,
    padding_multiple: int,
    use_region_id: bool,
    use_offset_id: bool,
    offset_vocab_size: int | None,
    byte_patch_size: int,
    megabyte_local_n_layer: int,
    megabyte_local_n_embd: int,
    megabyte_local_n_head: int,
) -> Config:
    """Build a size-controlled Pythia- or Qwen3-style byte model.

    The selected family controls the Transformer block at both MEGABYTE
    levels. Model dimensions remain explicit project hyperparameters; choosing
    ``qwen3`` does not import Qwen's text vocabulary, tokenizer, or weights.
    """
    if architecture not in BYTE_MODEL_ARCHITECTURES:
        choices = ", ".join(BYTE_MODEL_ARCHITECTURES)
        raise ValueError(
            f"Unsupported byte model architecture {architecture!r}; "
            f"choose one of: {choices}"
        )

    common: dict[str, Any] = dict(
        name=name,
        block_size=block_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        vocab_size=vocab_size,
        padding_multiple=padding_multiple,
        use_region_id=use_region_id,
        use_offset_id=use_offset_id,
        offset_vocab_size=offset_vocab_size,
        byte_model_architecture=architecture,
        byte_patch_size=byte_patch_size,
        megabyte_local_n_layer=megabyte_local_n_layer,
        megabyte_local_n_embd=megabyte_local_n_embd,
        megabyte_local_n_head=megabyte_local_n_head,
    )

    if architecture == "pythia":
        return Config(
            **common,
            norm_class_name="LayerNorm",
            norm_eps=1e-5,
            norm_qk=False,
            parallel_residual=True,
            n_query_groups=n_head,
            rotary_percentage=0.25,
            rope_base=10_000,
            bias=True,
            mlp_class_name="GptNeoxMLP",
            intermediate_size=4 * n_embd,
            megabyte_local_n_query_groups=megabyte_local_n_head,
            megabyte_local_intermediate_size=4 * megabyte_local_n_embd,
        )

    # Qwen3 dense blocks use sequential residuals, RMSNorm, full RoPE,
    # bias-free SwiGLU, QK normalization, and grouped-query attention. A 3x
    # SwiGLU hidden width plus a 4:1 query/KV ratio keeps each block close to
    # the parameter/FLOP budget of the Pythia-style 4x GELU/MHA block.
    return Config(
        **common,
        norm_class_name="RMSNorm",
        norm_eps=1e-6,
        norm_qk=True,
        parallel_residual=False,
        n_query_groups=_qwen3_kv_heads(n_head, component="global"),
        rotary_percentage=1.0,
        rope_base=1_000_000,
        bias=False,
        attn_bias=False,
        mlp_class_name="LLaMAMLP",
        intermediate_size=3 * n_embd,
        megabyte_local_n_query_groups=_qwen3_kv_heads(
            megabyte_local_n_head, component="local"
        ),
        megabyte_local_intermediate_size=3 * megabyte_local_n_embd,
    )
