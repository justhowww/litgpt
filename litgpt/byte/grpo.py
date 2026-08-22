"""Online GRPO for decoder-scored H.264 byte generation (megabyte-aware).

Sibling of :mod:`litgpt.byte.mrt`, structured the same way (candidate
generation, external decoder scoring, and the training loop kept separate),
but implementing literal GRPO -- uniform per-candidate weighting, advantage
normalized by the group's reward mean and std -- rather than MRT's
q-weighted expected-risk estimator. Generation is unmasked (no
``h264_mask`` constrained decoding): the decode-failure penalty in the
reward is what teaches validity.

Candidates are sampled without gradient tracking via the batched megabyte
generator (:func:`litgpt.byte.megabyte_inference.megabyte_generate_batch`),
then scored with a separate teacher-forced forward pass over the whole
group at once (see ``group_token_log_probabilities``). Batching the scoring
forward pass matters here specifically because GRPO's uniform weighting
gives (unlike MRT's q-weighting) essentially every candidate a nonzero
gradient coefficient -- an unbatched per-candidate loop would mean
``group_size`` separate forward+backward calls per optimizer step.

Supports ``mu > 1``: reusing one sampled group for several gradient steps
via the PPO-style clipped ratio surrogate (``grpo_clipped_loss``). At
``mu == 1`` (the default) the ratio is ~1 by construction (nothing has
changed the policy since sampling), so the clip is inert and the objective
reduces to plain advantage-weighted log-prob. ``mu > 1`` is what makes the
ratio/clip mechanism load-bearing: it caps how far a single inner step can
move probability on tokens that already drifted from the sampling policy,
which is what makes reusing a rollout for multiple updates safe.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from litgpt.byte.data import BYTE_VOCAB_SIZE, IGNORE_INDEX
from litgpt.byte.megabyte_inference import (
    megabyte_generate_batch,
    megabyte_teacher_forced_sample,
)
from litgpt.byte.reconstruction import (
    ReconstructionSample,
    _unwrap_model,
    decode_frame,
    image_psnr,
    replace_target_nal,
)

GRPO_EPS = 1e-6


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for sparse online GRPO updates."""

    interval: int = 0
    start_step: int = 0
    group_size: int = 64
    ar_pool_size: int = 16
    fim_pool_size: int = 16
    max_target_bytes: int = 2048
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    kl_coeff: float = 0.02
    psnr_cap: float = 40.0
    decode_failure_reward: float = -1.0
    timeout_sec: int = 30
    decode_workers: int = 8
    ffmpeg_binary: str = "ffmpeg"
    # Inner gradient steps taken on one sampled group before resampling, via
    # the PPO-style clipped ratio surrogate. mu=1 (default) is the on-policy
    # case where the clip is a no-op; mu>1 reuses the (expensive) rollout for
    # extra updates, trading rollout cost for a small off-policy-ness that the
    # clip keeps bounded.
    mu: int = 1
    clip_range: float = 0.2

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def validate(self) -> None:
        if self.interval < 0:
            raise ValueError("GRPO interval must be non-negative")
        if self.start_step < 0:
            raise ValueError("GRPO start step must be non-negative")
        if self.group_size < 2:
            raise ValueError("GRPO requires at least two candidates per group")
        if self.ar_pool_size <= 0 or self.fim_pool_size <= 0:
            raise ValueError("GRPO AR and FIM pool sizes must be positive")
        if self.max_target_bytes <= 0:
            raise ValueError("GRPO max target bytes must be positive")
        if self.temperature <= 0:
            raise ValueError("GRPO temperature must be positive")
        if self.kl_coeff < 0:
            raise ValueError("GRPO KL coefficient must be non-negative")
        if self.psnr_cap <= 0:
            raise ValueError("GRPO PSNR cap must be positive")
        if self.timeout_sec <= 0 or self.decode_workers <= 0:
            raise ValueError("GRPO decoder timeout and worker count must be positive")
        if self.mu < 1:
            raise ValueError("GRPO mu (inner steps per group) must be at least 1")
        if not 0 < self.clip_range < 1:
            raise ValueError("GRPO clip_range must be in (0, 1)")


@dataclass(frozen=True)
class PreparedGRPOStep:
    """Decoder-scored candidate group ready for a batched scoring forward."""

    sample: ReconstructionSample
    task: str
    candidates: tuple[bytes, ...]
    rewards: Tensor
    advantages: Tensor
    decoded: Tensor
    psnrs: Tensor
    decode_rate: float
    mean_reward: float


def should_run_grpo(next_step: int, config: GRPOConfig) -> bool:
    """Return whether the upcoming optimizer step includes a GRPO update."""
    return (
        config.enabled
        and next_step > config.start_step
        and (next_step - config.start_step) % config.interval == 0
    )


def candidate_reward(
    stream: bytes,
    sample: ReconstructionSample,
    reference: Tensor,
    candidate: bytes,
    config: GRPOConfig,
) -> tuple[float, bool, float | None]:
    """Splice, strictly decode, and score one candidate.

    Returns ``(reward, decoded, psnr)``. A failed/mismatched decode gets the
    fixed ``decode_failure_reward`` floor; a successful decode gets PSNR
    normalized into ``[0, 1]`` against ``psnr_cap``.
    """
    candidate_stream = replace_target_nal(stream, sample, candidate)
    frame, status = decode_frame(
        candidate_stream,
        sample.frame_index,
        config.ffmpeg_binary,
        config.timeout_sec,
        strict_syntax=True,
    )
    if frame is None or frame.shape != reference.shape:
        return config.decode_failure_reward, False, None
    psnr = image_psnr(reference, frame)
    psnr_value = config.psnr_cap if math.isinf(psnr) else psnr
    normalized = max(0.0, min(psnr_value, config.psnr_cap)) / config.psnr_cap
    return normalized, status == "decoded", psnr_value


def grpo_advantages(rewards: Tensor) -> Tensor:
    """Group-relative advantage: reward z-scored within the group."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + GRPO_EPS)


@torch.inference_mode()
def prepare_grpo_step(
    model: nn.Module,
    sample: ReconstructionSample,
    config: GRPOConfig,
    device: torch.device,
) -> PreparedGRPOStep | None:
    """Sample a group, decode-score every candidate, and compute advantages.

    Returns ``None`` if the reference frame fails to decode, the prompt/
    target doesn't fit the model, or the group is degenerate (every
    candidate scored identically, so the advantage z-score is undefined and
    there's no gradient signal worth a backward pass).
    """
    raw_model = _unwrap_model(model)
    stream = sample.h264_path.read_bytes()
    reference, _ = decode_frame(
        stream,
        sample.frame_index,
        config.ffmpeg_binary,
        config.timeout_sec,
        strict_syntax=True,
    )
    if reference is None:
        return None

    was_training = raw_model.training
    raw_model.eval()
    try:
        candidates = megabyte_generate_batch(
            raw_model,
            sample,
            device,
            config.group_size,
            config.temperature,
            config.top_k,
            config.top_p,
        )
    finally:
        raw_model.train(was_training)
    if not candidates:
        return None

    with ThreadPoolExecutor(max_workers=config.decode_workers) as executor:
        results = list(
            executor.map(
                lambda candidate: candidate_reward(stream, sample, reference, candidate, config),
                candidates,
            )
        )
    rewards = torch.tensor([reward for reward, _, _ in results], device=device)
    decoded = torch.tensor([ok for _, ok, _ in results], dtype=torch.bool, device=device)
    psnrs = torch.tensor(
        [float("nan") if psnr is None else psnr for _, _, psnr in results], device=device
    )
    if float(rewards.std(unbiased=False)) < GRPO_EPS:
        return None

    return PreparedGRPOStep(
        sample=sample,
        task=sample.task,
        candidates=tuple(candidates),
        rewards=rewards,
        advantages=grpo_advantages(rewards),
        decoded=decoded,
        psnrs=psnrs,
        decode_rate=float(decoded.float().mean()),
        mean_reward=float(rewards.mean()),
    )


def build_group_patch_inputs(
    sample: ReconstructionSample,
    candidates: tuple[bytes, ...],
    patch_size: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    """Build a batched, patch-aligned teacher-forcing input for a candidate group.

    All candidates share ``sample.target_length`` -- the batched generator
    always generates exactly that many bytes with no early EOS stopping --
    so every candidate patches to an identical ``(T, P)`` grid. That means
    no padding or length-masking is needed: the whole group goes through the
    model in a single forward call.
    """
    prompt_ids = sample.prompt_ids.to(device)
    prompt_region_ids = sample.prompt_region_ids.to(device)
    prompt_offset_ids = sample.prompt_offset_ids.to(device)

    patched_inputs: list[Tensor] = []
    patched_labels: list[Tensor] = []
    patched_regions: list[Tensor] = []
    patched_offsets: list[Tensor] = []
    for candidate in candidates:
        target_tokens = torch.tensor(list(candidate), dtype=torch.long, device=device)
        if target_tokens.numel() != sample.target_length:
            raise ValueError("candidate length must equal sample.target_length")
        appended_ids = target_tokens[:-1]
        input_ids = torch.cat((prompt_ids, appended_ids))
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[prompt_ids.numel() - 1 :] = target_tokens
        region_ids = torch.cat(
            (
                prompt_region_ids,
                torch.full(
                    (appended_ids.numel(),),
                    sample.generation_region_id,
                    dtype=torch.long,
                    device=device,
                ),
            )
        )
        offset_ids = torch.cat(
            (
                prompt_offset_ids,
                torch.arange(
                    sample.generation_offset_start,
                    sample.generation_offset_start + appended_ids.numel(),
                    dtype=torch.long,
                    device=device,
                ),
            )
        )
        patched = megabyte_teacher_forced_sample(
            input_ids, labels, region_ids, offset_ids, patch_size
        )
        patched_inputs.append(patched["input_ids"])
        patched_labels.append(patched["labels"])
        patched_regions.append(patched["region_ids"])
        patched_offsets.append(patched["offset_ids"])

    inputs = {
        "idx": torch.cat(patched_inputs, dim=0),
        "region_ids": torch.cat(patched_regions, dim=0),
        "offset_ids": torch.cat(patched_offsets, dim=0),
        "patch_targets": torch.cat(patched_labels, dim=0),
    }
    return inputs, inputs["patch_targets"]


def group_token_log_probabilities(
    model: nn.Module, inputs: dict[str, Tensor], labels: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(per-token log-probs, supervised mask)``, each shaped ``(B, T, P)``."""
    logits = _unwrap_model(model)(**inputs)
    supervised = labels != IGNORE_INDEX
    target = labels.clamp_min(0)
    log_probs = torch.log_softmax(logits[..., :BYTE_VOCAB_SIZE].float(), dim=-1)
    gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return gathered, supervised


def mean_log_probability(gathered: Tensor, supervised: Tensor) -> Tensor:
    """Length-normalized per-candidate log-prob, shape ``(B,)``."""
    masked = gathered.masked_fill(~supervised, 0.0)
    counts = supervised.sum(dim=(1, 2)).clamp_min(1)
    return masked.sum(dim=(1, 2)) / counts


def token_kl_to_reference(
    policy_gathered: Tensor, reference_gathered: Tensor, supervised: Tensor
) -> Tensor:
    """Per-candidate KL(policy || reference) via the k3 estimator.

    ``log_ratio = log p_ref(token) - log p_policy(token)``, evaluated at
    tokens sampled from the current policy -- exactly what the k3 estimator
    (Schulman, http://joschu.net/blog/kl-approx.html) requires for an
    unbiased, always-non-negative KL estimate.
    """
    log_ratio = (reference_gathered - policy_gathered).masked_fill(~supervised, 0.0)
    per_token_kl = (torch.expm1(log_ratio) - log_ratio).masked_fill(~supervised, 0.0)
    counts = supervised.sum(dim=(1, 2)).clamp_min(1)
    return per_token_kl.sum(dim=(1, 2)) / counts


def grpo_clipped_loss(
    gathered: Tensor,
    old_gathered: Tensor,
    supervised: Tensor,
    advantages: Tensor,
    reference_gathered: Tensor | None,
    kl_coeff: float,
    clip_range: float,
) -> tuple[Tensor, dict[str, float]]:
    """PPO-style clipped-surrogate GRPO loss for one inner step.

    ``old_gathered`` is the (detached) per-token log-prob under the policy
    that generated the candidates -- fixed for all ``mu`` inner steps on this
    group. At the first inner step ``gathered ~= old_gathered`` (nothing has
    updated the policy yet), so ``ratio ~= 1`` and the clip is inert; it only
    becomes load-bearing from the second inner step onward, once the first
    step's update has moved the policy away from the sampling distribution.
    """
    ratio = torch.exp(gathered - old_gathered)
    advantage = advantages.view(-1, 1, 1)
    surrogate = torch.minimum(
        ratio * advantage, torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage
    )
    surrogate = surrogate.masked_fill(~supervised, 0.0)
    counts = supervised.sum(dim=(1, 2)).clamp_min(1)
    policy_loss = -(surrogate.sum(dim=(1, 2)) / counts).mean()

    with torch.no_grad():
        has_supervised = bool(supervised.any())
        clipped = (ratio < 1 - clip_range) | (ratio > 1 + clip_range)
        clip_fraction = float(clipped[supervised].float().mean()) if has_supervised else 0.0
        ratio_mean = float(ratio[supervised].mean()) if has_supervised else 1.0

    metrics = {
        "grpo/policy_loss": float(policy_loss.detach()),
        "grpo/clip_fraction": clip_fraction,
        "grpo/ratio_mean": ratio_mean,
    }

    loss = policy_loss
    if reference_gathered is not None and kl_coeff > 0:
        kl_per_candidate = token_kl_to_reference(gathered, reference_gathered, supervised)
        kl_term = kl_coeff * kl_per_candidate.mean()
        loss = loss + kl_term
        metrics["grpo/kl_to_reference"] = float(kl_per_candidate.mean().detach())
        metrics["grpo/kl_term"] = float(kl_term.detach())

    return loss, metrics
