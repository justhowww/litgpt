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
from pathlib import Path

import torch
from torch import Tensor, nn

from litgpt.byte import h264_mask as HM
from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    IGNORE_INDEX,
    REGION_PAD,
    SEQ_EOS_ID,
    _pad_patch_axis,
)
from litgpt.byte.free_run_eval import (
    FreeRunSample,
    _survival_and_validity,
    megabyte_generate_batch_frames,
)
from litgpt.byte.megabyte_inference import (
    GeneratedCandidate,
    megabyte_generate_batch,
    megabyte_generate_batch_eos,
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
    # Candidates per context and per DDP rank.  Each rank owns a different
    # context; their independently normalized policy gradients are averaged.
    group_size: int = 64
    ar_pool_size: int = 16
    fim_pool_size: int = 16
    # ``fixed`` preserves the original materialized context pools. ``online``
    # visits the complete training split in deterministic shuffled epochs and
    # builds only the current prompt, drawing a fresh FIM hole on every revisit.
    context_sampling: str = "fixed"
    context_seed: int = 42
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
    # When True, rollouts use learned-EOS generation (unknown length,
    # candidates stop by emitting SEQ_EOS_ID, widened budget) instead of
    # oracle-length generation (told the true target length, EOS masked
    # out). Requires the checkpoint to have been trained with --use-eos.
    learned_eos: bool = False
    generation_budget_multiplier: float = 2.0
    # Window-mode AR pool: reuses litgpt.byte.free_run_eval's clip selection
    # (prefix_frames/cont_frames/slice_layout mirror FreeRunEvalConfig's
    # defaults) via megabyte_generate_batch_frames, which stops each
    # candidate independently once it closes ar_cont_frames real frames.
    # Only valid for checkpoints with region/offset conditioning disabled.
    ar_prefix_frames: int = 8
    ar_cont_frames: int = 4
    ar_slice_layout: str = "macroblock"

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
        if self.context_sampling not in {"fixed", "online"}:
            raise ValueError("GRPO context sampling must be 'fixed' or 'online'")
        if self.context_seed < 0:
            raise ValueError("GRPO context seed must be non-negative")
        if (
            self.context_sampling == "fixed"
            and (self.ar_pool_size <= 0 or self.fim_pool_size <= 0)
        ):
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
        if self.generation_budget_multiplier <= 1:
            raise ValueError("GRPO generation budget multiplier must exceed 1")
        if self.ar_prefix_frames <= 0 or self.ar_cont_frames <= 0:
            raise ValueError("GRPO AR prefix/continuation frame counts must be positive")
        if self.ar_slice_layout not in HM.SLICE_LAYOUTS:
            raise ValueError(f"GRPO AR slice layout must be one of {HM.SLICE_LAYOUTS}")


@dataclass(frozen=True)
class PreparedGRPOStep:
    """Decoder-scored candidate group ready for a batched scoring forward."""

    sample: ReconstructionSample | FreeRunSample
    task: str
    candidates: list[GeneratedCandidate]
    rewards: Tensor
    advantages: Tensor
    decoded: Tensor
    psnrs: Tensor
    decode_rate: float
    mean_reward: float


@dataclass(frozen=True)
class GRPOPreparationResult:
    """One rank's rollout preparation outcome and diagnostics.

    ``prepared`` remains available for a zero-variance group so the rank can
    enter DDP backward with zero policy advantage while useful peer ranks
    still update.  It is ``None`` only when scoring cannot be constructed at
    all, for example when the reference frame or candidate generation fails.
    """

    status: str
    prepared: PreparedGRPOStep | None = None
    candidate_count: int = 0
    stopped_count: int = 0
    decoded_count: int = 0
    reward_std: float | None = None
    has_policy_signal: bool = False


def _scored_preparation(
    *,
    sample: ReconstructionSample | FreeRunSample,
    task: str,
    candidates: list[GeneratedCandidate],
    results: list[tuple[float, bool, float | None]],
    device: torch.device,
) -> GRPOPreparationResult:
    """Package scored candidates without discarding zero-variance groups."""
    if len(results) != len(candidates):
        raise ValueError("GRPO candidate/result count mismatch")
    rewards = torch.tensor([reward for reward, _, _ in results], device=device)
    decoded = torch.tensor(
        [ok for _, ok, _ in results], dtype=torch.bool, device=device
    )
    psnrs = torch.tensor(
        [float("nan") if psnr is None else psnr for _, _, psnr in results],
        device=device,
    )
    reward_std = float(rewards.std(unbiased=False))
    has_policy_signal = reward_std >= GRPO_EPS
    stopped_count = sum(int(candidate.stopped) for candidate in candidates)
    decoded_count = int(decoded.sum())
    if decoded_count > stopped_count:
        raise RuntimeError(
            "GRPO invariant violated: a candidate decoded before satisfying its "
            "task stopping condition"
        )
    prepared = PreparedGRPOStep(
        sample=sample,
        task=task,
        candidates=candidates,
        rewards=rewards,
        advantages=grpo_advantages(rewards),
        decoded=decoded,
        psnrs=psnrs,
        decode_rate=float(decoded.float().mean()),
        mean_reward=float(rewards.mean()),
    )
    return GRPOPreparationResult(
        status="ready" if has_policy_signal else "zero_reward_variance",
        prepared=prepared,
        candidate_count=len(candidates),
        stopped_count=stopped_count,
        decoded_count=decoded_count,
        reward_std=reward_std,
        has_policy_signal=has_policy_signal,
    )


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
    candidate: GeneratedCandidate,
    config: GRPOConfig,
) -> tuple[float, bool, float | None]:
    """Splice, strictly decode, and score one candidate.

    Returns ``(reward, decoded, psnr)``. A failed/mismatched decode gets the
    fixed ``decode_failure_reward`` floor; a successful decode gets PSNR
    normalized into ``[0, 1]`` against ``psnr_cap``.

    Under learned-EOS generation, a candidate that ran to the budget ceiling
    without ever emitting SEQ_EOS gets the same decode-failure floor,
    skipping the decode entirely. Decode success alone doesn't reliably
    punish an unbounded-length candidate (trailing garbage past the true
    span can decode fine, or fail for reasons unrelated to length), so
    without this the model has no direct signal that it failed to stop --
    only the CE pretraining objective (which does supervise EOS directly)
    would be teaching that skill, with GRPO never touching it.
    """
    if config.learned_eos and not candidate.stopped:
        return config.decode_failure_reward, False, None
    candidate_stream = replace_target_nal(stream, sample, candidate.data)
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
) -> GRPOPreparationResult:
    """Sample a group, decode-score every candidate, and compute advantages.

    A zero-variance group is returned with zero advantages rather than being
    discarded.  This is important under DDP: the local rank can contribute a
    zero policy gradient without cancelling useful peer-rank groups.
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
        return GRPOPreparationResult(status="reference_decode_failed")

    was_training = raw_model.training
    raw_model.eval()
    try:
        if config.learned_eos:
            candidates = megabyte_generate_batch_eos(
                raw_model,
                sample,
                device,
                config.group_size,
                config.temperature,
                config.top_k,
                config.top_p,
                config.generation_budget_multiplier,
            )
        else:
            raw_candidates = megabyte_generate_batch(
                raw_model,
                sample,
                device,
                config.group_size,
                config.temperature,
                config.top_k,
                config.top_p,
            )
            candidates = (
                [GeneratedCandidate(data=c, stopped=False) for c in raw_candidates]
                if raw_candidates
                else None
            )
    finally:
        raw_model.train(was_training)
    if not candidates:
        return GRPOPreparationResult(status="generation_failed")

    with ThreadPoolExecutor(max_workers=config.decode_workers) as executor:
        results = list(
            executor.map(
                lambda candidate: candidate_reward(stream, sample, reference, candidate, config),
                candidates,
            )
        )
    return _scored_preparation(
        sample=sample,
        task=sample.task,
        candidates=candidates,
        results=results,
        device=device,
    )


def candidate_reward_ar(
    stream: bytes,
    sample: FreeRunSample,
    references: list[Tensor | None],
    candidate: GeneratedCandidate,
    config: GRPOConfig,
) -> tuple[float, bool, float | None]:
    """Parse, splice, decode, and score one AR continuation candidate.

    Mirrors ``candidate_reward``'s shape (reward, decoded, psnr) but scores
    potentially several continuation frames, not one. Uses
    ``_survival_and_validity`` (cheap, no ffmpeg) for frame-closure counting
    first: a candidate that never closed any valid frame, or that ran to
    budget without closing the target frame count (``not candidate.stopped``
    -- structurally the AR analog of FIM's no-EOS case), gets the same
    decode-failure floor. Otherwise splices the generated continuation in
    (truncated to ``result.survival`` -- bytes before any post-prefix
    desync) and decode+PSNRs whichever frames both parsed clean and have a
    successfully-decoded ground-truth reference, averaging their PSNR.
    """
    if not candidate.stopped:
        return config.decode_failure_reward, False, None
    result = _survival_and_validity(sample.prefix_bytes, candidate.data, sample.cont_frames)
    if result.valid_cont == 0:
        return config.decode_failure_reward, False, None

    candidate_stream = (
        stream[: sample.prefix_end_abs]
        + candidate.data[: result.survival]
        + stream[sample.gt_end_abs :]
    )
    psnrs: list[float] = []
    for k in range(result.valid_cont):
        reference = references[k] if k < len(references) else None
        if reference is None:
            continue
        frame, status = decode_frame(
            candidate_stream,
            sample.frame_index_base + k,
            config.ffmpeg_binary,
            config.timeout_sec,
            strict_syntax=True,
        )
        if frame is None or frame.shape != reference.shape:
            continue
        psnr = image_psnr(reference, frame)
        psnrs.append(config.psnr_cap if math.isinf(psnr) else psnr)
    if not psnrs:
        return config.decode_failure_reward, False, None
    mean_psnr = sum(psnrs) / len(psnrs)
    normalized = max(0.0, min(mean_psnr, config.psnr_cap)) / config.psnr_cap
    return normalized, True, mean_psnr


@torch.inference_mode()
def prepare_grpo_step_ar(
    model: nn.Module,
    sample: FreeRunSample,
    config: GRPOConfig,
    device: torch.device,
) -> GRPOPreparationResult:
    """Sample an AR group, decode-score every candidate, and compute advantages.

    Mirrors ``prepare_grpo_step``'s shape. Ground-truth reference frames are
    decoded once per group (not per candidate) since they don't depend on
    the candidate; ``references[k] is None`` short-circuits scoring for
    continuation frame ``k`` in every candidate that reaches it.
    """
    raw_model = _unwrap_model(model)
    stream = Path(sample.h264_path).read_bytes()
    references: list[Tensor | None] = []
    for k in range(sample.cont_frames):
        frame, _ = decode_frame(
            stream,
            sample.frame_index_base + k,
            config.ffmpeg_binary,
            config.timeout_sec,
            strict_syntax=True,
        )
        references.append(frame)
    if references[0] is None:
        return GRPOPreparationResult(status="reference_decode_failed")

    was_training = raw_model.training
    raw_model.eval()
    try:
        candidates = megabyte_generate_batch_frames(
            raw_model,
            sample,
            device,
            config.group_size,
            config.temperature,
            config.top_k,
            config.top_p,
            config.ar_cont_frames,
            config.generation_budget_multiplier,
            config.ar_slice_layout,
        )
    finally:
        raw_model.train(was_training)
    if not candidates:
        return GRPOPreparationResult(status="generation_failed")

    with ThreadPoolExecutor(max_workers=config.decode_workers) as executor:
        results = list(
            executor.map(
                lambda candidate: candidate_reward_ar(stream, sample, references, candidate, config),
                candidates,
            )
        )
    return _scored_preparation(
        sample=sample,
        task="ar",
        candidates=candidates,
        results=results,
        device=device,
    )


def build_group_patch_inputs(
    sample: ReconstructionSample | FreeRunSample,
    candidates: list[GeneratedCandidate],
    patch_size: int,
    device: torch.device,
    *,
    append_eos_on_stop: bool = True,
) -> tuple[dict[str, Tensor], Tensor]:
    """Build a batched, patch-aligned teacher-forcing input for a candidate group.

    Candidates can stop at different lengths -- right-pad every candidate's
    target span to the group's max length before patching, so every
    candidate reaches the same ``(T, P)`` grid and the whole group still
    goes through the model in one forward call. Padded positions get a
    harmless filler byte (0) for ``input_ids``/``labels`` -- always a valid
    gather index, masked out of the loss via the returned ``supervised``
    mask -- and ``REGION_PAD`` for ``region_ids``, matching the padding
    convention ``collate_byte_samples`` already uses elsewhere.

    ``append_eos_on_stop`` controls whether a stopped candidate's target
    sequence gets ``SEQ_EOS_ID`` appended: True for FIM's learned-EOS mode
    (stopping means the model sampled EOS -- it should get gradient credit
    for that prediction), False for AR's frame-count stopping (stopping
    there is a structural fact about the generated bytes, not a sampled
    token -- there is no EOS to score).

    ``sample.generation_region_id``/``generation_offset_start`` (only
    present on ``ReconstructionSample``) default to 0 for ``FreeRunSample``
    inputs, which have no such schedule -- safe because this is only called
    for checkpoints with region/offset conditioning disabled (see
    ``megabyte_generate_batch_frames``), where the values are inert.

    ``FreeRunSample.prompt_ids`` is ``[BOS, prefix bytes...]``, and both real
    AR training (``_build_ar_item``: supervision starts at position 0, so
    ``patch_byte_sample`` treats only the 1-byte BOS as its unsupervised
    "prompt") and ``megabyte_generate_batch_frames`` (which must match, since
    that's what the KV cache was actually built from) patch-align this
    sample with a 1-byte seed -- everything after it, INCLUDING the given
    prefix, packs into the same patch grid as the generated continuation.
    Scoring must reproduce that alignment or it evaluates log-probs against
    a misaligned patch structure the model never actually saw this way
    (confirmed as a real, silent bug via --verify before this was added).
    So for ``FreeRunSample`` inputs, the prefix bytes after the seed are
    prepended to each candidate's target sequence for PATCHING purposes,
    marked invalid (not supervised) via the same validity-mask mechanism
    already used for padding -- distinct from ``ReconstructionSample``,
    whose whole prompt genuinely is patch-aligned as one left-padded block.

    ``patch_byte_sample`` (via ``megabyte_teacher_forced_sample``) requires
    supervision to run through the literal end of the sequence, so the
    filler can't be encoded as ``IGNORE_INDEX`` the way real padding
    normally would be. It also internally enforces that ``input_ids`` really
    is the shifted ``labels`` (real byte content) as a consistency guard --
    calling it a second time with labels swapped for a validity flag against
    the SAME input_ids (an earlier version of this function did exactly
    that) always fails that guard. So the supervised mask here is instead
    computed directly via ``_pad_patch_axis`` -- the same pure,
    value-independent padding primitive ``patch_byte_sample`` itself uses
    for both its prompt and target sides -- applied to a plain validity
    array (1=real content, 0=padding/given-prefix) using the identical
    patch-count geometry the real-content call already produced.
    """
    prompt_ids = sample.prompt_ids.to(device)
    prompt_region_ids = sample.prompt_region_ids.to(device)
    prompt_offset_ids = sample.prompt_offset_ids.to(device)
    generation_region_id = getattr(sample, "generation_region_id", 0)
    generation_offset_start = getattr(sample, "generation_offset_start", 0)

    is_ar = isinstance(sample, FreeRunSample)
    seed_len = 1 if is_ar else prompt_ids.numel()
    seed_ids = prompt_ids[:seed_len]
    seed_region_ids = prompt_region_ids[:seed_len]
    seed_offset_ids = prompt_offset_ids[:seed_len]
    prefix_tail_ids = prompt_ids[seed_len:]
    prefix_tail_region_ids = prompt_region_ids[seed_len:]
    prefix_tail_offset_ids = prompt_offset_ids[seed_len:]
    prefix_tail_len = prefix_tail_ids.numel()

    target_token_lists: list[list[int]] = []
    for candidate in candidates:
        tokens = list(candidate.data)
        if candidate.stopped and append_eos_on_stop:
            tokens.append(SEQ_EOS_ID)
        target_token_lists.append(tokens)
    real_lens = [len(tokens) for tokens in target_token_lists]
    max_len = max(real_lens)
    if max_len == 0:
        raise ValueError("a non-terminating empty candidate has no probability")

    patched_inputs: list[Tensor] = []
    patched_labels: list[Tensor] = []
    patched_regions: list[Tensor] = []
    patched_offsets: list[Tensor] = []
    patched_validity: list[Tensor] = []
    for tokens, real_len in zip(target_token_lists, real_lens):
        pad_len = max_len - real_len
        combined_tokens = torch.cat(
            (
                prefix_tail_ids,
                torch.tensor(tokens, dtype=torch.long, device=device),
                torch.zeros((pad_len,), dtype=torch.long, device=device),
            )
        )
        combined_validity = torch.cat(
            (
                torch.zeros((prefix_tail_len,), dtype=torch.long, device=device),
                torch.ones((real_len,), dtype=torch.long, device=device),
                torch.zeros((pad_len,), dtype=torch.long, device=device),
            )
        )
        combined_regions = torch.cat(
            (
                prefix_tail_region_ids,
                torch.full((real_len,), generation_region_id, dtype=torch.long, device=device),
                torch.full((pad_len,), REGION_PAD, dtype=torch.long, device=device),
            )
        )
        combined_offsets = torch.cat(
            (
                prefix_tail_offset_ids,
                torch.arange(
                    generation_offset_start,
                    generation_offset_start + real_len,
                    dtype=torch.long,
                    device=device,
                ),
                torch.zeros((pad_len,), dtype=torch.long, device=device),
            )
        )

        appended_ids = combined_tokens[:-1]
        input_ids = torch.cat((seed_ids, appended_ids))
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[seed_len - 1 :] = combined_tokens
        region_ids = torch.cat((seed_region_ids, combined_regions[:-1]))
        offset_ids = torch.cat((seed_offset_ids, combined_offsets[:-1]))

        patched = megabyte_teacher_forced_sample(
            input_ids, labels, region_ids, offset_ids, patch_size
        )
        # patch_byte_sample enforces input_ids == shifted labels (real byte
        # content) as an internal consistency guard -- it cannot accept a
        # second call with labels swapped for 0/1 validity flags against the
        # SAME (real-content) input_ids, so re-running it for the validity
        # mask (as an earlier version of this function did) always raises
        # "teacher-forcing shift is inconsistent". Instead replicate its
        # patch geometry directly via _pad_patch_axis (the same pure,
        # value-independent padding primitive patch_byte_sample itself uses
        # for both the prompt and target sides) -- this only needs to know
        # WHERE positions fall, not validate matching byte values.
        prompt_patch_count = -(-seed_len // patch_size)
        target_validity = _pad_patch_axis(combined_validity, patch_size, 0, left=False)
        candidate_validity = torch.cat(
            (
                torch.zeros((prompt_patch_count - 1, patch_size), dtype=torch.long, device=device),
                target_validity,
            )
        )
        # megabyte_teacher_forced_sample returns its fields pre-unsqueezed
        # with a batch dim (see its `.unsqueeze(0)` for each key), so that
        # torch.cat(..., dim=0) below stacks candidates into (B, T, P).
        # candidate_validity is built by hand and needs the same unsqueeze,
        # or concatenation flattens candidates along the patch dimension
        # instead of stacking a batch dimension.
        patched_validity.append(candidate_validity.unsqueeze(0))
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
    supervised = torch.cat(patched_validity, dim=0) == 1
    return inputs, inputs["patch_targets"], supervised


def group_token_log_probabilities(
    model: nn.Module,
    inputs: dict[str, Tensor],
    labels: Tensor,
    *,
    include_eos: bool = True,
) -> Tensor:
    """Return per-token log-probs, shape ``(B, T, P)``.

    Deliberately call the supplied wrapper rather than ``_unwrap_model``: during
    multi-GPU GRPO this is the gradient-bearing DDP forward whose backward pass
    must all-reduce policy gradients.  Rollout generation is inference-only and
    may unwrap the replicated model; candidate scoring may not.

    ``include_eos`` must match the rollout action space. Learned-EOS FIM uses
    257 actions (256 bytes + EOS), while fixed-frame AR and oracle-length FIM
    sample only 256 bytes. Scoring a different action space from the behavior
    policy would give GRPO an incorrect likelihood ratio and KL penalty.
    """
    logits = model(**inputs)
    if include_eos:
        allowed_logits = torch.cat(
            (
                logits[..., :BYTE_VOCAB_SIZE],
                logits[..., SEQ_EOS_ID : SEQ_EOS_ID + 1],
            ),
            dim=-1,
        )
        target = torch.where(
            labels == SEQ_EOS_ID, BYTE_VOCAB_SIZE, labels.clamp_min(0)
        )
    else:
        if bool((labels == SEQ_EOS_ID).any()):
            raise ValueError("byte-only GRPO scoring received an EOS target")
        allowed_logits = logits[..., :BYTE_VOCAB_SIZE]
        target = labels.clamp_min(0)
    log_probs = torch.log_softmax(allowed_logits.float(), dim=-1)
    return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def mean_log_probability(gathered: Tensor, supervised: Tensor) -> Tensor:
    """Length-normalized per-candidate log-prob, shape ``(B,)``."""
    masked = gathered.masked_fill(~supervised, 0.0)
    counts = supervised.sum(dim=(1, 2)).clamp_min(1)
    return masked.sum(dim=(1, 2)) / counts


def grpo_update_direction_metrics(
    before_log_probs: Tensor,
    after_log_probs: Tensor,
    rewards: Tensor,
    advantages: Tensor,
    *,
    tie_tolerance: float = GRPO_EPS,
) -> dict[str, float]:
    """Measure whether one GRPO update moves probability toward reward.

    Inputs are one length-normalized sequence log-probability per candidate.
    The primary quantity is ``mean(advantage * log_probability)``: at ``mu=1``
    and before adding a KL term, its negative has the same local gradient as
    :func:`grpo_clipped_loss`. A correct sufficiently-small optimizer step must
    therefore increase this score.

    Pairwise metrics are diagnostic rather than the primary invariant. Shared
    model parameters can make some candidate pairs interfere even when the
    aggregate policy-gradient direction is correct.
    """
    tensors = (before_log_probs, after_log_probs, rewards, advantages)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("GRPO direction metrics expect one-dimensional tensors")
    if len({int(tensor.numel()) for tensor in tensors}) != 1:
        raise ValueError("GRPO direction metric tensors must have equal lengths")
    if before_log_probs.numel() < 2:
        raise ValueError("GRPO direction metrics require at least two candidates")

    before = before_log_probs.detach().float()
    after = after_log_probs.detach().float()
    reward = rewards.detach().float()
    advantage = advantages.detach().float()
    delta = after - before

    score_before = (advantage * before).mean()
    score_after = (advantage * after).mean()

    centered_advantage = advantage - advantage.mean()
    centered_delta = delta - delta.mean()
    correlation_denominator = torch.linalg.vector_norm(
        centered_advantage
    ) * torch.linalg.vector_norm(centered_delta)
    advantage_delta_correlation = (
        float((centered_advantage * centered_delta).sum() / correlation_denominator)
        if float(correlation_denominator) > tie_tolerance
        else float("nan")
    )

    improved_pairs = 0
    tied_pairs = 0
    comparable_pairs = 0
    for high in range(reward.numel()):
        for low in range(reward.numel()):
            if float(reward[high] - reward[low]) <= tie_tolerance:
                continue
            comparable_pairs += 1
            margin_delta = float(
                (after[high] - after[low]) - (before[high] - before[low])
            )
            if margin_delta > tie_tolerance:
                improved_pairs += 1
            elif abs(margin_delta) <= tie_tolerance:
                tied_pairs += 1

    return {
        "policy_score_before": float(score_before),
        "policy_score_after": float(score_after),
        "policy_score_delta": float(score_after - score_before),
        "advantage_delta_correlation": advantage_delta_correlation,
        "pairwise_comparable": float(comparable_pairs),
        "pairwise_improved": float(improved_pairs),
        "pairwise_tied": float(tied_pairs),
        "pairwise_improved_fraction": (
            improved_pairs / comparable_pairs if comparable_pairs else float("nan")
        ),
    }


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
