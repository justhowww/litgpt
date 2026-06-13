"""Online minimum-risk training for decoder-scored H.264 FIM candidates.

The module keeps candidate generation, external decoder scoring, and the MRT
surrogate separate from the generic LitGPT loop. The training loop remains
responsible only for scheduling the expensive step and applying its gradients.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from litgpt.byte.data import BYTE_VOCAB_SIZE, IGNORE_INDEX, SEQ_EOS_ID
from litgpt.byte.reconstruction import (
    ReconstructionSample,
    _ground_truth_replacement,
    _unwrap_model,
    decode_frame,
    replace_target_nal,
)

MRT_RISK_MODES = ("clipped_mse", "smooth_mse")


@dataclass(frozen=True)
class MRTConfig:
    """Configuration for sparse online minimum-risk updates."""

    interval: int = 0
    start_step: int = 0
    num_candidates: int = 16
    context_pool_size: int = 64
    max_target_bytes: int = 2048
    oracle_length: bool = False
    learned_eos: bool = False
    temperature: float = 1.0
    candidate_alpha: float = 1.0
    weight: float = 4.0
    risk_mode: str = "clipped_mse"
    mse_weight: float = 1000.0
    mse_tau: float = 0.002
    decode_failure_weight: float = 2.0
    max_risk: float = 2.0
    timeout_sec: int = 30
    decode_workers: int = 8
    ffmpeg_binary: str = "ffmpeg"

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def validate(self) -> None:
        if self.interval < 0:
            raise ValueError("MRT interval must be non-negative")
        if self.start_step < 0:
            raise ValueError("MRT start step must be non-negative")
        if self.num_candidates < 2:
            raise ValueError("MRT requires at least two candidates")
        if self.context_pool_size <= 0:
            raise ValueError("MRT context pool size must be positive")
        if self.max_target_bytes <= 0:
            raise ValueError("MRT max target bytes must be positive")
        if self.oracle_length and self.learned_eos:
            raise ValueError(
                "MRT oracle length and learned EOS are mutually exclusive"
            )
        if self.temperature <= 0:
            raise ValueError("MRT temperature must be positive")
        if self.candidate_alpha <= 0:
            raise ValueError("MRT candidate alpha must be positive")
        if self.risk_mode not in MRT_RISK_MODES:
            raise ValueError(f"MRT risk mode must be one of {MRT_RISK_MODES}")
        if self.weight < 0 or self.mse_weight < 0:
            raise ValueError("MRT and MSE weights must be non-negative")
        if self.mse_tau <= 0:
            raise ValueError("MRT MSE tau must be positive")
        if self.decode_failure_weight < 0 or self.max_risk <= 0:
            raise ValueError("MRT failure weight and max risk must be positive")
        if self.timeout_sec <= 0 or self.decode_workers <= 0:
            raise ValueError("MRT decoder timeout and worker count must be positive")


@dataclass(frozen=True)
class Candidate:
    """One sampled replacement and its current-policy sequence score."""

    data: bytes
    stopped: bool
    mean_log_probability: float | None = None
    is_ground_truth: bool = False

    def target_tokens(self, device: torch.device | None = None) -> Tensor:
        tokens = list(self.data)
        if self.stopped:
            tokens.append(SEQ_EOS_ID)
        return torch.tensor(tokens, dtype=torch.long, device=device)


@dataclass(frozen=True)
class PreparedMRTStep:
    """Decoder-scored candidates ready for differentiable rescoring."""

    sample: ReconstructionSample
    candidates: tuple[Candidate, ...]
    risks: Tensor
    candidate_mses: Tensor
    coefficients: Tensor
    expected_risk: float
    decode_rate: float
    ground_truth_probability: float


def visual_risk(mse: float, config: MRTConfig) -> float:
    """Map decoded-frame MSE to the configured bounded MRT risk."""
    if mse < 0:
        raise ValueError("MSE must be non-negative")
    if config.risk_mode == "smooth_mse":
        return mse / (mse + config.mse_tau)
    return min(config.max_risk, config.mse_weight * mse)


def should_run_mrt(next_step: int, config: MRTConfig) -> bool:
    """Return whether the upcoming optimizer step includes an MRT update."""
    return config.enabled and next_step > config.start_step and (next_step - config.start_step) % config.interval == 0


def minimum_risk_coefficients(
    mean_log_probabilities: Tensor,
    risks: Tensor,
    alpha: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute q, expected risk, and the exact score-function coefficients.

    The returned coefficients are detached. Multiplying each coefficient by
    its differentiable sequence score gives the gradient of E_q[risk] without
    retaining every candidate's forward graph at once.
    """
    if mean_log_probabilities.ndim != 1 or risks.ndim != 1:
        raise ValueError("MRT scores and risks must be one-dimensional")
    if mean_log_probabilities.shape != risks.shape:
        raise ValueError("MRT scores and risks must have matching shapes")
    if mean_log_probabilities.numel() < 2:
        raise ValueError("MRT requires at least two unique candidates")
    q = torch.softmax(alpha * mean_log_probabilities.float(), dim=0)
    expected_risk = (q * risks.float()).sum()
    coefficients = alpha * q * (risks.float() - expected_risk)
    return q.detach(), expected_risk.detach(), coefficients.detach()


def build_candidate_inputs(
    sample: ReconstructionSample,
    candidate: Candidate,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    """Build the aligned teacher-forcing sequence for one generated span."""
    target_tokens = candidate.target_tokens(device)
    if target_tokens.numel() == 0:
        raise ValueError("A non-terminating empty candidate has no probability")

    prompt_ids = sample.prompt_ids.to(device)
    appended_ids = target_tokens[:-1]
    input_ids = torch.cat((prompt_ids, appended_ids)).unsqueeze(0)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[0, prompt_ids.numel() - 1 :] = target_tokens

    region_ids = torch.cat(
        (
            sample.prompt_region_ids.to(device),
            torch.full(
                (appended_ids.numel(),),
                sample.generation_region_id,
                dtype=torch.long,
                device=device,
            ),
        )
    ).unsqueeze(0)
    offset_ids = torch.cat(
        (
            sample.prompt_offset_ids.to(device),
            torch.arange(
                sample.generation_offset_start,
                sample.generation_offset_start + appended_ids.numel(),
                dtype=torch.long,
                device=device,
            ),
        )
    ).unsqueeze(0)
    return {
        "idx": input_ids,
        "region_ids": region_ids,
        "offset_ids": offset_ids,
    }, labels


def candidate_mean_log_probability(
    model: nn.Module,
    sample: ReconstructionSample,
    candidate: Candidate,
    device: torch.device,
    temperature: float = 1.0,
) -> Tensor:
    """Return length-normalized log probability under the active policy."""
    if temperature <= 0:
        raise ValueError("Candidate temperature must be positive")
    inputs, labels = build_candidate_inputs(sample, candidate, device)
    # Candidate lengths vary. Bypass torch.compile here so an MRT step does not
    # compile a new graph for every sampled length; gradients still reach the
    # same underlying parameters.
    logits = _unwrap_model(model)(**inputs)[0]
    supervised = labels[0] != IGNORE_INDEX
    target = labels[0, supervised]
    selected_logits = logits[supervised].float()

    if candidate.stopped:
        allowed_logits = torch.cat(
            (
                selected_logits[:, :BYTE_VOCAB_SIZE],
                selected_logits[:, SEQ_EOS_ID : SEQ_EOS_ID + 1],
            ),
            dim=-1,
        )
        allowed_target = torch.where(target == SEQ_EOS_ID, BYTE_VOCAB_SIZE, target)
    else:
        # Oracle-length MRT has no EOS action: normalize over raw bytes only.
        allowed_logits = selected_logits[:, :BYTE_VOCAB_SIZE]
        allowed_target = target
    allowed_logits = allowed_logits / temperature
    log_probabilities = torch.log_softmax(allowed_logits, dim=-1)
    return log_probabilities.gather(1, allowed_target.unsqueeze(1)).mean()


@torch.inference_mode()
def sample_candidates(
    model: nn.Module,
    sample: ReconstructionSample,
    device: torch.device,
    num_candidates: int,
    temperature: float,
    oracle_length: bool = False,
    learned_eos: bool = False,
) -> list[Candidate]:
    """Sample byte spans in one batched KV-cache decode."""
    if not oracle_length and not learned_eos and sample.stop_token != SEQ_EOS_ID:
        raise ValueError("MRT candidate generation requires SEQ_EOS supervision")

    raw_model = _unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()
    prompt = sample.prompt_ids.to(device).unsqueeze(0).expand(num_candidates, -1)
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0).expand(num_candidates, -1)
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0).expand(num_candidates, -1)
    prompt_length = prompt.size(1)
    generation_limit = (
        sample.target_length if oracle_length else 2 * sample.target_length
    )
    max_new = min(
        generation_limit,
        raw_model.max_seq_length - prompt_length + 1,
    )
    if max_new <= 0:
        return []

    generated = [[] for _ in range(num_candidates)]
    log_probability_sums = torch.zeros(num_candidates, device=device)
    token_counts = torch.zeros(num_candidates, dtype=torch.long, device=device)
    active = torch.ones(num_candidates, dtype=torch.bool, device=device)
    stopped = torch.zeros(num_candidates, dtype=torch.bool, device=device)
    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw_model.parameters()).dtype
    raw_model.set_kv_cache(
        batch_size=num_candidates,
        max_seq_length=raw_model.max_seq_length,
        device=device,
        dtype=cache_dtype,
    )
    try:
        input_pos = torch.arange(prompt_length, device=device, dtype=torch.long)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = raw_model(
                prompt,
                input_pos=input_pos,
                input_pos_maxp1=prompt_length,
                region_ids=region_ids,
                offset_ids=offset_ids,
            )

        for generated_idx in range(max_new):
            next_logits = logits[:, -1].float()
            if oracle_length:
                allowed_logits = next_logits[:, :BYTE_VOCAB_SIZE] / temperature
            else:
                allowed_logits = (
                    torch.cat(
                        (
                            next_logits[:, :BYTE_VOCAB_SIZE],
                            next_logits[:, SEQ_EOS_ID : SEQ_EOS_ID + 1],
                        ),
                        dim=-1,
                    )
                    / temperature
                )
            log_probs = torch.log_softmax(allowed_logits, dim=-1)
            sampled = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(1)
            sampled_log_probs = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            if not oracle_length:
                sampled = torch.where(
                    sampled == BYTE_VOCAB_SIZE,
                    torch.full_like(sampled, SEQ_EOS_ID),
                    sampled,
                )

            active_indices = active.nonzero(as_tuple=False).flatten()
            log_probability_sums[active] += sampled_log_probs[active]
            token_counts[active] += 1
            for row in active_indices.tolist():
                token = int(sampled[row])
                if not oracle_length and token == SEQ_EOS_ID:
                    stopped[row] = True
                    active[row] = False
                else:
                    generated[row].append(token)
            if not active.any() or generated_idx == max_new - 1:
                break

            # Finished rows receive a harmless byte while active rows advance;
            # their later logits are ignored.
            fed_tokens = torch.where(active, sampled, torch.zeros_like(sampled))
            position = prompt_length + generated_idx
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = raw_model(
                    fed_tokens.unsqueeze(1),
                    input_pos=torch.tensor([position], device=device),
                    input_pos_maxp1=position + 1,
                    region_ids=torch.full(
                        (num_candidates, 1),
                        sample.generation_region_id,
                        dtype=torch.long,
                        device=device,
                    ),
                    offset_ids=torch.full(
                        (num_candidates, 1),
                        sample.generation_offset_start + generated_idx,
                        dtype=torch.long,
                        device=device,
                    ),
                )
    finally:
        raw_model.clear_kv_cache()
        raw_model.train(was_training)

    return [
        Candidate(
            data=bytes(tokens),
            stopped=bool(stopped[row]) if not oracle_length else False,
            mean_log_probability=float(log_probability_sums[row] / token_counts[row].clamp_min(1)),
        )
        for row, tokens in enumerate(generated)
    ]


def _candidate_risk(
    stream: bytes,
    sample: ReconstructionSample,
    candidate: Candidate,
    reference: Tensor,
    config: MRTConfig,
) -> tuple[float, bool, float | None]:
    reconstructed_stream = replace_target_nal(stream, sample, candidate.data)
    reconstruction, status = decode_frame(
        reconstructed_stream,
        sample.frame_index,
        config.ffmpeg_binary,
        config.timeout_sec,
        strict_syntax=True,
    )
    if reconstruction is None or reconstruction.shape != reference.shape:
        return config.decode_failure_weight, False, None
    mse = torch.nn.functional.mse_loss(reconstruction, reference).item()
    return visual_risk(mse, config), status == "decoded", mse


def prepare_mrt_step(
    model: nn.Module,
    sample: ReconstructionSample,
    config: MRTConfig,
    device: torch.device,
) -> PreparedMRTStep | None:
    """Generate, strictly decode, and score one online MRT candidate set."""
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

    ground_truth = Candidate(
        data=_ground_truth_replacement(stream, sample),
        stopped=not config.oracle_length,
        is_ground_truth=True,
    )
    sampled = sample_candidates(
        model,
        sample,
        device,
        num_candidates=config.num_candidates - 1,
        temperature=config.temperature,
        oracle_length=config.oracle_length,
        learned_eos=config.learned_eos,
    )
    candidates: list[Candidate] = []
    seen: set[tuple[bytes, bool]] = set()
    for candidate in (ground_truth, *sampled):
        identity = (candidate.data, candidate.stopped)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(candidate)
    if len(candidates) < 2:
        return None

    with ThreadPoolExecutor(max_workers=config.decode_workers) as executor:
        risk_results = list(
            executor.map(
                lambda candidate: _candidate_risk(stream, sample, candidate, reference, config),
                candidates,
            )
        )
    risks = torch.tensor([risk for risk, _, _ in risk_results], device=device)
    candidate_mses = torch.tensor(
        [float("nan") if mse is None else mse for _, _, mse in risk_results],
        device=device,
    )
    decode_rate = sum(decoded for _, decoded, _ in risk_results) / len(risk_results)

    scores: list[float] = []
    for candidate in candidates:
        if candidate.mean_log_probability is not None:
            scores.append(candidate.mean_log_probability)
            continue
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            score = candidate_mean_log_probability(
                model,
                sample,
                candidate,
                device,
                temperature=config.temperature,
            )
        scores.append(float(score))

    q, expected_risk, coefficients = minimum_risk_coefficients(
        torch.tensor(scores, device=device),
        risks,
        config.candidate_alpha,
    )
    return PreparedMRTStep(
        sample=sample,
        candidates=tuple(candidates),
        risks=risks.detach(),
        candidate_mses=candidate_mses.detach(),
        coefficients=coefficients,
        expected_risk=float(expected_risk),
        decode_rate=decode_rate,
        ground_truth_probability=float(q[0]),
    )
