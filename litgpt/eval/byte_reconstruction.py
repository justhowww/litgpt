"""Sparse decoder-level validation for H.264 byte reconstruction."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Subset

from litgpt.data.byte_data import (
    BYTE_VOCAB_SIZE,
    IGNORE_INDEX,
    REGION_BRIDGE,
    REGION_TARGET,
    SEQ_EOS_ID,
    VCL_NAL_TYPES,
    ByteSliceDataset,
)


@dataclass(frozen=True)
class ReconstructionEvalConfig:
    """Controls the expensive AR or FIM decoder-level probe."""

    interval: int = 1000
    num_samples: int = 5
    timeout_sec: int = 30
    ffmpeg_binary: str = "ffmpeg"
    max_target_bytes: int = 2048
    task: str = "ar"  # "ar", "fim", or "both"
    evaluate_oracle_length: bool = False
    evaluate_error_exploding: bool = False
    evaluate_fim_baselines: bool = False


@dataclass(frozen=True)
class ReconstructionSample:
    h264_path: Path
    target_start: int
    target_end: int
    target_nal_index: int
    frame_index: int
    prompt_ids: Tensor
    prompt_region_ids: Tensor
    prompt_offset_ids: Tensor
    target_length: int
    task: str = "ar"
    replacement_start: int = 0
    replacement_end: int | None = None
    generation_region_id: int = REGION_TARGET
    generation_offset_start: int = 0
    # When set, greedy decoding stops as soon as this token is produced (the
    # learned SEQ_EOS marker). For EOS samples generation may run up to ~2x the
    # true length, so the stop metrics see both early and late termination.
    stop_token: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    data: bytes
    stopped: bool


def select_reconstruction_samples(
    dataset: object, num_samples: int, max_target_bytes: int, task: str = "ar"
) -> list[ReconstructionSample]:
    """Select a deterministic prefix of held-out AR or FIM samples."""
    if task not in {"ar", "fim"}:
        raise ValueError("Reconstruction task must be 'ar' or 'fim'")
    if num_samples <= 0:
        return []
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)
    else:
        base_dataset = dataset
        indices = list(range(len(dataset))) if hasattr(dataset, "__len__") else []
    if not isinstance(base_dataset, ByteSliceDataset):
        return []

    use_eos = bool(getattr(base_dataset, "use_eos", False))
    selected: list[ReconstructionSample] = []
    for idx in indices:
        item = base_dataset[idx]
        if item["sample_meta"]["task"] != task:
            continue
        supervised = item["labels"] != IGNORE_INDEX
        # When use_eos is on the final supervised label is SEQ_EOS, which is not
        # a real byte; exclude it from the reconstructed span length.
        target_length = int(supervised.sum()) - (1 if use_eos else 0)
        if target_length <= 0 or target_length > max_target_bytes:
            continue
        first_target_input = int(supervised.nonzero()[0])
        prompt_end = first_target_input + 1
        replacement_start = (
            int(item["offset_ids"][first_target_input]) if task == "fim" else 0
        )
        replacement_end = (
            replacement_start + target_length if task == "fim" else None
        )

        sample = base_dataset.samples[idx]
        nals = base_dataset.nal_index[str(sample.h264_path)]
        target_nal = nals[sample.target_index]
        frame_index = sum(
            nal.nal_type in VCL_NAL_TYPES for nal in nals[: sample.target_index]
        )
        selected.append(
            ReconstructionSample(
                h264_path=sample.h264_path,
                target_start=target_nal.start,
                target_end=target_nal.end,
                target_nal_index=sample.target_index,
                frame_index=frame_index,
                prompt_ids=item["input_ids"][:prompt_end].clone(),
                prompt_region_ids=item["region_ids"][:prompt_end].clone(),
                prompt_offset_ids=item["offset_ids"][:prompt_end].clone(),
                target_length=target_length,
                task=task,
                replacement_start=replacement_start,
                replacement_end=replacement_end,
                generation_region_id=(
                    REGION_BRIDGE if task == "fim" else REGION_TARGET
                ),
                generation_offset_start=(
                    replacement_start + 1 if task == "fim" else prompt_end
                ),
                stop_token=SEQ_EOS_ID if use_eos else None,
            )
        )
        if len(selected) >= num_samples:
            break
    return selected


def save_reconstruction_sample_manifest(
    samples: list[ReconstructionSample], path: Path
) -> None:
    rows = [
        {
            "h264_path": str(sample.h264_path),
            "target_nal_index": sample.target_nal_index,
            "frame_index": sample.frame_index,
            "target_length": sample.target_length,
            "task": sample.task,
            "replacement_start": sample.replacement_start,
            "replacement_end": sample.replacement_end,
        }
        for sample in samples
    ]
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attr in ("_forward_module", "_orig_mod", "module"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module) and child is not current:
                current = child
                break
        else:
            return current
    return current


@torch.inference_mode()
def generate_target_slice(
    model: nn.Module,
    sample: ReconstructionSample,
    device: torch.device,
    *,
    oracle_length: bool = False,
) -> GenerationResult | None:
    """Generate bytes using either learned EOS stopping or an oracle byte count."""
    raw_model = _unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0)
    prompt_length = prompt.size(1)
    # Generation must at least fit the oracle-length span.
    if prompt_length + sample.target_length - 1 > raw_model.max_seq_length:
        return None
    # With EOS, allow generation up to ~2x the true length so the stop metrics
    # capture late and non-terminating cases, not just early stops; the model is
    # expected to fire EOS well before the cap. Without EOS the oracle length is
    # an exact count, so non-EOS baselines are unchanged.
    if sample.stop_token is not None and not oracle_length:
        max_new = min(
            2 * sample.target_length,
            raw_model.max_seq_length - prompt_length + 1,
        )
    else:
        max_new = sample.target_length

    generated: list[int] = []
    stopped = False
    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw_model.parameters()).dtype
    raw_model.set_kv_cache(
        batch_size=1,
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
            next_logits = logits[0, -1]
            if oracle_length:
                # Exclude control tokens so this probe isolates byte-content
                # quality from the model's learned stopping behavior.
                token = int(next_logits[:BYTE_VOCAB_SIZE].argmax())
            else:
                token = int(next_logits[: raw_model.config.vocab_size].argmax())
            if sample.stop_token is not None and token == sample.stop_token:
                # Model chose to terminate the span on its own.
                stopped = True
                break
            if token >= BYTE_VOCAB_SIZE:
                return None
            generated.append(token)
            if generated_idx == max_new - 1:
                break

            position = prompt_length + generated_idx
            token_tensor = torch.tensor([[token]], device=device, dtype=torch.long)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = raw_model(
                    token_tensor,
                    input_pos=torch.tensor([position], device=device),
                    input_pos_maxp1=position + 1,
                    region_ids=torch.tensor(
                        [[sample.generation_region_id]], device=device
                    ),
                    offset_ids=torch.tensor(
                        [[sample.generation_offset_start + generated_idx]],
                        device=device,
                    ),
                )
    finally:
        raw_model.clear_kv_cache()
        raw_model.train(was_training)
    return GenerationResult(bytes(generated), stopped)


def replace_target_nal(stream: bytes, sample: ReconstructionSample, generated: bytes) -> bytes:
    if sample.task == "fim":
        replacement_end = sample.replacement_end
        if replacement_end is None:
            raise ValueError("FIM sample is missing replacement_end")
        start = sample.target_start + sample.replacement_start
        end = sample.target_start + replacement_end
        return stream[:start] + generated + stream[end:]
    return stream[: sample.target_start] + generated + stream[sample.target_end :]


def decode_frame(
    stream: bytes,
    frame_index: int,
    ffmpeg_binary: str,
    timeout_sec: int,
    *,
    error_exploding: bool = False,
    strict_syntax: bool = False,
) -> tuple[Tensor | None, str]:
    """Decode one frame, optionally disabling concealment and rejecting errors."""
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if strict_syntax:
        command.extend(
            ["-ec", "0", "-err_detect", "explode+bitstream+buffer+compliant"]
        )
    elif error_exploding:
        command.extend(["-err_detect", "explode"])
    command.extend(
        [
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "pipe:1",
        ]
    )
    try:
        result = subprocess.run(
            command,
            input=stream,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if (error_exploding or strict_syntax) and result.returncode != 0:
        return None, "decoder_error"
    if not result.stdout:
        return None, "no_frame"
    try:
        return parse_ppm(result.stdout), "decoded"
    except ValueError:
        return None, "invalid_frame"


def parse_ppm(data: bytes) -> Tensor:
    """Parse a binary P6 PPM frame into float RGB in [0, 1]."""
    tokens: list[bytes] = []
    cursor = 0
    while len(tokens) < 4:
        while cursor < len(data) and data[cursor : cursor + 1].isspace():
            cursor += 1
        if cursor >= len(data):
            raise ValueError("incomplete PPM header")
        if data[cursor : cursor + 1] == b"#":
            cursor = data.find(b"\n", cursor)
            if cursor < 0:
                raise ValueError("unterminated PPM comment")
            continue
        end = cursor
        while end < len(data) and not data[end : end + 1].isspace():
            end += 1
        tokens.append(data[cursor:end])
        cursor = end

    if tokens[0] != b"P6":
        raise ValueError("expected P6 PPM")
    width, height, max_value = map(int, tokens[1:])
    if max_value != 255:
        raise ValueError("only 8-bit PPM is supported")
    if cursor >= len(data) or not data[cursor : cursor + 1].isspace():
        raise ValueError("PPM header is not terminated")
    cursor += 1
    pixels = data[cursor:]
    if len(pixels) != width * height * 3:
        raise ValueError("PPM pixel payload has the wrong size")
    return torch.frombuffer(bytearray(pixels), dtype=torch.uint8).reshape(height, width, 3).float() / 255.0


def image_psnr(reference: Tensor, reconstruction: Tensor) -> float:
    mse = F.mse_loss(reconstruction, reference).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def image_ssim(reference: Tensor, reconstruction: Tensor) -> float:
    """Compute standard windowed SSIM and average over RGB channels."""
    reference = reference.permute(2, 0, 1).unsqueeze(0)
    reconstruction = reconstruction.permute(2, 0, 1).unsqueeze(0)
    kernel_1d = torch.tensor(
        [
            0.001028,
            0.007599,
            0.036001,
            0.109361,
            0.213006,
            0.266012,
            0.213006,
            0.109361,
            0.036001,
            0.007599,
            0.001028,
        ],
        dtype=reference.dtype,
    )
    kernel = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel.expand(3, 1, 11, 11)
    mu_x = F.conv2d(reference, kernel, padding=5, groups=3)
    mu_y = F.conv2d(reconstruction, kernel, padding=5, groups=3)
    sigma_x = F.conv2d(reference * reference, kernel, padding=5, groups=3) - mu_x.square()
    sigma_y = F.conv2d(reconstruction * reconstruction, kernel, padding=5, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(reference * reconstruction, kernel, padding=5, groups=3) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim.mean().item()


def run_reconstruction_probe(
    model: nn.Module,
    samples: list[ReconstructionSample],
    config: ReconstructionEvalConfig,
    device: torch.device,
) -> dict[str, float]:
    method_stats: dict[tuple[str, str], dict[str, object]] = {}
    stop_records: list[tuple[int, int, bool]] = []
    unexpected_failures = 0
    invalid_generation = 0

    def stats_for(method: str, mode: str) -> dict[str, object]:
        return method_stats.setdefault(
            (method, mode),
            {
                "attempted": 0,
                "decoded": 0,
                "timeouts": 0,
                "missing_frames": 0,
                "psnr": [],
                "ssim": [],
            },
        )

    for sample in samples:
        stage = "read_stream"
        try:
            modes = ["normal"]
            if config.evaluate_error_exploding:
                modes.append("error_exploding")
            for mode in modes:
                stats_for("model_learned", mode)["attempted"] += 1

            stream = sample.h264_path.read_bytes()
            stage = "decode_reference"
            reference, reference_status = decode_frame(
                stream, sample.frame_index, config.ffmpeg_binary, config.timeout_sec
            )
            if reference is None:
                for mode in modes:
                    target_stats = stats_for("model_learned", mode)
                    target_stats["timeouts"] += int(reference_status == "timeout")
                    target_stats["missing_frames"] += int(
                        reference_status != "timeout"
                    )
                continue

            stage = "generate_learned"
            learned = generate_target_slice(model, sample, device)
            if learned is None:
                invalid_generation += 1
            elif sample.stop_token is not None:
                stop_records.append(
                    (len(learned.data), sample.target_length, learned.stopped)
                )

            candidates: dict[str, bytes] = {}
            if learned is not None:
                candidates["model_learned"] = learned.data
            if config.evaluate_oracle_length:
                stage = "generate_oracle"
                oracle = generate_target_slice(
                    model, sample, device, oracle_length=True
                )
                if oracle is not None:
                    candidates["model_oracle"] = oracle.data

            if sample.task == "fim" and config.evaluate_fim_baselines:
                ground_truth = _ground_truth_replacement(stream, sample)
                candidates["ground_truth"] = ground_truth
                candidates["deleted_gap"] = b""
                candidates["random_bytes"] = _deterministic_random_bytes(
                    sample, sample.target_length
                )

            for method, replacement in candidates.items():
                stage = f"replace_{method}"
                reconstructed_stream = replace_target_nal(
                    stream, sample, replacement
                )
                for mode in modes:
                    candidate_stats = stats_for(method, mode)
                    if method != "model_learned":
                        candidate_stats["attempted"] += 1
                    stage = f"decode_{method}_{mode}"
                    reconstruction, status = decode_frame(
                        reconstructed_stream,
                        sample.frame_index,
                        config.ffmpeg_binary,
                        config.timeout_sec,
                        error_exploding=mode == "error_exploding",
                    )
                    if reconstruction is None:
                        candidate_stats["timeouts"] += int(status == "timeout")
                        candidate_stats["missing_frames"] += int(status != "timeout")
                        continue
                    candidate_stats["decoded"] += 1
                    candidate_stats["psnr"].append(
                        image_psnr(reference, reconstruction)
                    )
                    candidate_stats["ssim"].append(
                        image_ssim(reference, reconstruction)
                    )
        except Exception as exc:
            unexpected_failures += 1
            if unexpected_failures <= 3:
                print(
                    "Reconstruction probe error "
                    f"[{stage}] {sample.h264_path} frame={sample.frame_index}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    learned_normal = stats_for("model_learned", "normal")
    attempted = int(learned_normal["attempted"])
    decoded = int(learned_normal["decoded"])
    metrics = {
        "reconstruction/attempted": float(attempted),
        "reconstruction/decoded": float(decoded),
        "reconstruction/decode_rate": decoded / attempted if attempted else 0.0,
        "reconstruction/invalid_generation": float(invalid_generation),
        "reconstruction/timeouts": float(learned_normal["timeouts"]),
        "reconstruction/missing_target_frames": float(
            learned_normal["missing_frames"]
        ),
        "reconstruction/unexpected_failures": float(unexpected_failures),
    }
    metrics.update(_quality_metrics("reconstruction/", learned_normal))

    if stop_records:
        n = len(stop_records)
        abs_errs = [
            abs(generated_length - oracle_length)
            for generated_length, oracle_length, _ in stop_records
        ]
        metrics["reconstruction/gen_len_abs_err_mean"] = sum(abs_errs) / n
        metrics["reconstruction/stop_exact_rate"] = (
            sum(
                stopped and generated == oracle
                for generated, oracle, stopped in stop_records
            )
            / n
        )
        metrics["reconstruction/stop_early_rate"] = (
            sum(stopped and generated < oracle for generated, oracle, stopped in stop_records)
            / n
        )
        metrics["reconstruction/stop_late_rate"] = (
            sum(
                (stopped and generated > oracle) or not stopped
                for generated, oracle, stopped in stop_records
            )
            / n
        )
        metrics["reconstruction/stop_no_stop_rate"] = (
            sum(not stopped for _, _, stopped in stop_records) / n
        )

    for (method, mode), method_values in method_stats.items():
        prefix = f"reconstruction/{method}/{mode}/"
        attempted = int(method_values["attempted"])
        decoded = int(method_values["decoded"])
        metrics[f"{prefix}attempted"] = float(attempted)
        metrics[f"{prefix}decoded"] = float(decoded)
        metrics[f"{prefix}decode_rate"] = (
            decoded / attempted if attempted else 0.0
        )
        metrics[f"{prefix}timeouts"] = float(method_values["timeouts"])
        metrics[f"{prefix}missing_target_frames"] = float(
            method_values["missing_frames"]
        )
        metrics.update(_quality_metrics(prefix, method_values))
    for mode in ("normal", "error_exploding"):
        oracle_prefix = f"reconstruction/model_oracle/{mode}/"
        concealment_prefix = f"reconstruction/deleted_gap/{mode}/"
        for metric_name in ("psnr_mean_valid", "ssim_mean_valid"):
            oracle_key = f"{oracle_prefix}{metric_name}"
            concealment_key = f"{concealment_prefix}{metric_name}"
            if oracle_key in metrics and concealment_key in metrics:
                metrics[
                    f"reconstruction/model_oracle_vs_deleted_gap/{mode}/{metric_name}_delta"
                ] = metrics[oracle_key] - metrics[concealment_key]
    return metrics


def _quality_metrics(prefix: str, stats: dict[str, object]) -> dict[str, float]:
    psnr_values = stats["psnr"]
    ssim_values = stats["ssim"]
    if not psnr_values:
        return {}
    finite_psnr = [value for value in psnr_values if math.isfinite(value)]
    return {
        f"{prefix}psnr_mean_valid": (
            sum(finite_psnr) / len(finite_psnr) if finite_psnr else float("inf")
        ),
        f"{prefix}psnr_median_valid": float(torch.tensor(psnr_values).median()),
        f"{prefix}ssim_mean_valid": sum(ssim_values) / len(ssim_values),
        f"{prefix}ssim_median_valid": float(torch.tensor(ssim_values).median()),
    }


def _ground_truth_replacement(
    stream: bytes, sample: ReconstructionSample
) -> bytes:
    if sample.task == "fim":
        if sample.replacement_end is None:
            raise ValueError("FIM sample is missing replacement_end")
        start = sample.target_start + sample.replacement_start
        end = sample.target_start + sample.replacement_end
        return stream[start:end]
    return stream[sample.target_start : sample.target_end]


def _deterministic_random_bytes(
    sample: ReconstructionSample, length: int
) -> bytes:
    identity = (
        f"{sample.h264_path}:{sample.target_nal_index}:"
        f"{sample.replacement_start}"
    )
    digest = hashlib.sha256(
        identity.encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return bytes(rng.randrange(BYTE_VOCAB_SIZE) for _ in range(length))
