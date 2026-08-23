"""Offline byte-checkpoint reconstruction evaluation.

This script is intentionally separate from training. It loads a fixed set of
validation reconstruction contexts once, evaluates each checkpoint on exactly
those contexts, and writes scalar metrics plus visual comparison panels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in missing-dependency envs.
    Image = None

from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    FIM_FORMATS,
    REFERENCE_MODES,
    ByteDataConfig,
    ByteDataModule,
)
from litgpt.byte.megabyte_inference import sample_tokens
from litgpt.byte.reconstruction import (
    ReconstructionSample,
    _unwrap_model,
    decode_frame,
    image_psnr,
    image_ssim,
    replace_target_nal,
    save_reconstruction_sample_manifest,
    select_reconstruction_samples,
)
from litgpt.config import Config
from litgpt.model import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-manifest-rows", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--task", choices=("fim",), default="fim")
    parser.add_argument("--dataset-mode", choices=("slice", "window"), default="slice")
    parser.add_argument("--window-min-frames", type=int, default=2)
    parser.add_argument("--fixed-fim-holes", action="store_true")
    parser.add_argument("--fixed-fim-holes-per-window", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--split-by-video", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-fim", type=float, default=1.0)
    parser.add_argument("--fim-format", choices=FIM_FORMATS, default="psm")
    parser.add_argument("--fim-min-gap", type=int, default=1024)
    parser.add_argument("--fim-max-gap", type=int, default=8192)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument("--num-ref-slices", type=int, default=1)
    parser.add_argument("--reference-mode", choices=REFERENCE_MODES, default="normal")
    parser.add_argument("--target-nal-types", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--use-eos", action="store_true")
    parser.add_argument("--no-sps-pps-conditioning", action="store_true")
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--max-target-bytes", type=int, default=8192)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("greedy", "best_of_n", "beam"),
        default=["greedy", "best_of_n"],
    )
    parser.add_argument("--best-of-n", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--save-candidate-details", action="store_true")
    return parser.parse_args()


def require_png_writer() -> None:
    if Image is None:
        raise RuntimeError(
            "PNG output requires Pillow. Install it in the evaluation env with: "
            "pip install pillow"
        )


def load_model(checkpoint_dir: Path, device: torch.device) -> GPT:
    print(f"Loading checkpoint: {checkpoint_dir}", flush=True)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    model = GPT(config)
    checkpoint = torch.load(
        checkpoint_dir / "lit_model.pth",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {_strip_compile_prefix(key): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_dir.name}", flush=True)
    return model


def _strip_compile_prefix(key: str) -> str:
    for prefix in ("_forward_module.", "_orig_mod.", "module."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def build_eval_samples(args: argparse.Namespace) -> list[ReconstructionSample]:
    print("Building byte validation dataset...", flush=True)
    data_config = ByteDataConfig(
        p_fim=args.p_fim,
        fim_format=args.fim_format,
        use_eos=args.use_eos,
        num_ref_slices=args.num_ref_slices,
        reference_mode=args.reference_mode,
        target_nal_types=tuple(args.target_nal_types),
        fim_min_gap=args.fim_min_gap,
        fim_max_gap=args.fim_max_gap,
        slice_header_guard_bytes=args.slice_header_guard_bytes,
        num_workers=args.num_workers,
        condition_on_sps_pps=not args.no_sps_pps_conditioning,
        default_max_seq_length=args.block_size,
        val_fraction=args.val_fraction,
        split_by_video=args.split_by_video,
        seed=args.seed,
        # Default "slice" builds ByteSliceDataset, which has no notion of a
        # gap spanning many NALs. Per-macroblock-sliced corpora train FIM with
        # --dataset-mode window (ByteStreamWindowDataset) specifically because
        # a 64-1400B gap there spans ~100+ single-MB NALs; evaluating such a
        # checkpoint without the matching mode here silently selects zero (or
        # architecturally mismatched) FIM samples.
        dataset_mode=getattr(args, "dataset_mode", "slice"),
        window_min_frames=getattr(args, "window_min_frames", 2),
        fixed_fim_holes=getattr(args, "fixed_fim_holes", False),
        fixed_fim_holes_per_window=getattr(args, "fixed_fim_holes_per_window", 0),
    )
    data = ByteDataModule(
        manifest_path=args.manifest,
        config=data_config,
        max_manifest_rows=None if args.max_manifest_rows == 0 else args.max_manifest_rows,
        nal_index_path=args.nal_index_path,
    )
    data.connect(tokenizer=None, batch_size=1, max_seq_length=args.block_size)
    print("Setting up ByteDataModule...", flush=True)
    data.setup()
    if data.val_dataset is None:
        raise RuntimeError("ByteDataModule did not produce a validation dataset")
    print("Selecting fixed reconstruction samples...", flush=True)
    samples = select_reconstruction_samples(
        data.val_dataset,
        args.num_samples,
        args.max_target_bytes,
        args.task,
        force_eos_stopping=args.use_eos,
    )
    if not samples:
        raise RuntimeError("No reconstruction samples matched the requested filters")
    print(f"Selected {len(samples)} reconstruction samples", flush=True)
    return samples


@torch.inference_mode()
def generate_bytes(
    model: nn.Module,
    sample: ReconstructionSample,
    device: torch.device,
    *,
    strategy: str,
    num_candidates: int = 1,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> list[bytes]:
    if strategy not in {"greedy", "sample"}:
        raise ValueError(f"Unknown generation strategy: {strategy}")
    raw_model = _unwrap_model(model)
    raw_model.eval()
    batch_size = 1 if strategy == "greedy" else num_candidates
    prompt = sample.prompt_ids.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
    prompt_length = prompt.size(1)
    if prompt_length + sample.target_length - 1 > raw_model.max_seq_length:
        return []

    generated = [[] for _ in range(batch_size)]
    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw_model.parameters()).dtype
    raw_model.set_kv_cache(
        batch_size=batch_size,
        max_seq_length=raw_model.max_seq_length,
        device=device,
        dtype=cache_dtype,
    )
    try:
        input_pos = torch.arange(prompt_length, device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = raw_model(
                prompt,
                input_pos=input_pos,
                input_pos_maxp1=prompt_length,
                region_ids=region_ids,
                offset_ids=offset_ids,
            )
        for generated_idx in range(sample.target_length):
            next_logits = logits[:, -1, :BYTE_VOCAB_SIZE]
            if strategy == "greedy":
                tokens = next_logits.argmax(dim=-1)
            else:
                tokens = sample_tokens(next_logits, temperature, top_k, top_p)
            for candidate_idx, token in enumerate(tokens.tolist()):
                generated[candidate_idx].append(token)
            if generated_idx == sample.target_length - 1:
                break
            position = prompt_length + generated_idx
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = raw_model(
                    tokens.unsqueeze(1),
                    input_pos=torch.full((batch_size, 1), position, device=device, dtype=torch.long),
                    input_pos_maxp1=position + 1,
                    region_ids=torch.full(
                        (batch_size, 1),
                        sample.generation_region_id,
                        device=device,
                        dtype=torch.long,
                    ),
                    offset_ids=torch.full(
                        (batch_size, 1),
                        sample.generation_offset_start + generated_idx,
                        device=device,
                        dtype=torch.long,
                    ),
                )
    finally:
        raw_model.clear_kv_cache()
    return [bytes(candidate) for candidate in generated]


@torch.inference_mode()
def beam_search_bytes(
    model: nn.Module,
    sample: ReconstructionSample,
    device: torch.device,
    beam_width: int,
) -> list[bytes]:
    """Return fixed-length byte candidates sorted by model log probability."""
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    raw_model = _unwrap_model(model)
    raw_model.eval()
    prompt = sample.prompt_ids.to(device).unsqueeze(0).expand(beam_width, -1).contiguous()
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0).expand(beam_width, -1).contiguous()
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0).expand(beam_width, -1).contiguous()
    prompt_length = prompt.size(1)
    if prompt_length + sample.target_length - 1 > raw_model.max_seq_length:
        return []

    sequences = torch.empty((beam_width, 0), device=device, dtype=torch.long)
    scores = torch.full((beam_width,), float("-inf"), device=device)
    scores[0] = 0.0
    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw_model.parameters()).dtype
    raw_model.set_kv_cache(
        batch_size=beam_width,
        max_seq_length=raw_model.max_seq_length,
        device=device,
        dtype=cache_dtype,
    )
    try:
        input_pos = torch.arange(prompt_length, device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = raw_model(
                prompt,
                input_pos=input_pos,
                input_pos_maxp1=prompt_length,
                region_ids=region_ids,
                offset_ids=offset_ids,
            )
        for generated_idx in range(sample.target_length):
            log_probs = F.log_softmax(logits[:, -1, :BYTE_VOCAB_SIZE].float(), dim=-1)
            flat_scores = (scores[:, None] + log_probs).reshape(-1)
            next_scores, flat_indices = torch.topk(flat_scores, k=beam_width)
            parents = flat_indices // BYTE_VOCAB_SIZE
            tokens = flat_indices % BYTE_VOCAB_SIZE
            _reorder_kv_cache(raw_model, parents)
            sequences = torch.cat(
                [sequences.index_select(0, parents), tokens[:, None]], dim=1
            )
            scores = next_scores
            if generated_idx == sample.target_length - 1:
                break
            position = prompt_length + generated_idx
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = raw_model(
                    tokens.unsqueeze(1),
                    input_pos=torch.full((beam_width, 1), position, device=device, dtype=torch.long),
                    input_pos_maxp1=position + 1,
                    region_ids=torch.full(
                        (beam_width, 1),
                        sample.generation_region_id,
                        device=device,
                        dtype=torch.long,
                    ),
                    offset_ids=torch.full(
                        (beam_width, 1),
                        sample.generation_offset_start + generated_idx,
                        device=device,
                        dtype=torch.long,
                    ),
                )
    finally:
        raw_model.clear_kv_cache()
    return [bytes(row.tolist()) for row in sequences.detach().cpu()]


def _reorder_kv_cache(model: nn.Module, parents: Tensor) -> None:
    """Keep beam-search KV cache aligned with selected parent beams."""
    for block in model.transformer.h:
        kv_cache = block.attn.kv_cache
        if kv_cache is None:
            continue
        kv_cache.k = kv_cache.k.index_select(0, parents)
        kv_cache.v = kv_cache.v.index_select(0, parents)


def evaluate_checkpoint(
    model: GPT,
    samples: list[ReconstructionSample],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Tensor | str]]]:
    totals: dict[str, list[float] | int] = {
        "greedy_psnr": [],
        "greedy_ssim": [],
        "best_psnr": [],
        "best_ssim": [],
        "valid_candidates": [],
        "best_rank": [],
        "unique_candidates": [],
        "beam_top_psnr": [],
        "beam_top_ssim": [],
        "beam_best_psnr": [],
        "beam_best_ssim": [],
        "beam_valid_candidates": [],
        "beam_best_rank": [],
        "deleted_strict_psnr": [],
        "deleted_strict_ssim": [],
        "deleted_default_psnr": [],
        "deleted_default_ssim": [],
        "attempted": 0,
        "greedy_decoded": 0,
        "best_any_decoded": 0,
        "beam_top_decoded": 0,
        "beam_any_decoded": 0,
        "deleted_strict_decoded": 0,
        "deleted_default_decoded": 0,
    }
    details: list[dict[str, Any]] = []
    frames: dict[int, dict[str, Tensor | str]] = {}

    for sample_index, sample in enumerate(samples):
        if sample_index == 0 or (sample_index + 1) % 10 == 0:
            print(
                f"  {checkpoint_name}: sample {sample_index + 1}/{len(samples)}",
                flush=True,
            )
        stream = sample.h264_path.read_bytes()
        reference, reference_status = decode_frame(
            stream, sample.frame_index, args.ffmpeg_binary, args.timeout_sec, strict_syntax=True
        )
        totals["attempted"] += 1
        if reference is None:
            details.append({
                "checkpoint": checkpoint_name,
                "sample_index": sample_index,
                "reference_status": reference_status,
            })
            continue

        sample_frames: dict[str, Tensor | str] = {"ground_truth": reference.cpu()}
        deleted_stream = replace_target_nal(stream, sample, b"")
        deleted_strict, deleted_strict_status = decode_frame(
            deleted_stream, sample.frame_index, args.ffmpeg_binary, args.timeout_sec, strict_syntax=True
        )
        deleted_default, deleted_default_status = decode_frame(
            deleted_stream, sample.frame_index, args.ffmpeg_binary, args.timeout_sec
        )
        if deleted_strict is not None:
            totals["deleted_strict_decoded"] += 1
            totals["deleted_strict_psnr"].append(image_psnr(reference, deleted_strict))
            totals["deleted_strict_ssim"].append(image_ssim(reference, deleted_strict))
            sample_frames["deleted_strict"] = deleted_strict.cpu()
        if deleted_default is not None:
            totals["deleted_default_decoded"] += 1
            totals["deleted_default_psnr"].append(image_psnr(reference, deleted_default))
            totals["deleted_default_ssim"].append(image_ssim(reference, deleted_default))
            sample_frames["deleted_default"] = deleted_default.cpu()

        greedy_record: dict[str, Any] = {"decoded": False}
        if "greedy" in args.strategies:
            if sample_index == 0:
                print(f"  {checkpoint_name}: running greedy decoding", flush=True)
            greedy_candidates = generate_bytes(model, sample, device, strategy="greedy")
            if greedy_candidates:
                greedy_stream = replace_target_nal(stream, sample, greedy_candidates[0])
                greedy_frame, greedy_status = decode_frame(
                    greedy_stream,
                    sample.frame_index,
                    args.ffmpeg_binary,
                    args.timeout_sec,
                    strict_syntax=True,
                )
                greedy_record["status"] = greedy_status
                if greedy_frame is not None:
                    psnr = image_psnr(reference, greedy_frame)
                    ssim = image_ssim(reference, greedy_frame)
                    totals["greedy_decoded"] += 1
                    totals["greedy_psnr"].append(psnr)
                    totals["greedy_ssim"].append(ssim)
                    sample_frames["greedy"] = greedy_frame.cpu()
                    greedy_record.update({"decoded": True, "psnr": psnr, "ssim": ssim})

        best_record: dict[str, Any] = {"decoded": False, "num_valid": 0}
        if "best_of_n" in args.strategies:
            if sample_index == 0:
                print(
                    f"  {checkpoint_name}: running best@{args.best_of_n} sampling",
                    flush=True,
                )
            sampled = generate_bytes(
                model,
                sample,
                device,
                strategy="sample",
                num_candidates=args.best_of_n,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            totals["unique_candidates"].append(len(set(sampled)))
            best = None
            valid = 0
            for rank, candidate in enumerate(sampled):
                candidate_stream = replace_target_nal(stream, sample, candidate)
                candidate_frame, candidate_status = decode_frame(
                    candidate_stream,
                    sample.frame_index,
                    args.ffmpeg_binary,
                    args.timeout_sec,
                    strict_syntax=True,
                )
                if candidate_frame is None:
                    continue
                valid += 1
                psnr = image_psnr(reference, candidate_frame)
                ssim = image_ssim(reference, candidate_frame)
                if best is None or psnr > best["psnr"]:
                    best = {
                        "rank": rank,
                        "psnr": psnr,
                        "ssim": ssim,
                        "frame": candidate_frame.cpu(),
                    }
            totals["valid_candidates"].append(valid)
            if best is not None:
                totals["best_any_decoded"] += 1
                totals["best_psnr"].append(best["psnr"])
                totals["best_ssim"].append(best["ssim"])
                totals["best_rank"].append(best["rank"])
                sample_frames["best_of_n"] = best["frame"]
                best_record.update({
                    "decoded": True,
                    "num_valid": valid,
                    "best_rank": best["rank"],
                    "best_psnr": best["psnr"],
                    "best_ssim": best["ssim"],
                })
            else:
                best_record["num_valid"] = valid

        beam_record: dict[str, Any] = {"top_decoded": False, "num_valid": 0}
        if "beam" in args.strategies:
            if sample_index == 0:
                print(
                    f"  {checkpoint_name}: running beam search width {args.beam_width}",
                    flush=True,
                )
            beam_candidates = beam_search_bytes(model, sample, device, args.beam_width)
            beam_best = None
            beam_valid = 0
            for rank, candidate in enumerate(beam_candidates):
                candidate_stream = replace_target_nal(stream, sample, candidate)
                candidate_frame, candidate_status = decode_frame(
                    candidate_stream,
                    sample.frame_index,
                    args.ffmpeg_binary,
                    args.timeout_sec,
                    strict_syntax=True,
                )
                if candidate_frame is None:
                    if rank == 0:
                        beam_record["top_status"] = candidate_status
                    continue
                beam_valid += 1
                psnr = image_psnr(reference, candidate_frame)
                ssim = image_ssim(reference, candidate_frame)
                if rank == 0:
                    totals["beam_top_decoded"] += 1
                    totals["beam_top_psnr"].append(psnr)
                    totals["beam_top_ssim"].append(ssim)
                    sample_frames["beam_top"] = candidate_frame.cpu()
                    beam_record.update(
                        {"top_decoded": True, "top_psnr": psnr, "top_ssim": ssim}
                    )
                if beam_best is None or psnr > beam_best["psnr"]:
                    beam_best = {
                        "rank": rank,
                        "psnr": psnr,
                        "ssim": ssim,
                        "frame": candidate_frame.cpu(),
                    }
            totals["beam_valid_candidates"].append(beam_valid)
            if beam_best is not None:
                totals["beam_any_decoded"] += 1
                totals["beam_best_psnr"].append(beam_best["psnr"])
                totals["beam_best_ssim"].append(beam_best["ssim"])
                totals["beam_best_rank"].append(beam_best["rank"])
                sample_frames["beam_best"] = beam_best["frame"]
                beam_record.update(
                    {
                        "num_valid": beam_valid,
                        "best_rank": beam_best["rank"],
                        "best_psnr": beam_best["psnr"],
                        "best_ssim": beam_best["ssim"],
                    }
                )
            else:
                beam_record["num_valid"] = beam_valid

        if sample_index < args.num_visualizations:
            sample_frames["deleted_strict_status"] = deleted_strict_status
            sample_frames["deleted_default_status"] = deleted_default_status
            frames[sample_index] = sample_frames

        details.append({
            "checkpoint": checkpoint_name,
            "sample_index": sample_index,
            "h264_path": str(sample.h264_path),
            "frame_index": sample.frame_index,
            "target_length": sample.target_length,
            "greedy": greedy_record,
            "best_of_n": best_record,
            "beam": beam_record,
        })

    attempted = int(totals["attempted"])
    summary = {
        "checkpoint": checkpoint_name,
        "num_samples": attempted,
        "greedy_decode_rate": _rate(totals["greedy_decoded"], attempted),
        "greedy_psnr_mean": _mean(totals["greedy_psnr"]),
        "greedy_ssim_mean": _mean(totals["greedy_ssim"]),
        "best_of_n": args.best_of_n,
        "best_decode_rate_any": _rate(totals["best_any_decoded"], attempted),
        "best_psnr_mean": _mean(totals["best_psnr"]),
        "best_ssim_mean": _mean(totals["best_ssim"]),
        "valid_candidates_mean": _mean(totals["valid_candidates"]),
        "valid_candidates_median": _median(totals["valid_candidates"]),
        "best_rank_mean": _mean(totals["best_rank"]),
        "unique_candidates_mean": _mean(totals["unique_candidates"]),
        "beam_width": args.beam_width,
        "beam_top_decode_rate": _rate(totals["beam_top_decoded"], attempted),
        "beam_top_psnr_mean": _mean(totals["beam_top_psnr"]),
        "beam_top_ssim_mean": _mean(totals["beam_top_ssim"]),
        "beam_best_decode_rate_any": _rate(totals["beam_any_decoded"], attempted),
        "beam_best_psnr_mean": _mean(totals["beam_best_psnr"]),
        "beam_best_ssim_mean": _mean(totals["beam_best_ssim"]),
        "beam_valid_candidates_mean": _mean(totals["beam_valid_candidates"]),
        "beam_valid_candidates_median": _median(totals["beam_valid_candidates"]),
        "beam_best_rank_mean": _mean(totals["beam_best_rank"]),
        "deleted_strict_decode_rate": _rate(totals["deleted_strict_decoded"], attempted),
        "deleted_strict_psnr_mean": _mean(totals["deleted_strict_psnr"]),
        "deleted_strict_ssim_mean": _mean(totals["deleted_strict_ssim"]),
        "deleted_default_decode_rate": _rate(totals["deleted_default_decoded"], attempted),
        "deleted_default_psnr_mean": _mean(totals["deleted_default_psnr"]),
        "deleted_default_ssim_mean": _mean(totals["deleted_default_ssim"]),
        "greedy_vs_deleted_default_psnr_delta": _delta_mean(
            totals["greedy_psnr"], totals["deleted_default_psnr"]
        ),
        "best_vs_deleted_default_psnr_delta": _delta_mean(
            totals["best_psnr"], totals["deleted_default_psnr"]
        ),
        "beam_top_vs_deleted_default_psnr_delta": _delta_mean(
            totals["beam_top_psnr"], totals["deleted_default_psnr"]
        ),
        "beam_best_vs_deleted_default_psnr_delta": _delta_mean(
            totals["beam_best_psnr"], totals["deleted_default_psnr"]
        ),
    }
    return summary, details, frames


def _rate(numerator: Any, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _mean(values: Any) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: Any) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _delta_mean(left: Any, right: Any) -> float | None:
    if not left or not right:
        return None
    return _mean(left) - _mean(right)


def save_panels(
    frames: dict[int, dict[str, Tensor | str]], checkpoint_dir: Path, checkpoint_name: str
) -> None:
    if frames:
        print(f"Saving {len(frames)} visual panels for {checkpoint_name}", flush=True)
    for sample_index, sample_frames in frames.items():
        reference = sample_frames["ground_truth"]
        assert isinstance(reference, Tensor)
        missing = torch.zeros_like(reference)
        missing[..., 0] = 1.0
        columns = [
            sample_frames.get("ground_truth", missing),
            sample_frames.get("deleted_strict", missing),
            sample_frames.get("deleted_default", missing),
            sample_frames.get("greedy", missing),
            sample_frames.get("best_of_n", missing),
            sample_frames.get("beam_top", missing),
            sample_frames.get("beam_best", missing),
        ]
        separator = torch.ones((reference.shape[0], 4, reference.shape[2]), dtype=reference.dtype)
        panel_parts: list[Tensor] = []
        for idx, column in enumerate(columns):
            if idx:
                panel_parts.append(separator)
            assert isinstance(column, Tensor)
            panel_parts.append(column)
        panel = torch.cat(panel_parts, dim=1).clamp(0, 1)
        save_png(panel, checkpoint_dir / f"sample_{sample_index:04d}_panel.png")
        legend = {
            "columns": [
                "ground_truth",
                "deleted_gap_strict",
                "deleted_gap_ffmpeg_default",
                "model_greedy_strict",
                "model_best_of_n_strict",
                "model_beam_top_strict",
                "model_beam_best_strict",
            ],
            "red_tile": "decode failed or no frame",
            "checkpoint": checkpoint_name,
            "sample_index": sample_index,
        }
        (checkpoint_dir / f"sample_{sample_index:04d}_panel.json").write_text(
            json.dumps(legend, indent=2) + "\n", encoding="utf-8"
        )


def save_png(image: Tensor, path: Path) -> None:
    if Image is None:
        raise RuntimeError("PNG output requires Pillow")
    image_u8 = (image.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    height, width, channels = image_u8.shape
    if channels != 3:
        raise ValueError("Expected RGB image tensor")
    Image.fromarray(image_u8.numpy(), mode="RGB").save(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = True) -> None:
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=True) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    keys = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    require_png_writer()
    torch.set_float32_matmul_precision("high")
    if args.temperature <= 0 and "best_of_n" in args.strategies:
        raise ValueError("--temperature must be positive for best_of_n sampling")
    if args.beam_width <= 0:
        raise ValueError("--beam-width must be positive")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps(jsonable(vars(args)), indent=2) + "\n",
        encoding="utf-8",
    )

    samples = build_eval_samples(args)
    save_reconstruction_sample_manifest(samples, args.out_dir / "samples.json")
    print(f"Saved fixed sample manifest to {args.out_dir / 'samples.json'}", flush=True)
    summaries: list[dict[str, Any]] = []
    metrics_path = args.out_dir / "metrics.jsonl"
    details_path = args.out_dir / "sample_details.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    details_path.write_text("", encoding="utf-8")

    for checkpoint_dir in args.checkpoint_dirs:
        checkpoint_name = checkpoint_dir.name
        print(f"Evaluating {checkpoint_name} on {len(samples)} samples", flush=True)
        model = load_model(checkpoint_dir, device)
        summary, details, frames = evaluate_checkpoint(model, samples, args, device, checkpoint_name)
        summaries.append(summary)
        write_jsonl(metrics_path, [summary])
        if args.save_candidate_details:
            write_jsonl(details_path, details)
        frame_dir = args.out_dir / "frames" / checkpoint_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        save_panels(frames, frame_dir, checkpoint_name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(summary, indent=2), flush=True)

    write_summary_csv(args.out_dir / "summary.csv", summaries)
    print(f"Wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
