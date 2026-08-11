"""Benchmark G-sample rollout generation + ffmpeg decode timing.

Measures wall-clock cost of the two building blocks a GRPO-style RL loop
needs per policy step: (1) sampling G candidate continuations per prefix from
the current policy, and (2) scoring each candidate by splicing it into the
stream and decoding it with ffmpeg. The two are timed and reported
separately so it's clear which side (model forward pass vs. subprocess
decode) dominates, and whether G groups of size 8-64 are seconds or minutes
per step.

Reuses the same generation/decode plumbing as the reconstruction-eval
harness (helpers/checkpoint_eval_helpers.py) so the timings reflect the real
code path rather than a reimplementation.

Example:
    python scripts/byte/eval/benchmark_rollout_decode.py \\
        manifest.jsonl --checkpoint-dir out/checkpoint \\
        --num-prefixes 8 --group-sizes 4 8 16 32 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte.data import BYTE_VOCAB_SIZE  # noqa: E402
from litgpt.byte.megabyte_inference import (  # noqa: E402
    MegabyteInference,
    megabyte_max_new_bytes,
    megabyte_prompt_patches,
)
from litgpt.byte.reconstruction import (  # noqa: E402
    _unwrap_model,
    decode_frame,
    replace_target_nal,
)
from scripts.byte.eval.helpers.checkpoint_eval_helpers import (  # noqa: E402
    build_eval_samples,
    generate_bytes,
    load_model,
    sample_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("out/rollout_decode_benchmark"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--max-manifest-rows", type=int, default=0)
    parser.add_argument("--task", choices=("ar", "fim"), default="ar")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--split-by-video", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-fim", type=float, default=0.0)
    parser.add_argument("--fim-format", default="psm")
    parser.add_argument("--fim-min-gap", type=int, default=1024)
    parser.add_argument("--fim-max-gap", type=int, default=8192)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument("--num-ref-slices", type=int, default=1)
    parser.add_argument("--reference-mode", default="normal")
    parser.add_argument("--target-nal-types", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--use-eos", action="store_true")
    parser.add_argument("--no-sps-pps-conditioning", action="store_true")
    parser.add_argument("--max-target-bytes", type=int, default=2048)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Values of G (rollouts per prefix) to sweep.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--megabyte-mode",
        choices=("sequential", "batched"),
        default="sequential",
        help=(
            "Only applies to byte_patch_size>1 checkpoints. 'sequential' runs G "
            "independent single-sample rollouts (matches production MegabyteInference "
            "today). 'batched' is an experimental path that batches all G candidates "
            "into one loop -- only valid for fixed-schedule reconstruction samples, "
            "not free-run generation."
        ),
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=8,
        help="Thread pool size for parallel ffmpeg decode calls (subprocess-bound, so threads help).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each (prefix, G) timing this many times to see variance.",
    )
    return parser.parse_args()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 90),
        "min": min(values),
        "max": max(values),
    }


def time_decode_group(
    stream: bytes,
    sample: Any,
    candidates: list[bytes],
    ffmpeg_binary: str,
    timeout_sec: int,
    decode_workers: int,
) -> tuple[float, int]:
    """Decode all candidates in a group, in parallel. Returns (wall_time, num_decoded_ok)."""

    def _decode_one(candidate: bytes) -> bool:
        candidate_stream = replace_target_nal(stream, sample, candidate)
        frame, status = decode_frame(
            candidate_stream,
            sample.frame_index,
            ffmpeg_binary,
            timeout_sec,
            strict_syntax=True,
        )
        return frame is not None

    start = time.perf_counter()
    if decode_workers <= 1:
        results = [_decode_one(c) for c in candidates]
    else:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            results = list(pool.map(_decode_one, candidates))
    elapsed = time.perf_counter() - start
    return elapsed, sum(results)


@torch.inference_mode()
def _megabyte_generate_one(
    raw_model: Any,
    sample: Any,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
) -> bytes | None:
    """Generate one candidate for a byte_patch_size>1 checkpoint.

    MegabyteInference is single-sample only (its global-decoder KV cache is
    batch_size=1 by construction), so unlike the flat-vocab path there is no
    batched-candidates shortcut here -- this is called once per candidate.
    """
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0)
    mi = MegabyteInference(raw_model, prompt, region_ids, offset_ids, device)
    generated: list[int] = []
    try:
        if sample.target_length > mi.max_new_bytes:
            return None
        for generated_idx in range(sample.target_length):
            logits = mi.next_logits()[:, :BYTE_VOCAB_SIZE]
            token = int(sample_tokens(logits, temperature, top_k, top_p).item())
            generated.append(token)
            mi.append(
                token,
                sample.generation_region_id,
                sample.generation_offset_start + generated_idx,
            )
    finally:
        mi.close()
    return bytes(generated)


@torch.inference_mode()
def _megabyte_generate_batch(
    raw_model: Any,
    sample: Any,
    device: torch.device,
    batch_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[bytes] | None:
    """Generate a full group of candidates in one batched megabyte loop.

    Only valid because reconstruction-sample generation has a *deterministic*
    region/offset schedule (fixed by sample.generation_region_id and
    sample.generation_offset_start + step index) -- it does not depend on
    which bytes get sampled, unlike free_run_rollout's start-code-tracking
    schedule. That means all G candidates share identical patch metadata and
    can run through the same global-decoder forward passes together; only the
    sampled byte content differs per batch row. This does NOT generalize to
    free-run generation as-is.
    """
    patch_size = int(raw_model.config.byte_patch_size)
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0)
    patched_ids, patched_regions, patched_offsets = megabyte_prompt_patches(
        prompt, region_ids, offset_ids, patch_size
    )
    prompt_patches = patched_ids.size(1)
    if prompt_patches > int(raw_model.max_seq_length):
        return None
    if sample.target_length > megabyte_max_new_bytes(raw_model, prompt.size(1)):
        return None

    def _autocast():
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        )

    b = batch_size
    patched_ids_b = patched_ids.expand(b, -1, -1).contiguous()
    patched_regions_b = patched_regions.expand(b, -1, -1).contiguous()
    patched_offsets_b = patched_offsets.expand(b, -1, -1).contiguous()

    # Size the KV cache to what this generation actually needs (prompt patches
    # plus the patches the target span will occupy), not the model's full
    # max_seq_length. At batch_size=1 (sequential mode) the gap between "needed"
    # and "full" is harmless; multiplied by a batch dimension of G it is not --
    # this is what made G=16 borderline and G=32/64 OOM.
    needed_patches = min(
        int(raw_model.max_seq_length),
        prompt_patches + -(-sample.target_length // patch_size),
    )
    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw_model.parameters()).dtype
    try:
        raw_model.set_kv_cache(
            batch_size=b, max_seq_length=needed_patches, device=device, dtype=cache_dtype
        )
        with _autocast():
            global_output = raw_model.megabyte_global_forward(
                patched_ids_b,
                input_pos=torch.arange(prompt_patches, device=device, dtype=torch.long),
                input_pos_maxp1=prompt_patches,
                region_ids=patched_regions_b,
                offset_ids=patched_offsets_b,
            )
        global_output = global_output[:, -1]  # (B, n_embd)

        generated: list[list[int]] = [[] for _ in range(b)]
        current_tokens = torch.zeros((b, 0), dtype=torch.long, device=device)
        current_region_ids: list[int] = []
        current_offset_ids: list[int] = []
        position = prompt_patches
        for step in range(sample.target_length):
            if current_tokens.size(1) == patch_size:
                # Commit the just-completed patch: one batched global-forward step.
                patch_regions = torch.tensor(
                    current_region_ids, device=device, dtype=torch.long
                ).view(1, 1, patch_size).expand(b, -1, -1)
                patch_offsets = torch.tensor(
                    current_offset_ids, device=device, dtype=torch.long
                ).view(1, 1, patch_size).expand(b, -1, -1)
                with _autocast():
                    out = raw_model.megabyte_global_forward(
                        current_tokens.view(b, 1, patch_size),
                        input_pos=torch.tensor([position], device=device, dtype=torch.long),
                        input_pos_maxp1=position + 1,
                        region_ids=patch_regions,
                        offset_ids=patch_offsets,
                    )
                global_output = out[:, -1]
                position += 1
                current_tokens = torch.zeros((b, 0), dtype=torch.long, device=device)
                current_region_ids = []
                current_offset_ids = []

            with _autocast():
                logits = raw_model.megabyte_local_next_logits(
                    global_output, current_tokens
                )[:, :BYTE_VOCAB_SIZE]
            tokens = sample_tokens(logits, temperature, top_k, top_p)  # (B,)
            for row, token in enumerate(tokens.tolist()):
                generated[row].append(token)
            current_tokens = torch.cat([current_tokens, tokens.unsqueeze(1)], dim=1)
            current_region_ids.append(sample.generation_region_id)
            current_offset_ids.append(sample.generation_offset_start + step)
    finally:
        raw_model.clear_kv_cache()
    return [bytes(row) for row in generated]


def generate_group(
    model: Any,
    sample: Any,
    device: torch.device,
    *,
    group_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    megabyte_mode: str = "sequential",
) -> tuple[list[bytes], str]:
    """Dispatch to the flat batched sampler or one of the megabyte samplers.

    Returns (candidates, generation_mode):
      "batched"             -- patch_size==1, one forward pass covers all G.
      "megabyte_sequential" -- patch_size>1, G separate single-sample rollouts
                                (the current real per-step cost).
      "megabyte_batched"    -- patch_size>1, experimental: one batched loop
                                over all G, exploiting the fixed metadata
                                schedule of reconstruction samples.
    """
    raw_model = _unwrap_model(model)
    patch_size = int(raw_model.config.byte_patch_size)
    if patch_size <= 1:
        return (
            generate_bytes(
                model,
                sample,
                device,
                strategy="sample",
                num_candidates=group_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ),
            "batched",
        )
    if megabyte_mode == "batched":
        candidates = _megabyte_generate_batch(
            raw_model, sample, device, group_size, temperature, top_k, top_p
        )
        return (candidates or [], "megabyte_batched")
    candidates = []
    for _ in range(group_size):
        candidate = _megabyte_generate_one(
            raw_model, sample, device, temperature, top_k, top_p
        )
        if candidate is None:
            return [], "megabyte_sequential"
        candidates.append(candidate)
    return candidates, "megabyte_sequential"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    args.num_samples = args.num_prefixes
    samples = build_eval_samples(args)
    samples = samples[: args.num_prefixes]
    print(f"Benchmarking on {len(samples)} prefixes", flush=True)

    model = load_model(args.checkpoint_dir, device)

    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        stream = sample.h264_path.read_bytes()
        print(
            f"[{sample_index + 1}/{len(samples)}] target_length={sample.target_length} "
            f"frame_index={sample.frame_index}",
            flush=True,
        )
        for group_size in args.group_sizes:
            for repeat in range(args.repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                gen_start = time.perf_counter()
                candidates, generation_mode = generate_group(
                    model,
                    sample,
                    device,
                    group_size=group_size,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    megabyte_mode=args.megabyte_mode,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                gen_time = time.perf_counter() - gen_start

                if not candidates:
                    print(
                        f"  G={group_size} repeat={repeat}: prompt+target exceeds max_seq_length, skipped",
                        flush=True,
                    )
                    continue

                decode_time, num_ok = time_decode_group(
                    stream,
                    sample,
                    candidates,
                    args.ffmpeg_binary,
                    args.timeout_sec,
                    args.decode_workers,
                )

                row = {
                    "sample_index": sample_index,
                    "target_length": sample.target_length,
                    "group_size": group_size,
                    "repeat": repeat,
                    "generation_mode": generation_mode,
                    "gen_time_sec": gen_time,
                    "gen_time_per_sample_sec": gen_time / group_size,
                    "decode_time_sec": decode_time,
                    "decode_time_per_sample_sec": decode_time / group_size,
                    "total_time_sec": gen_time + decode_time,
                    "num_decoded_ok": num_ok,
                    "unique_candidates": len(set(candidates)),
                }
                rows.append(row)
                print(
                    f"  G={group_size} repeat={repeat} [{generation_mode}]: gen={gen_time:.3f}s "
                    f"decode={decode_time:.3f}s ({decode_time / group_size * 1000:.1f}ms/sample, "
                    f"{num_ok}/{group_size} ok) total={gen_time + decode_time:.3f}s",
                    flush=True,
                )

    csv_path = args.out_dir / "rollout_decode_timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-call timings to {csv_path}", flush=True)

    summary: dict[str, Any] = {}
    for group_size in args.group_sizes:
        group_rows = [r for r in rows if r["group_size"] == group_size]
        if not group_rows:
            continue
        summary[str(group_size)] = {
            "gen_time_sec": _summarize([r["gen_time_sec"] for r in group_rows]),
            "decode_time_sec": _summarize([r["decode_time_sec"] for r in group_rows]),
            "total_time_sec": _summarize([r["total_time_sec"] for r in group_rows]),
            "decode_time_per_sample_ms": _summarize(
                [r["decode_time_per_sample_sec"] * 1000 for r in group_rows]
            ),
            "num_groups": len(group_rows),
        }

    summary_path = args.out_dir / "rollout_decode_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}", flush=True)

    print("\n=== Summary (median seconds per group) ===")
    print(f"{'G':>4} | {'gen(s)':>10} | {'decode(s)':>10} | {'total(s)':>10} | {'decode/sample(ms)':>18}")
    for group_size in args.group_sizes:
        stats = summary.get(str(group_size))
        if stats is None:
            continue
        print(
            f"{group_size:>4} | {stats['gen_time_sec']['median']:>10.3f} | "
            f"{stats['decode_time_sec']['median']:>10.3f} | {stats['total_time_sec']['median']:>10.3f} | "
            f"{stats['decode_time_per_sample_ms']['median']:>18.1f}"
        )


if __name__ == "__main__":
    main()
