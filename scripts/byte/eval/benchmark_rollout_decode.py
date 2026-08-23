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
    megabyte_generate_batch,
    megabyte_max_new_bytes,
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
    parser.add_argument(
        "--dataset-mode",
        choices=("slice", "window"),
        default="slice",
        help=(
            "Must match the dataset_mode the checkpoint was trained with. "
            "Per-macroblock-sliced corpora train FIM with 'window' "
            "(ByteStreamWindowDataset) because a gap spans many single-MB "
            "NALs; 'slice' (the default, ByteSliceDataset) has no such "
            "notion and will silently select zero FIM samples if the "
            "checkpoint actually needs window mode."
        ),
    )
    parser.add_argument("--window-min-frames", type=int, default=2)
    parser.add_argument("--fixed-fim-holes", action="store_true")
    parser.add_argument("--fixed-fim-holes-per-window", type=int, default=0)
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Skip the timing sweep. Instead, for each prefix run the sequential "
            "(MegabyteInference) and batched megabyte generators at group_size=1, "
            "temperature=0 (greedy) and assert byte-identical output. Only "
            "meaningful for byte_patch_size>1 checkpoints."
        ),
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
        candidates = megabyte_generate_batch(
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


def verify_batched_generation(
    model: Any,
    samples: list[Any],
    device: torch.device,
) -> bool:
    """Greedy-diff sequential vs. batched megabyte generation, byte-for-byte.

    Runs at temperature=0 (argmax) so there is exactly one correct output per
    prefix, then checks the batched generator (group_size=1) reproduces
    exactly what the production sequential path (MegabyteInference) produces.
    Only meaningful for byte_patch_size>1 checkpoints; a mismatch here means
    the batched KV-cache/masking logic has a bug that would otherwise
    silently corrupt every gradient computed from it.
    """
    raw_model = _unwrap_model(model)
    patch_size = int(raw_model.config.byte_patch_size)
    if patch_size <= 1:
        print("--verify only applies to byte_patch_size>1 checkpoints; nothing to check.")
        return True

    all_ok = True
    for sample_index, sample in enumerate(samples):
        sequential = _megabyte_generate_one(raw_model, sample, device, 0.0, 0, 1.0)
        batched = megabyte_generate_batch(raw_model, sample, device, 1, 0.0, 0, 1.0)
        if sequential is None or batched is None:
            print(f"[{sample_index}] SKIP (target_length exceeds max_seq_length)")
            continue
        batched_bytes = batched[0]
        ok = sequential == batched_bytes
        all_ok = all_ok and ok
        status = "OK" if ok else "MISMATCH"
        print(
            f"[{sample_index}] {status} target_length={sample.target_length} "
            f"frame_index={sample.frame_index}",
            flush=True,
        )
        if not ok:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(sequential, batched_bytes)) if a != b),
                min(len(sequential), len(batched_bytes)),
            )
            print(
                f"    lengths: sequential={len(sequential)} batched={len(batched_bytes)}; "
                f"first differing byte at index {first_diff}: "
                f"{sequential[first_diff:first_diff + 8]!r} vs {batched_bytes[first_diff:first_diff + 8]!r}",
                flush=True,
            )
    return all_ok


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    args.num_samples = args.num_prefixes
    samples = build_eval_samples(args)
    samples = samples[: args.num_prefixes]
    print(f"Benchmarking on {len(samples)} prefixes", flush=True)

    model = load_model(args.checkpoint_dir, device)

    if args.verify:
        ok = verify_batched_generation(model, samples, device)
        print("\nVERIFY " + ("PASSED" if ok else "FAILED"), flush=True)
        sys.exit(0 if ok else 1)

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
