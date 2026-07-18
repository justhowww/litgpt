"""Report semantic FIM context-window sizes in bytes/tokens.

This script answers the paper-facing question:

    "If the hyperparameter is K prior frames of context, what byte/token budget
    does that imply on the AVC-LM window-FIM corpus?"

It does not train or evaluate a model. It samples deterministic window-FIM holes
with the same frame/gap/guard logic as ``ByteStreamWindowDataset`` and reports the
actual prompt/target/token costs after applying a semantic context cap.

For PSM + EOS:

    generation_prompt_tokens = metadata + previous-frame context + prefix + orphan + 3
    target_tokens            = hole + 1
    train_input_tokens       = generation_prompt_tokens + target_tokens - 1

The "-1" is the teacher-forcing shift: the final target token is a label, not an
input token.

Usage:
    python scripts/byte/reports/context_window_calculation.py MANIFEST \
        --nal-index-path DATA/nal_index.sqlite \
        --context-frames 4 \
        --max-window-bytes 16384 \
        --out-dir reports/context_window_4f
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte.data import (  # noqa: E402
    FIM_FORMATS,
    ByteStreamWindowDataset,
    default_nal_index_path,
    load_manifest_rows,
    load_nal_index,
)
from scripts.byte.eval.helpers.checkpoint_eval_helpers import jsonable  # noqa: E402


@dataclass(frozen=True)
class ContextWindowRow:
    sample_id: int
    dataset_index: int
    h264_path: str
    start_nal: int
    end_nal: int
    damaged_frame_index: int
    context_frames_requested: int
    context_frames_available: int
    context_frames_used: int
    metadata_bytes: int
    prior_context_bytes: int
    total_context_bytes: int
    damaged_frame_bytes: int
    prefix_bytes: int
    hole_bytes: int
    orphan_bytes: int
    prompt_tokens: int
    target_tokens: int
    train_input_tokens: int
    train_total_tokens: int
    fits_block_size: bool
    prompt_fraction_of_block: float
    train_input_fraction_of_block: float
    hole_fraction_of_damaged_frame: float
    prefix_fraction_of_damaged_frame: float
    orphan_fraction_of_damaged_frame: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-manifest-rows", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--context-frames",
        type=int,
        required=True,
        help="Number of complete prior frames to include before the damaged frame.",
    )
    parser.add_argument(
        "--include-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep SPS/PPS/non-VCL metadata at the start of the context window.",
    )
    parser.add_argument("--max-window-bytes", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--window-min-frames", type=int, default=2)
    parser.add_argument("--fim-format", choices=FIM_FORMATS, default="psm")
    parser.add_argument(
        "--use-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count SEQ_EOS in the target span. Phase 2 defaults to true.",
    )
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": float(max(values)),
    }


def fim_marker_count(fim_format: str) -> int:
    if fim_format == "psm":
        return 3  # FIM_BEGIN, FIM_HOLE, FIM_END
    return 1  # SPAN_BOS-style bridge layout


def build_dataset(args: argparse.Namespace) -> ByteStreamWindowDataset:
    rows = load_manifest_rows(
        args.manifest,
        max_rows=args.max_manifest_rows or None,
        report_progress=True,
    )
    index_path = args.nal_index_path or default_nal_index_path(args.manifest)
    nal_index = load_nal_index(index_path, args.manifest, rows) if index_path.is_file() else None
    if args.nal_index_path is not None and nal_index is None:
        raise FileNotFoundError(f"NAL index does not exist: {index_path}")
    return ByteStreamWindowDataset(
        rows,
        max_seq_length=args.max_window_bytes,
        min_frames=args.window_min_frames,
        p_fim=1.0,
        fim_format=args.fim_format,
        use_eos=args.use_eos,
        fim_min_gap=args.fim_min_gap,
        fim_max_gap=args.fim_max_gap,
        frame_guard_bytes=args.slice_header_guard_bytes,
        resample_fim=False,
        nal_index=nal_index,
        seed=args.seed,
    )


def analyze(args: argparse.Namespace) -> list[ContextWindowRow]:
    if args.context_frames < 0:
        raise ValueError("--context-frames must be non-negative")
    dataset = build_dataset(args)
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    rows: list[ContextWindowRow] = []
    markers = fim_marker_count(args.fim_format)
    eos_tokens = 1 if args.use_eos else 0

    for dataset_index in indices:
        sample = dataset.samples[dataset_index]
        data = sample.h264_path.read_bytes()
        window, _, _ = dataset._window_tensors(sample, data)
        window_bytes = bytes(window.tolist())
        frame_bounds = dataset._frame_bounds(sample, data)
        candidates = dataset._fim_candidates(sample, data)
        if not frame_bounds or not candidates:
            continue

        item_rng = dataset._rng_for(dataset_index)
        item_rng.random()  # mirrors __getitem__'s p_fim coin flip before hole sampling
        frame_lo, frame_hi = item_rng.choice(candidates)
        lo = frame_lo + args.slice_header_guard_bytes
        gap = item_rng.randint(args.fim_min_gap, min(args.fim_max_gap, frame_hi - lo - 1))
        split = item_rng.randint(lo, frame_hi - gap)

        damaged_frame_index = next(
            i for i, (lo_i, hi_i) in enumerate(frame_bounds) if lo_i == frame_lo and hi_i == frame_hi
        )
        first_frame_lo = frame_bounds[0][0]
        context_start_frame = max(0, damaged_frame_index - args.context_frames)
        semantic_context_lo = frame_bounds[context_start_frame][0]
        metadata_bytes = first_frame_lo if args.include_metadata else 0
        prior_context_bytes = frame_lo - semantic_context_lo
        total_context_bytes = metadata_bytes + prior_context_bytes

        damaged_frame_bytes = frame_hi - frame_lo
        prefix_bytes = split - frame_lo
        hole_bytes = gap
        orphan_bytes = frame_hi - (split + gap)
        prompt_tokens = total_context_bytes + prefix_bytes + orphan_bytes + markers
        target_tokens = hole_bytes + eos_tokens
        train_input_tokens = prompt_tokens + target_tokens - 1
        train_total_tokens = prompt_tokens + target_tokens
        rows.append(
            ContextWindowRow(
                sample_id=len(rows),
                dataset_index=dataset_index,
                h264_path=str(sample.h264_path),
                start_nal=sample.start_nal,
                end_nal=sample.end_nal,
                damaged_frame_index=damaged_frame_index,
                context_frames_requested=args.context_frames,
                context_frames_available=damaged_frame_index,
                context_frames_used=min(args.context_frames, damaged_frame_index),
                metadata_bytes=metadata_bytes,
                prior_context_bytes=prior_context_bytes,
                total_context_bytes=total_context_bytes,
                damaged_frame_bytes=damaged_frame_bytes,
                prefix_bytes=prefix_bytes,
                hole_bytes=hole_bytes,
                orphan_bytes=orphan_bytes,
                prompt_tokens=prompt_tokens,
                target_tokens=target_tokens,
                train_input_tokens=train_input_tokens,
                train_total_tokens=train_total_tokens,
                fits_block_size=train_input_tokens <= args.block_size,
                prompt_fraction_of_block=prompt_tokens / args.block_size,
                train_input_fraction_of_block=train_input_tokens / args.block_size,
                hole_fraction_of_damaged_frame=hole_bytes / max(damaged_frame_bytes, 1),
                prefix_fraction_of_damaged_frame=prefix_bytes / max(damaged_frame_bytes, 1),
                orphan_fraction_of_damaged_frame=orphan_bytes / max(damaged_frame_bytes, 1),
            )
        )
        if len(rows) >= args.num_samples:
            break

    if not rows:
        raise RuntimeError("No FIM-capable window samples found under the requested settings")
    return rows


def summarize(rows: list[ContextWindowRow], args: argparse.Namespace) -> dict[str, Any]:
    as_dicts = [r.__dict__ for r in rows]

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in as_dicts]

    overflow = [r for r in as_dicts if not r["fits_block_size"]]
    return {
        "manifest": str(args.manifest),
        "max_manifest_rows": args.max_manifest_rows,
        "num_samples": len(rows),
        "context_frames": args.context_frames,
        "include_metadata": args.include_metadata,
        "fim_format": args.fim_format,
        "use_eos": args.use_eos,
        "fim_min_gap": args.fim_min_gap,
        "fim_max_gap": args.fim_max_gap,
        "slice_header_guard_bytes": args.slice_header_guard_bytes,
        "max_window_bytes": args.max_window_bytes,
        "block_size": args.block_size,
        "fit_rate": 1.0 - (len(overflow) / len(rows)),
        "overflow_count": len(overflow),
        "context_frames_used": stats(col("context_frames_used")),
        "metadata_bytes": stats(col("metadata_bytes")),
        "prior_context_bytes": stats(col("prior_context_bytes")),
        "total_context_bytes": stats(col("total_context_bytes")),
        "damaged_frame_bytes": stats(col("damaged_frame_bytes")),
        "prefix_bytes": stats(col("prefix_bytes")),
        "hole_bytes": stats(col("hole_bytes")),
        "orphan_bytes": stats(col("orphan_bytes")),
        "prompt_tokens": stats(col("prompt_tokens")),
        "target_tokens": stats(col("target_tokens")),
        "train_input_tokens": stats(col("train_input_tokens")),
        "train_total_tokens": stats(col("train_total_tokens")),
        "hole_fraction_of_damaged_frame": stats(col("hole_fraction_of_damaged_frame")),
        "prefix_fraction_of_damaged_frame": stats(col("prefix_fraction_of_damaged_frame")),
        "orphan_fraction_of_damaged_frame": stats(col("orphan_fraction_of_damaged_frame")),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = analyze(args)
    summary = summarize(rows, args)

    csv_path = args.out_dir / "context_window_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(jsonable(row.__dict__))

    summary_path = args.out_dir / "context_window_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8")

    print(f"samples: {len(rows)}")
    print(f"context_frames: {args.context_frames}")
    print(f"fit_rate: {summary['fit_rate']:.3f} ({len(rows) - summary['overflow_count']}/{len(rows)})")
    print(
        "total_context_bytes: "
        f"mean={summary['total_context_bytes']['mean']:.1f} "
        f"p50={summary['total_context_bytes']['p50']:.1f} "
        f"p90={summary['total_context_bytes']['p90']:.1f} "
        f"max={summary['total_context_bytes']['max']:.1f}"
    )
    print(
        "train_input_tokens: "
        f"mean={summary['train_input_tokens']['mean']:.1f} "
        f"p50={summary['train_input_tokens']['p50']:.1f} "
        f"p90={summary['train_input_tokens']['p90']:.1f} "
        f"max={summary['train_input_tokens']['max']:.1f}"
    )
    print(f"-> {summary_path}")
    print(f"-> {csv_path}")


if __name__ == "__main__":
    main()
