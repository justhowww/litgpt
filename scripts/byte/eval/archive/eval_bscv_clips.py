"""Whole-clip BSCV-style H.264 corruption evaluation.

This evaluator is intentionally offline. It corrupts multiple VCL NALs per GOP,
decodes the whole stream end-to-end, and compares deleted-gap baselines against
model-filled spans. This is closer to the BSCV benchmark protocol than the
training-time reconstruction probe, which only measures one local missing span.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Running as ``python scripts/byte/eval/eval_bscv_clips.py`` puts the script's
# own directory on sys.path, not the repo root, so the ``scripts`` package below
# is not importable. Add the repo root (parents[3]) explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import Tensor

from litgpt.byte.data import (
    FIM_BEGIN_ID,
    FIM_END_ID,
    FIM_FORMATS,
    FIM_HOLE_ID,
    PARAMETER_SET_NAL_TYPES,
    REGION_BRIDGE,
    REGION_META,
    REGION_ORPHAN,
    REGION_PREFIX,
    REGION_REF,
    SPAN_BOS_ID,
    VCL_NAL_TYPES,
    NALUnit,
    bytes_to_ids,
    default_nal_index_path,
    load_manifest_rows,
    load_nal_index,
)
from litgpt.byte.reconstruction import (
    ReconstructionSample,
    image_psnr,
    image_ssim,
    parse_ppm,
)
from scripts.byte.eval.helpers.checkpoint_eval_helpers import generate_bytes, jsonable, load_model, save_png


@dataclass(frozen=True)
class CorruptionSpan:
    nal_index: int
    frame_index: int
    start: int
    end: int
    offset_in_nal: int
    length: int
    gop_distance: int


@dataclass(frozen=True)
class ClipExample:
    h264_path: Path
    nals: list[NALUnit]
    spans: list[CorruptionSpan]
    frame_gop_distances: list[int]
    # BSCV corrupt_Gen.py-faithful spans (IDR-eligible pool + small-frame fallback),
    # used for the side-by-side "BSCV FFmpeg default" reference column. Empty unless
    # --bscv-reference is set.
    bscv_spans: list[CorruptionSpan] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-manifest-rows", type=int, default=0)
    parser.add_argument("--num-clips", type=int, default=20)
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--viz-fps", type=int, default=6, help="Frame rate for the per-GOP comparison videos.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=16384)
    parser.add_argument("--num-ref-slices", type=int, default=1)
    parser.add_argument("--fim-format", choices=FIM_FORMATS, default="psm")
    parser.add_argument("--target-nal-types", type=int, nargs="+", default=[1], help="NAL types eligible for corruption. Default 1 matches current P-slice-only FIM training; add 5 only after training includes IDR/I-slice FIM.")
    parser.add_argument("--condition-on-sps-pps", action="store_true", default=True)
    parser.add_argument("--no-sps-pps-conditioning", dest="condition_on_sps_pps", action="store_false")
    parser.add_argument("--bscv-reference", action="store_true", default=True, help="Add a faithful BSCV corrupt_Gen.py corruption (IDR-eligible + small-frame fallback) decoded ffmpeg-default, as the 'BSCV FFmpeg default' column next to our 'deleted FFmpeg default'.")
    parser.add_argument("--no-bscv-reference", dest="bscv_reference", action="store_false")
    parser.add_argument("--corr-prob", type=int, default=1, help="Number of VCL frames corrupted per GOP, matching BSCV corr_prob semantics.")
    parser.add_argument("--corr-pos", type=float, default=0.4, help="Relative deletion position inside each selected VCL NAL payload.")
    parser.add_argument("--corr-len-bytes", type=int, default=2048, help="Max deleted payload bytes per selected frame, clamped to the frame's available payload. BSCV's corr_len is hex chars, so divide their value by two.")
    parser.add_argument("--min-corr-bytes", type=int, default=256, help="Minimum payload a frame must have to be eligible for corruption. Keeps the per-GOP pool ~uniform over frames (small P-frames stay eligible) so the IDR is not over-selected; the excision is clamped to the available payload.")
    parser.add_argument("--slice-header-guard-bytes", type=int, default=0, help="Optional extra guard after the NAL header before deletion. Use 0 for closest BSCV-style corruption.")
    parser.add_argument("--max-spans-per-clip", type=int, default=0, help="Optional cap on corrupted spans per clip after BSCV-style GOP sampling. 0 keeps all selected spans.")
    parser.add_argument("--gop-position-mode", choices=("random", "early", "late"), default="random", help="Which frames in each GOP to corrupt: random (uniform over eligible), early (earliest eligible — worst-case forward propagation), or late (latest eligible — minimal propagation before the next IDR).")
    parser.add_argument("--max-frames", type=int, default=96, help="Maximum decoded frames per clip for metric computation and memory control.")
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy byte generation; >0 enables sampling.")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.corr_prob < 1:
        raise ValueError("--corr-prob must be at least 1")
    if not 0.0 <= args.corr_pos <= 1.0:
        raise ValueError("--corr-pos must be in [0, 1]")
    if args.corr_len_bytes < 1:
        raise ValueError("--corr-len-bytes must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps(jsonable(vars(args)), indent=2) + "\n", encoding="utf-8"
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print("Loading manifest and NAL index...", flush=True)
    rows = load_manifest_rows(
        args.manifest,
        max_rows=None if args.max_manifest_rows == 0 else args.max_manifest_rows,
        report_progress=True,
    )
    index_path = args.nal_index_path or default_nal_index_path(args.manifest)
    nal_index = load_nal_index(index_path, args.manifest, rows)
    clips = select_clip_examples(rows, nal_index, args)
    if not clips:
        raise RuntimeError("No clips have eligible BSCV-style corruption spans")
    write_corruption_manifest(clips, args.out_dir / "corruptions.json")
    print(f"Selected {len(clips)} whole-clip examples", flush=True)

    summaries: list[dict[str, Any]] = []
    metrics_path = args.out_dir / "metrics.jsonl"
    details_path = args.out_dir / "clip_details.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    details_path.write_text("", encoding="utf-8")

    for checkpoint_dir in args.checkpoint_dirs:
        checkpoint_name = checkpoint_dir.name
        print(f"Evaluating {checkpoint_name} on {len(clips)} clips", flush=True)
        model = load_model(checkpoint_dir, device)
        summary, details, frames = evaluate_checkpoint(model, clips, args, device, checkpoint_name)
        summaries.append(summary)
        append_jsonl(metrics_path, [summary])
        append_jsonl(details_path, details)
        frame_dir = args.out_dir / "frames" / checkpoint_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        save_panels(frames, frame_dir, checkpoint_name)
        save_videos(frames, frame_dir, checkpoint_name, args)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps(summary, indent=2, allow_nan=True), flush=True)

    write_summary_csv(args.out_dir / "summary.csv", summaries)
    print(f"Wrote {args.out_dir}", flush=True)


def select_clip_examples(
    rows: list[dict[str, Any]],
    nal_index: dict[str, list[NALUnit]],
    args: argparse.Namespace,
) -> list[ClipExample]:
    rng = random.Random(args.seed)
    # Separate stream so enabling --bscv-reference never perturbs the main clip/span
    # draws — the two corruptions overlay on the exact same selected clips.
    bscv_rng = random.Random(args.seed + 9973)
    candidate_rows = list(rows)
    rng.shuffle(candidate_rows)
    examples: list[ClipExample] = []
    for row in candidate_rows:
        path = Path(row["h264_path"])
        nals = nal_index[str(path)]
        distances = compute_frame_gop_distances(nals)
        spans = select_bscv_spans(nals, distances, rng, args)
        if not spans:
            continue
        bscv_spans = bscv_reference_spans(nals, distances, bscv_rng, args) if args.bscv_reference else []
        examples.append(ClipExample(path, nals, spans, distances, bscv_spans))
        if len(examples) >= args.num_clips:
            break
    return examples


def compute_frame_gop_distances(nals: list[NALUnit]) -> list[int]:
    """Distance in decoded frames from the most recent IDR, one entry per VCL frame.

    Decode order equals display order because B-frames are disabled, so the i-th
    entry corresponds to the i-th decoded frame. The IDR frame itself is 0; each
    subsequent inter frame increments until the next IDR resets the reference
    chain. This is the propagation depth a concealment error inherits.
    """
    distances: list[int] = []
    distance = 0
    for nal in nals:
        if nal.nal_type not in VCL_NAL_TYPES:
            continue
        distance = 0 if nal.nal_type == 5 else distance + 1
        distances.append(distance)
    return distances


def select_bscv_spans(
    nals: list[NALUnit],
    distances: list[int],
    rng: random.Random,
    args: argparse.Namespace,
) -> list[CorruptionSpan]:
    target_nal_types = set(args.target_nal_types)
    vcl_indices = [idx for idx, nal in enumerate(nals) if nal.nal_type in VCL_NAL_TYPES]
    if not vcl_indices:
        return []

    frame_index_by_nal = {nal_idx: frame_idx for frame_idx, nal_idx in enumerate(vcl_indices)}
    gops: list[list[int]] = []
    current: list[int] = []
    for nal_idx in vcl_indices:
        if nals[nal_idx].nal_type == 5 and current:
            gops.append(current)
            current = []
        current.append(nal_idx)
    if current:
        gops.append(current)

    spans: list[CorruptionSpan] = []
    for gop in gops:
        # Gate by min_corr_bytes, NOT corr_len_bytes. Gating at the full corr_len
        # excludes most (small) P-frames while the always-large IDR stays eligible,
        # which inflates IDR selection to >50%. A small floor keeps the per-GOP pool
        # ~uniform over frames, so the IDR is hit at its true ~1/|GOP| rate (matching
        # BSCV); the excision is clamped to the available payload below.
        eligible = [
            nal_idx
            for nal_idx in gop
            if nals[nal_idx].nal_type in target_nal_types
            and frame_index_by_nal[nal_idx] < args.max_frames
            and eligible_payload_space(nals[nal_idx], args) >= args.min_corr_bytes
        ]
        if not eligible:
            continue
        k = min(args.corr_prob, len(eligible))
        if args.gop_position_mode == "early":
            selected = sorted(eligible, key=lambda nal_idx: frame_index_by_nal[nal_idx])[:k]
        elif args.gop_position_mode == "late":
            selected = sorted(eligible, key=lambda nal_idx: frame_index_by_nal[nal_idx], reverse=True)[:k]
        else:
            selected = rng.sample(eligible, k=k)
        for nal_idx in selected:
            nal = nals[nal_idx]
            payload_start = nal.payload_start + args.slice_header_guard_bytes
            # Clamp the excision to the available payload so small frames are still
            # corrupted (just by less) instead of being excluded.
            length = min(args.corr_len_bytes, nal.end - payload_start)
            if length < args.min_corr_bytes:
                continue
            max_start = nal.end - length
            start = payload_start + int((max_start - payload_start) * args.corr_pos)
            end = start + length
            frame_index = frame_index_by_nal[nal_idx]
            spans.append(
                CorruptionSpan(
                    nal_index=nal_idx,
                    frame_index=frame_index,
                    start=start,
                    end=end,
                    offset_in_nal=start - nal.start,
                    length=length,
                    gop_distance=distances[frame_index] if frame_index < len(distances) else -1,
                )
            )
    if args.max_spans_per_clip > 0 and len(spans) > args.max_spans_per_clip:
        spans = sorted(rng.sample(spans, k=args.max_spans_per_clip), key=lambda span: span.start)
    return spans


def eligible_payload_space(nal: NALUnit, args: argparse.Namespace) -> int:
    return max(0, nal.end - (nal.payload_start + args.slice_header_guard_bytes))


def bscv_reference_spans(
    nals: list[NALUnit],
    distances: list[int],
    rng: random.Random,
    args: argparse.Namespace,
) -> list[CorruptionSpan]:
    """Faithful port of BSCV's corrupt_Gen.py span selection
    (tests/eval/reference/bscv_corrupt_gen.py). It differs from select_bscv_spans in
    the two ways that actually change the corrupted bytes:

    1. the per-GOP sample pool is *every* VCL frame including the IDR (nal type 5) —
       BSCV draws from ``I + P + B`` uniformly, so the ~corr_prob/|GOP| keyframe-hit
       catastrophic tail is included rather than excluded;
    2. an undersized frame (payload <= corr_len) triggers BSCV's fallback — delete the
       payload *through the next start code*, wrecking NAL framing for the following
       frame — instead of being skipped.

    Placement and length otherwise match select_bscv_spans; ``corr_len_bytes`` is
    BSCV's hex ``corr_len`` halved (their offsets are hex chars, two per byte).

    The sample pool is restricted to ``target_nal_types`` so the column stays scoped
    to the same frames the eval corrupts: pass ``1`` for a P-only run (matching a
    P-only-trained checkpoint), ``1 5`` for the keyframe-inclusive BSCV-faithful run.
    """
    target_nal_types = set(args.target_nal_types)
    vcl_indices = [idx for idx, nal in enumerate(nals) if nal.nal_type in VCL_NAL_TYPES]
    if not vcl_indices:
        return []
    frame_index_by_nal = {nal_idx: frame_idx for frame_idx, nal_idx in enumerate(vcl_indices)}
    gops: list[list[int]] = []
    current: list[int] = []
    for nal_idx in vcl_indices:
        if nals[nal_idx].nal_type == 5 and current:
            gops.append(current)
            current = []
        current.append(nal_idx)
    if current:
        gops.append(current)

    corr_len = args.corr_len_bytes
    spans: list[CorruptionSpan] = []
    for gop in gops:
        pool = [
            nal_idx
            for nal_idx in gop
            if nals[nal_idx].nal_type in target_nal_types and frame_index_by_nal[nal_idx] < args.max_frames
        ]
        if not pool:
            continue
        k = min(args.corr_prob, len(pool))
        selected = rng.sample(pool, k=k)  # BSCV: uniform over the whole GOP, IDR included
        for nal_idx in selected:
            nal = nals[nal_idx]
            payload_start = nal.payload_start + args.slice_header_guard_bytes
            frame_index = frame_index_by_nal[nal_idx]
            if nal.end - payload_start > corr_len:
                max_start = nal.end - corr_len
                start = payload_start + int((max_start - payload_start) * args.corr_pos)
                end = start + corr_len
            else:
                # Fallback: excise through the first byte of the next start code so the
                # following frame's framing desyncs (delete_spans tolerates end > len).
                start = payload_start
                end = nal.end + 1
            spans.append(
                CorruptionSpan(
                    nal_index=nal_idx,
                    frame_index=frame_index,
                    start=start,
                    end=end,
                    offset_in_nal=start - nal.start,
                    length=end - start,
                    gop_distance=distances[frame_index] if frame_index < len(distances) else -1,
                )
            )
    return spans


def evaluate_checkpoint(
    model: torch.nn.Module,
    clips: list[ClipExample],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    method_names = ["deleted_default", "deleted_strict", "model_default", "model_strict"]
    if args.bscv_reference:
        method_names.append("bscv_default")
    accum = {name: empty_accumulator() for name in method_names}
    details: list[dict[str, Any]] = []
    frames: dict[int, dict[str, Any]] = {}

    for clip_idx, clip in enumerate(clips):
        print(f"  {checkpoint_name}: clip {clip_idx + 1}/{len(clips)}", flush=True)
        stream = clip.h264_path.read_bytes()
        clean_frames, clean_status = decode_stream_frames(stream, args, strict=False)
        if not clean_frames:
            details.append({"checkpoint": checkpoint_name, "clip_index": clip_idx, "status": "clean_decode_failed", "clean_status": clean_status})
            continue

        deleted_stream = delete_spans(stream, clip.spans)
        model_stream, generation_records = fill_spans_with_model(stream, clip, model, args, device)

        method_streams = {
            "deleted_default": (deleted_stream, False),
            "deleted_strict": (deleted_stream, True),
            "model_default": (model_stream, False),
            "model_strict": (model_stream, True),
        }
        if args.bscv_reference:
            method_streams["bscv_default"] = (delete_spans(stream, clip.bscv_spans), False)
        decoded: dict[str, list[Tensor]] = {}
        statuses: dict[str, str] = {"clean": clean_status}
        for method, (candidate_stream, strict) in method_streams.items():
            candidate_frames, status = decode_stream_frames(candidate_stream, args, strict=strict)
            decoded[method] = candidate_frames
            statuses[method] = status
            update_accumulator(accum[method], clean_frames, candidate_frames, status, clip.frame_gop_distances)

        details.append(
            {
                "checkpoint": checkpoint_name,
                "clip_index": clip_idx,
                "h264_path": str(clip.h264_path),
                "num_spans": len(clip.spans),
                "span_frames": [span.frame_index for span in clip.spans],
                "span_gop_distances": [span.gop_distance for span in clip.spans],
                "span_lengths": [span.length for span in clip.spans],
                "statuses": statuses,
                "generations": generation_records,
            }
        )
        if clip_idx < args.num_visualizations:
            frames[clip_idx] = build_visual_sequence(clean_frames, decoded, clip.spans, clip.frame_gop_distances)
            frames[clip_idx]["h264_path"] = str(clip.h264_path)

    attempted = len(clips)
    summary: dict[str, Any] = {"checkpoint": checkpoint_name, "num_clips": attempted}
    for method, values in accum.items():
        summary.update(flatten_accumulator(method, values, attempted))
    summary["corr_prob"] = args.corr_prob
    summary["corr_len_bytes"] = args.corr_len_bytes
    summary["corr_pos"] = args.corr_pos
    summary["gop_position_mode"] = args.gop_position_mode
    summary["target_nal_types"] = ",".join(str(t) for t in args.target_nal_types)
    return summary, details, frames


GOP_BUCKETS = ("idr", "near", "mid", "far", "unknown")

# image_psnr returns +inf for a byte-identical frame (MSE == 0), which is common
# for uncorrupted IDR frames. A single inf poisons every mean it lands in (sum/len
# -> inf). Clamp to a finite sentinel so perfect frames count as "very high PSNR"
# the same way SSIM counts them as 1.0, instead of nuking the aggregate.
PSNR_PERFECT_CAP = 100.0


def gop_distance_bucket(distance: int) -> str:
    """Bucket a frame by its propagation depth from the last IDR (GOP ~16)."""
    if distance < 0:
        return "unknown"
    if distance == 0:
        return "idr"
    if distance <= 3:
        return "near"
    if distance <= 9:
        return "mid"
    return "far"


def empty_accumulator() -> dict[str, Any]:
    return {
        "decoded": 0,
        "timeouts": 0,
        "no_frame": 0,
        "decoder_error": 0,
        "other_failures": 0,
        "frame_count_match": 0,
        "frame_coverage": [],
        "psnr": [],
        "ssim": [],
        "bucket_psnr": {bucket: [] for bucket in GOP_BUCKETS},
        "bucket_ssim": {bucket: [] for bucket in GOP_BUCKETS},
    }


def update_accumulator(
    accum: dict[str, Any],
    clean_frames: list[Tensor],
    candidate_frames: list[Tensor],
    status: str,
    frame_gop_distances: list[int],
) -> None:
    if status == "timeout":
        accum["timeouts"] += 1
        return
    elif not candidate_frames:
        if status in accum:
            accum[status] += 1
        else:
            accum["other_failures"] += 1
        return

    accum["decoded"] += 1
    if len(candidate_frames) == len(clean_frames):
        accum["frame_count_match"] += 1
    # Positional alignment is only trustworthy when frame counts match; PSNR for
    # mismatched-count clips is reported but should be read against
    # frame_count_match_rate.
    matched = 0
    limit = min(len(clean_frames), len(candidate_frames))
    for index in range(limit):
        reference = clean_frames[index]
        candidate = candidate_frames[index]
        if reference.shape != candidate.shape:
            continue
        matched += 1
        psnr = image_psnr(reference, candidate)
        if psnr == float("inf"):
            psnr = PSNR_PERFECT_CAP
        ssim = image_ssim(reference, candidate)
        accum["psnr"].append(psnr)
        accum["ssim"].append(ssim)
        distance = frame_gop_distances[index] if index < len(frame_gop_distances) else -1
        bucket = gop_distance_bucket(distance)
        accum["bucket_psnr"][bucket].append(psnr)
        accum["bucket_ssim"][bucket].append(ssim)
    accum["frame_coverage"].append(matched / max(len(clean_frames), 1))


def flatten_accumulator(prefix: str, values: dict[str, Any], attempted: int) -> dict[str, Any]:
    flat = {
        f"{prefix}_decode_rate": values["decoded"] / attempted if attempted else 0.0,
        f"{prefix}_frame_coverage_mean": mean(values["frame_coverage"]),
        f"{prefix}_frame_count_match_rate": values["frame_count_match"] / attempted if attempted else 0.0,
        f"{prefix}_psnr_mean": mean(values["psnr"]),
        f"{prefix}_ssim_mean": mean(values["ssim"]),
        f"{prefix}_timeouts": values["timeouts"],
        f"{prefix}_no_frame": values["no_frame"],
        f"{prefix}_decoder_error": values["decoder_error"],
        f"{prefix}_other_failures": values["other_failures"],
    }
    for bucket in GOP_BUCKETS:
        flat[f"{prefix}_gop_{bucket}_psnr_mean"] = mean(values["bucket_psnr"][bucket])
        flat[f"{prefix}_gop_{bucket}_ssim_mean"] = mean(values["bucket_ssim"][bucket])
        flat[f"{prefix}_gop_{bucket}_frames"] = len(values["bucket_psnr"][bucket])
    return flat


def fill_spans_with_model(
    stream: bytes,
    clip: ClipExample,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[bytes, list[dict[str, Any]]]:
    replacements: list[tuple[int, int, bytes]] = []
    records: list[dict[str, Any]] = []
    for span in clip.spans:
        sample = build_fim_sample(stream, clip.nals, span, clip.h264_path, args)
        if sample is None:
            replacement = bytes([0]) * span.length
            records.append({"frame_index": span.frame_index, "status": "prompt_too_long", "length": span.length})
        else:
            strategy = "greedy" if args.temperature <= 0 else "sample"
            candidates = generate_bytes(
                model,
                sample,
                device,
                strategy=strategy,
                num_candidates=1,
                temperature=max(args.temperature, 1e-8),
                top_k=args.top_k,
                top_p=args.top_p,
            )
            replacement = candidates[0] if candidates else bytes([0]) * span.length
            if len(replacement) != span.length:
                replacement = replacement[: span.length].ljust(span.length, b"\x00")
            records.append({"frame_index": span.frame_index, "status": "generated" if candidates else "generation_failed", "length": span.length})
        replacements.append((span.start, span.end, replacement))
    return replace_spans(stream, replacements), records


def build_fim_sample(
    stream: bytes,
    nals: list[NALUnit],
    span: CorruptionSpan,
    path: Path,
    args: argparse.Namespace,
) -> ReconstructionSample | None:
    target_nal = nals[span.nal_index]
    vcl_indices = [idx for idx, nal in enumerate(nals) if nal.nal_type in VCL_NAL_TYPES]
    target_vcl_pos = vcl_indices.index(span.nal_index)
    ref_indices = vcl_indices[max(0, target_vcl_pos - args.num_ref_slices) : target_vcl_pos]
    meta_indices = latest_parameter_sets(nals, span.nal_index) if args.condition_on_sps_pps else []

    meta = bytes_to_ids(b"".join(stream[nals[idx].start : nals[idx].end] for idx in meta_indices))
    ref_chunks = [bytes_to_ids(stream[nals[idx].start : nals[idx].end]) for idx in ref_indices]
    target = bytes_to_ids(stream[target_nal.start : target_nal.end])
    replacement_start = span.start - target_nal.start
    replacement_end = span.end - target_nal.start
    prefix = target[:replacement_start]
    orphan = target[replacement_end:]
    prompt_target_overhead = 3 if args.fim_format == "psm" else 1
    meta, ref = fit_conditioning(meta, ref_chunks, args.block_size - target.numel() - prompt_target_overhead)

    if args.fim_format == "psm":
        input_ids = torch.cat(
            (
                meta,
                ref,
                torch.tensor([FIM_BEGIN_ID], dtype=torch.long),
                prefix,
                torch.tensor([FIM_HOLE_ID], dtype=torch.long),
                orphan,
                torch.tensor([FIM_END_ID], dtype=torch.long),
            )
        )
        region_ids = torch.cat(
            (
                torch.full((meta.numel(),), REGION_META, dtype=torch.long),
                torch.full((ref.numel(),), REGION_REF, dtype=torch.long),
                torch.full((1 + prefix.numel(),), REGION_PREFIX, dtype=torch.long),
                torch.full((1 + orphan.numel(),), REGION_ORPHAN, dtype=torch.long),
                torch.full((1,), REGION_BRIDGE, dtype=torch.long),
            )
        )
        offset_ids = torch.cat(
            (
                torch.arange(meta.numel(), dtype=torch.long),
                torch.arange(ref.numel(), dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
                torch.arange(prefix.numel(), dtype=torch.long),
                torch.tensor([replacement_end], dtype=torch.long),
                torch.arange(replacement_end, target.numel(), dtype=torch.long),
                torch.tensor([replacement_start], dtype=torch.long),
            )
        )
    else:
        input_ids = torch.cat((meta, ref, prefix, orphan, torch.tensor([SPAN_BOS_ID], dtype=torch.long)))
        region_ids = torch.cat(
            (
                torch.full((meta.numel(),), REGION_META, dtype=torch.long),
                torch.full((ref.numel(),), REGION_REF, dtype=torch.long),
                torch.full((prefix.numel(),), REGION_PREFIX, dtype=torch.long),
                torch.full((orphan.numel(),), REGION_ORPHAN, dtype=torch.long),
                torch.full((1,), REGION_BRIDGE, dtype=torch.long),
            )
        )
        offset_ids = torch.cat(
            (
                torch.arange(meta.numel(), dtype=torch.long),
                torch.arange(ref.numel(), dtype=torch.long),
                torch.arange(prefix.numel(), dtype=torch.long),
                torch.arange(replacement_end, target.numel(), dtype=torch.long),
                torch.tensor([replacement_start], dtype=torch.long),
            )
        )
    if input_ids.numel() + span.length - 1 > args.block_size:
        return None
    return ReconstructionSample(
        h264_path=path,
        target_start=target_nal.start,
        target_end=target_nal.end,
        target_nal_index=span.nal_index,
        frame_index=span.frame_index,
        prompt_ids=input_ids,
        prompt_region_ids=region_ids,
        prompt_offset_ids=offset_ids,
        target_length=span.length,
        task="fim",
        replacement_start=replacement_start,
        replacement_end=replacement_end,
        generation_region_id=REGION_BRIDGE,
        generation_offset_start=replacement_start + 1,
    )


def latest_parameter_sets(nals: list[NALUnit], target_index: int) -> list[int]:
    latest: dict[int, int] = {}
    for idx, nal in enumerate(nals[:target_index]):
        if nal.nal_type in PARAMETER_SET_NAL_TYPES:
            latest[nal.nal_type] = idx
    return [latest[key] for key in sorted(latest)]


def fit_conditioning(meta: Tensor, ref_chunks: list[Tensor], budget: int) -> tuple[Tensor, Tensor]:
    if budget <= 0:
        return meta[:0], torch.empty(0, dtype=torch.long)
    if meta.numel() >= budget:
        return meta[-budget:], torch.empty(0, dtype=torch.long)
    remaining = budget - meta.numel()
    kept: list[Tensor] = []
    used = 0
    for chunk in reversed(ref_chunks):
        if used + chunk.numel() > remaining:
            continue
        kept.append(chunk)
        used += chunk.numel()
    kept.reverse()
    return meta, torch.cat(kept) if kept else torch.empty(0, dtype=torch.long)


def delete_spans(stream: bytes, spans: list[CorruptionSpan]) -> bytes:
    output = stream
    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        output = output[: span.start] + output[span.end :]
    return output


def replace_spans(stream: bytes, replacements: list[tuple[int, int, bytes]]) -> bytes:
    output = stream
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        output = output[:start] + replacement + output[end:]
    return output


def decode_stream_frames(stream: bytes, args: argparse.Namespace, *, strict: bool) -> tuple[list[Tensor], str]:
    command = [args.ffmpeg_binary, "-hide_banner", "-loglevel", "error"]
    if strict:
        command.extend(["-ec", "0", "-err_detect", "explode+bitstream+buffer+compliant"])
    command.extend(
        [
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vsync",
            "0",
            "-frames:v",
            str(args.max_frames),
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "pipe:1",
        ]
    )
    try:
        result = subprocess.run(command, input=stream, capture_output=True, timeout=args.timeout_sec)
    except FileNotFoundError:
        return [], "ffmpeg_not_found"
    except subprocess.TimeoutExpired:
        return [], "timeout"
    if strict and result.returncode != 0:
        return [], "decoder_error"
    if not result.stdout:
        return [], "no_frame"
    try:
        frames = parse_ppm_sequence(result.stdout)
    except ValueError:
        return [], "invalid_frame"
    return frames, "decoded" if frames else "no_frame"


def parse_ppm_sequence(data: bytes) -> list[Tensor]:
    frames: list[Tensor] = []
    cursor = 0
    while cursor < len(data):
        while cursor < len(data) and data[cursor : cursor + 1].isspace():
            cursor += 1
        if cursor >= len(data):
            break
        frame, next_cursor = parse_one_ppm(data, cursor)
        frames.append(frame)
        cursor = next_cursor
    return frames


def parse_one_ppm(data: bytes, start: int) -> tuple[Tensor, int]:
    tokens: list[bytes] = []
    cursor = start
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
    payload_len = width * height * 3
    pixels = data[cursor : cursor + payload_len]
    if len(pixels) != payload_len:
        raise ValueError("PPM pixel payload has the wrong size")
    frame = parse_ppm(data[start : cursor + payload_len])
    return frame, cursor + payload_len


VIZ_COLUMN_ORDER = (
    "ground_truth",
    "deleted_strict",
    "deleted_default",
    "bscv_default",
    "model_strict",
    "model_default",
)
VIZ_COLUMN_LABELS = {
    "ground_truth": "GT",
    "deleted_strict": "our deleted strict",
    "deleted_default": "our deleted FFmpeg default",
    "bscv_default": "BSCV FFmpeg default",
    "model_strict": "model strict",
    "model_default": "model FFmpeg default",
}


def visible_columns(seq: dict[str, Any]) -> list[str]:
    """Columns in display order that this sequence actually has frames for."""
    return [column for column in VIZ_COLUMN_ORDER if column in seq]


def gop_window(distances: list[int], frame_index: int, n_frames: int) -> tuple[int, int]:
    """[start, end) of the GOP containing ``frame_index`` — from its IDR (distance 0)
    to the frame just before the next IDR — clamped to the decoded frame count."""
    if not distances or n_frames <= 0:
        return 0, n_frames
    fi = min(frame_index, len(distances) - 1, n_frames - 1)
    start = fi
    while start > 0 and distances[start] != 0:
        start -= 1
    end = fi + 1
    while end < len(distances) and distances[end] != 0:
        end += 1
    return start, min(end, n_frames)


def build_visual_sequence(
    clean_frames: list[Tensor],
    decoded: dict[str, list[Tensor]],
    spans: list[CorruptionSpan],
    distances: list[int],
) -> dict[str, Any]:
    """Collect every frame of the GOP containing the first corruption, per method,
    so the cascade can be played as a side-by-side video. Missing/shape-mismatched
    frames (e.g. truncated strict decodes) become ``None`` -> red tile."""
    n = len(clean_frames)
    frame_idx = min(spans[0].frame_index if spans else 0, max(n - 1, 0))
    start, end = gop_window(distances, frame_idx, n)
    indices = list(range(start, end))
    seq: dict[str, Any] = {
        "frame_indices": indices,
        "corrupted_frames": sorted({s.frame_index for s in spans if start <= s.frame_index < end}),
        "gop_distances": [distances[i] if i < len(distances) else -1 for i in indices],
        "ground_truth": [clean_frames[i].cpu() for i in indices],
    }
    for method, mframes in decoded.items():
        seq[method] = [
            mframes[i].cpu() if i < len(mframes) and mframes[i].shape == clean_frames[i].shape else None
            for i in indices
        ]
    return seq


def composite_panel(seq: dict[str, Any], t: int) -> Tensor:
    """One side-by-side frame: GT | deleted strict | deleted default | model strict | model default."""
    reference = seq["ground_truth"][t]
    missing = torch.zeros_like(reference)
    missing[..., 0] = 1.0
    separator = torch.ones((reference.shape[0], 4, reference.shape[2]), dtype=reference.dtype)
    parts: list[Tensor] = []
    for col_idx, column in enumerate(visible_columns(seq)):
        if col_idx:
            parts.append(separator)
        frame = seq[column][t]
        parts.append(frame if isinstance(frame, Tensor) else missing)
    return torch.cat(parts, dim=1).clamp(0, 1)


def save_panels(sequences: dict[int, dict[str, Any]], frame_dir: Path, checkpoint_name: str) -> None:
    if sequences:
        print(f"Saving {len(sequences)} BSCV visual panels for {checkpoint_name}", flush=True)
    for clip_idx, seq in sequences.items():
        indices = seq["frame_indices"]
        if not indices:
            continue
        corrupted = seq.get("corrupted_frames") or []
        t = indices.index(corrupted[0]) if corrupted and corrupted[0] in indices else 0
        panel = composite_panel(seq, t)
        save_png(panel, frame_dir / f"clip_{clip_idx:04d}_panel.png")
        (frame_dir / f"clip_{clip_idx:04d}_panel.json").write_text(
            json.dumps(
                {
                    "checkpoint": checkpoint_name,
                    "clip_index": clip_idx,
                    "frame_index": indices[t],
                    "columns": [VIZ_COLUMN_LABELS[c] for c in visible_columns(seq)],
                    "red_tile": "decode failed or missing frame",
                    "h264_path": seq.get("h264_path"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def save_videos(
    sequences: dict[int, dict[str, Any]], frame_dir: Path, checkpoint_name: str, args: argparse.Namespace
) -> None:
    """Encode the whole-GOP side-by-side comparison as an mp4 per clip, so error
    propagation across the GOP is visible rather than a single corrupted frame."""
    if not sequences:
        return
    print(f"Saving {len(sequences)} BSCV GOP videos for {checkpoint_name}", flush=True)
    for clip_idx, seq in sequences.items():
        indices = seq["frame_indices"]
        if not indices:
            continue
        panels = [composite_panel(seq, t) for t in range(len(indices))]
        # rawvideo -> mpeg4 needs even dimensions; pad the bottom/right edge if odd.
        h, w = panels[0].shape[0], panels[0].shape[1]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            panels = [
                torch.nn.functional.pad(p.permute(2, 0, 1), (0, pad_w, 0, pad_h)).permute(1, 2, 0).contiguous()
                for p in panels
            ]
        height, width = panels[0].shape[0], panels[0].shape[1]
        buffer = b"".join(
            p.mul(255).round().clamp(0, 255).to(torch.uint8).contiguous().numpy().tobytes() for p in panels
        )
        out_path = frame_dir / f"clip_{clip_idx:04d}_gop.mp4"
        command = [
            args.ffmpeg_binary, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(args.viz_fps),
            "-i", "pipe:0", "-an", "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", str(out_path),
        ]
        try:
            subprocess.run(command, input=buffer, capture_output=True, check=True, timeout=args.timeout_sec)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            detail = exc.stderr.decode("utf-8", "ignore") if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else exc
            print(f"  GOP video encode failed for clip {clip_idx}: {detail}", flush=True)
            continue
        (frame_dir / f"clip_{clip_idx:04d}_gop.json").write_text(
            json.dumps(
                {
                    "checkpoint": checkpoint_name,
                    "clip_index": clip_idx,
                    "frame_indices": indices,
                    "corrupted_frames": seq.get("corrupted_frames"),
                    "gop_distances": seq.get("gop_distances"),
                    "fps": args.viz_fps,
                    "columns": [VIZ_COLUMN_LABELS[c] for c in visible_columns(seq)],
                    "red_tile": "decode failed or missing frame",
                    "h264_path": seq.get("h264_path"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def write_corruption_manifest(clips: list[ClipExample], path: Path) -> None:
    rows = []
    for clip_idx, clip in enumerate(clips):
        rows.append(
            {
                "clip_index": clip_idx,
                "h264_path": str(clip.h264_path),
                "spans": [span.__dict__ for span in clip.spans],
            }
        )
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
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


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


if __name__ == "__main__":
    main()
