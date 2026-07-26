"""Clean-prefix continuation probe for the multi-frame AR model (H0).

Verifies the AVC-LM/JPEG-LM generation legacy on *our* setup: give the model the
first N frames of a stream window as a clean prefix, free-run generate the next M
frames (byte-by-byte, external frame-count stop -- the AR model has no learned
EOS), strict-decode the result, and compare the generated continuation to the
ground-truth continuation.

H0 pass = decode-valid + plausible continuation; PSNR/SSIM vs GT is secondary
(the model invents a plausible-but-different future, so low PSNR is expected even
for good generation). The headline signal is the validity rate + the side-by-side
GT-vs-model continuation video.

Plausibility is quantified by a *distributional* score (the FVD/MAUVE analog for
generation, where there is no per-sample reference): a Fréchet distance between
the population of real continuation frames and the population of model-generated
ones, in a cheap, dependency-free feature space (per-frame appearance + frame-to-
frame motion). It separates "valid bytes" from "valid video": the two component
metrics, ``frechet_appearance`` and ``frechet_motion``, plus the interpretable
``motion_energy_*`` / ``grad_energy_*`` means, catch the frozen (motion ~ 0),
runaway (motion >> real), and blur (low gradient) failures that decode cleanly
but are implausible. The feature extractor is deliberately swappable: drop in an
I3D/Inception embedding here later for an FVD/FID comparable to published numbers.
Note this score is conditional on decode success -- it only sees frames that
decoded, so the validity rate above is what accounts for outright failures.

Usage:
    python scripts/byte/eval/eval_ar_continuation.py MANIFEST \
        --checkpoint-dirs RUN/step-XXXX [...] --out-dir OUT \
        --prefix-frames 8 --cont-frames 4 --num-clips 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte.data import (  # noqa: E402
    BYTE_VOCAB_SIZE,
    PARAMETER_SET_NAL_TYPES,
    REGION_META,
    REGION_TARGET,
    SLICE_BOS_ID,
    VCL_NAL_TYPES,
    NALUnit,
    bytes_to_ids,
    default_nal_index_path,
    load_manifest_rows,
    load_nal_index,
)
from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte import h264_mask as HM  # noqa: E402
from litgpt.byte.free_run_eval import (
    _survival_and_validity,
    free_run_rollout,
)  # noqa: E402
from litgpt.byte.reconstruction import image_psnr, image_ssim, parse_ppm  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import (
    jsonable,
    load_model,
)  # noqa: E402
from scripts.byte.eval.helpers.comparison_video import (  # noqa: E402
    save_comparison_video,
)
from scripts.byte.eval.helpers.clip_set import (  # noqa: E402
    load_clip_identifiers,
    match_manifest_rows,
)

PSNR_PERFECT_CAP = 100.0
START_CODE = (0, 0, 1)


@dataclass
class ContinuationClip:
    h264_path: Path
    prefix_end_nal: int  # exclusive: NALs [0:prefix_end_nal] form the clean prefix
    cont_end_nal: int  # exclusive: NALs [0:cont_end_nal] are the GT prefix+continuation
    prefix_frames: int
    cont_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-manifest-rows", type=int, default=0)
    parser.add_argument("--num-clips", type=int, default=20)
    parser.add_argument(
        "--clip-list",
        type=Path,
        default=None,
        help=(
            "Replay an explicit ordered video set instead of random selection. "
            "Accepts text/JSON paths or a prior clip_details.jsonl. Paths are "
            "matched by their suffix below h264/, so the same source clips can "
            "be evaluated across differently encoded corpus roots."
        ),
    )
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefix-frames",
        type=int,
        default=8,
        help="N clean frames given to the model.",
    )
    parser.add_argument(
        "--cont-frames", type=int, default=4, help="M frames the model must generate."
    )
    parser.add_argument(
        "--max-window-bytes",
        type=int,
        default=16384,
        help="Byte budget for prefix+continuation packing.",
    )
    parser.add_argument(
        "--max-gen-multiple",
        type=float,
        default=2.0,
        help="Cap generated bytes at this multiple of the GT continuation length.",
    )
    parser.add_argument(
        "--eval-intra",
        action="store_true",
        default=True,
        help="Also run intra (I-frame) generation: prompt = the parameter sets "
        "before an IDR, generate the IDR from scratch (the H.264 analog of JPEG-LM "
        "intra generation). Reported separately as mode=intra.",
    )
    parser.add_argument("--no-eval-intra", dest="eval_intra", action="store_false")
    parser.add_argument(
        "--eval-teacher-forced",
        action="store_true",
        default=True,
        help="Also run a teacher-forced pass over the continuation clips: one "
        "full-window forward with GT bytes always fed in, reporting per-byte top-1 "
        "accuracy and whether the argmax reconstruction decodes. Isolates local "
        "bit-fidelity from free-run error accumulation (mode=teacher_forced).",
    )
    parser.add_argument(
        "--no-eval-teacher-forced", dest="eval_teacher_forced", action="store_false"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Nucleus/top-k sampling: keep the k highest-prob bytes. AVC-LM uses 50 "
        "(with --temperature 1.0). 0 = off. Greedy (temp 0) degenerates for these AR "
        "byte-LMs; top-k+top-p is the AVC-LM-faithful setting.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top-p (nucleus) sampling: smallest byte set with cumulative prob >= p. "
        "AVC-LM uses 0.9 (with --temperature 1.0). 0 = off.",
    )
    parser.add_argument(
        "--stop-pad-run",
        type=int,
        default=0,
        help="Gave-up stop: end free-run when the model emits this many identical bytes "
        "in a row (the padding/rbsp_trailing attractor it falls into without a learned "
        "EOS), trimming the run. 0 = off. Try 32.",
    )
    parser.add_argument(
        "--mask-illegal-bytes",
        action="store_true",
        help="Constrained decoding for the supported phase-profile NAL header, "
        "slice header, slice data, picture progression, and Annex-B framing.",
    )
    parser.add_argument(
        "--slice-layout",
        choices=HM.SLICE_LAYOUTS,
        default=HM.SLICE_LAYOUT_MACROBLOCK,
        help=(
            "Slice extent assumed by constrained decoding and rollout stopping. "
            "macroblock preserves AVC-LM's one-MB slices; frame requires one complete "
            "progressive picture per slice and resolves its MB count from SPS."
        ),
    )
    parser.add_argument(
        "--mask-residual-only",
        action="store_true",
        help="With --mask-illegal-bytes, reproduce the legacy residual-only, "
        "fail-open mask for ablation. The default is the full profile mask.",
    )
    parser.add_argument(
        "--mask-debug",
        action="store_true",
        help="Print low-volume h264_mask diagnostics: state failures, legacy "
        "permissive fallbacks, and periodic mask summaries.",
    )
    parser.add_argument(
        "--mask-debug-stages",
        action="store_true",
        help="Verbose automaton trace showing syntax-element transitions such as "
        "mb_type, prediction, CBP, residual CAVLC fields, and RBSP trailing bits. "
        "Implies --mask-debug and can produce substantial output.",
    )
    parser.add_argument(
        "--survival-only",
        action="store_true",
        help="Continuation mode: skip the ffmpeg GT-decode + frame-count gate and the "
        "frame metrics (PSNR/Fréchet); report only byte/slice-level survival + desync "
        "(completed_bytes, desync_region/reason, slice-level success_rate). "
        "Needed for the per-MB (slice-max-mbs=1) corpus where 'frames' are macroblocks, and "
        "handy without ffmpeg.",
    )
    parser.add_argument(
        "--train-split-file",
        type=Path,
        default=None,
        help="Path to a train_split.json written by train.py. Restricts eval to the exact "
        "videos that were TRAINED (drops any held-out videos), giving an unambiguous "
        "training-set eval. Use the same --manifest/--max-manifest-rows as training.",
    )
    parser.add_argument(
        "--exclude-param-sets",
        action="store_true",
        help="Feed the model a window that STARTS at the first VCL NAL (IDR), dropping the "
        "leading SPS/PPS/SEI -- matching the training windows (which begin at the IDR). "
        "Parsing/metrics still use the full stream (SPS/PPS kept only for the parser), so "
        "correct/legal mb_type is measured with the model conditioned exactly as in training.",
    )
    parser.add_argument("--viz-fps", type=int, default=6)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mask_residual_only and not args.mask_illegal_bytes:
        raise SystemExit("--mask-residual-only requires --mask-illegal-bytes")
    HM.configure_debug(
        args.mask_debug or args.mask_debug_stages, stages=args.mask_debug_stages
    )
    if args.prefix_frames < 1 or args.cont_frames < 1:
        raise SystemExit("--prefix-frames and --cont-frames must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    rows = load_manifest_rows(
        args.manifest, max_rows=args.max_manifest_rows or None, report_progress=True
    )
    if args.train_split_file is not None:
        split = json.loads(Path(args.train_split_file).read_text(encoding="utf-8"))
        train_videos = {str(Path(v)) for v in split.get("videos", [])}
        before = len(rows)
        rows = [r for r in rows if str(Path(r["h264_path"])) in train_videos]
        print(
            f"train-split filter: {len(rows)}/{before} rows kept "
            f"({split.get('n_train_videos')} train videos from {args.train_split_file})",
            flush=True,
        )
        if not rows:
            raise SystemExit(
                "No manifest rows matched the train split -- check --manifest / "
                "--max-manifest-rows match the training run."
            )
    if args.clip_list is not None:
        try:
            identifiers = load_clip_identifiers(args.clip_list)
            rows = match_manifest_rows(rows, identifiers)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"Explicit clip-set selection failed: {exc}") from exc
        if len(rows) < args.num_clips:
            raise SystemExit(
                f"Clip list matched {len(rows)} videos but --num-clips requests "
                f"{args.num_clips}. Lower --num-clips or provide more clips."
            )
        rows = rows[: args.num_clips]
        print(
            f"explicit clip-set filter: selected {len(rows)} ordered videos from "
            f"{args.clip_list}",
            flush=True,
        )
    index_path = args.nal_index_path or default_nal_index_path(args.manifest)
    nal_index = load_nal_index(
        index_path, args.manifest, rows
    )  # dict[h264_path: str, list[NALUnit]]

    clips = select_continuation_clips(rows, nal_index, args)
    print(f"Selected {len(clips)} continuation clips", flush=True)
    if not clips:
        raise SystemExit(
            "No clips with enough frames; lower --prefix-frames/--cont-frames or raise --max-window-bytes"
        )
    if args.mask_illegal_bytes:
        _preflight_gt_mask(clips, nal_index, args.slice_layout)
    intra_clips = select_intra_clips(rows, nal_index, args) if args.eval_intra else []
    if args.eval_intra:
        print(f"Selected {len(intra_clips)} intra (I-frame) clips", flush=True)

    (args.out_dir / "config.json").write_text(
        json.dumps(jsonable(vars(args)), indent=2) + "\n", encoding="utf-8"
    )

    metrics_path = args.out_dir / "metrics.jsonl"
    details_path = args.out_dir / "clip_details.jsonl"
    summaries: list[dict[str, Any]] = []
    for checkpoint_dir in args.checkpoint_dirs:
        name = checkpoint_dir.name
        print(f"Loading checkpoint: {checkpoint_dir}", flush=True)
        model = load_model(checkpoint_dir, device)
        frame_dir = args.out_dir / "frames" / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        for mode, mode_clips in (("continuation", clips), ("intra", intra_clips)):
            if not mode_clips:
                continue
            summary, details, viz = evaluate_checkpoint(
                model, mode_clips, nal_index, args, device, name, mode
            )
            summaries.append(summary)
            save_continuation_videos(viz, frame_dir, name, args, tag=mode)
            append_jsonl(metrics_path, [summary])
            append_jsonl(details_path, details)
            print(json.dumps(summary, indent=2), flush=True)
        if args.eval_teacher_forced:
            tf_summary, tf_details = evaluate_teacher_forced(
                model, clips, nal_index, args, device, name
            )
            summaries.append(tf_summary)
            append_jsonl(metrics_path, [tf_summary])
            append_jsonl(details_path, tf_details)
            print(json.dumps(tf_summary, indent=2), flush=True)
        del model

    write_summary_csv(args.out_dir / "summary.csv", summaries)
    print(f"Continuation evaluation complete: {args.out_dir}", flush=True)


def select_continuation_clips(
    rows: list[dict[str, Any]],
    nal_index: dict[str, list[NALUnit]],
    args: argparse.Namespace,
) -> list[ContinuationClip]:
    import random

    rng = random.Random(args.seed)
    candidate_rows = list(rows)
    if args.clip_list is None:
        rng.shuffle(candidate_rows)
    needed = args.prefix_frames + args.cont_frames
    clips: list[ContinuationClip] = []
    for row in candidate_rows:
        path = Path(row["h264_path"])
        nals = nal_index[str(path)]
        data = path.read_bytes()
        clip = _first_qualifying_window(
            data, path, nals, needed, args.max_window_bytes, args.prefix_frames
        )
        if clip is not None:
            clips.append(clip)
        if len(clips) >= args.num_clips:
            break
    if args.clip_list is not None and len(clips) != len(candidate_rows):
        selected = {str(clip.h264_path) for clip in clips}
        ineligible = [
            str(row["h264_path"])
            for row in candidate_rows
            if str(Path(row["h264_path"])) not in selected
        ]
        raise SystemExit(
            "Explicit clip set must not silently substitute or drop videos. "
            f"{len(ineligible)} clip(s) cannot provide "
            f"{args.prefix_frames}+{args.cont_frames} frames within "
            f"--max-window-bytes={args.max_window_bytes}: {ineligible[:5]}. "
            "Raise --max-window-bytes or choose a different fixed set."
        )
    return clips


def _mask_audit_snapshot(state: HM.MaskState, byte_offset: int) -> dict[str, Any]:
    auto = state.automaton
    return {
        "byte_offset": byte_offset,
        "failure_reason": state.failure_reason,
        "nal_index": state.nal_index,
        "stage": getattr(auto, "stage", None),
        "syntax": getattr(auto, "ae_tag", None),
        "rbsp_bit": getattr(auto, "pos", None),
        "mbs_done": getattr(auto, "mbs_done", None),
        "slice_max_mbs": getattr(auto, "max_mbs", None),
    }


def audit_gt_continuation_mask(
    prefix_bytes: bytes, continuation_bytes: bytes, slice_layout: str
) -> dict[str, Any]:
    """Replay the exact generation boundary and prove the mask accepts GT.

    Seeding with ``advance`` mirrors free_run_rollout: the prefix is observed but not
    constrained, then every continuation byte is checked before commitment. This
    catches layout/parser disagreement before a model result can be blamed for it.
    """
    state = HM.MaskState(
        slice_max_mbs=HM.slice_max_mbs_for_layout(slice_layout),
        fail_closed=True,
    )
    for byte in prefix_bytes:
        HM.advance(state, byte)
    state.generation_started = True

    for offset, byte in enumerate(continuation_bytes):
        allowed = HM.get_valid_byte_mask(state)
        if not any(allowed):
            return {
                "ok": False,
                "reason": "no_allowed_byte",
                "gt_byte": byte,
                **_mask_audit_snapshot(state, offset),
            }
        if not allowed[byte]:
            return {
                "ok": False,
                "reason": "gt_byte_rejected",
                "gt_byte": byte,
                "allowed_count": sum(allowed),
                **_mask_audit_snapshot(state, offset),
            }
        HM.advance(state, byte)
        if state.automaton_unknown:
            return {
                "ok": False,
                "reason": "automaton_invalid_after_gt_byte",
                "gt_byte": byte,
                **_mask_audit_snapshot(state, offset),
            }

    auto = state.automaton
    if (
        not state.cur_is_vcl
        or auto is None
        or state.automaton_unknown
        or auto.stage != "done"
    ):
        return {
            "ok": False,
            "reason": "gt_continuation_ends_before_complete_slice",
            **_mask_audit_snapshot(state, len(continuation_bytes)),
        }
    return {
        "ok": True,
        "checked_bytes": len(continuation_bytes),
        "completed_mbs": auto.mbs_done,
        "slice_max_mbs": auto.max_mbs,
    }


def _preflight_gt_mask(
    clips: list[ContinuationClip],
    nal_index: dict[str, list[NALUnit]],
    slice_layout: str,
) -> None:
    started = time.perf_counter()
    checked_bytes = 0
    for clip_index, clip in enumerate(clips):
        data = clip.h264_path.read_bytes()
        nals = nal_index[str(clip.h264_path)]
        prefix = _concat_nals(data, nals, 0, clip.prefix_end_nal)
        continuation = _concat_nals(
            data, nals, clip.prefix_end_nal, clip.cont_end_nal
        )
        result = audit_gt_continuation_mask(prefix, continuation, slice_layout)
        if not result["ok"]:
            raise SystemExit(
                "GT syntax-mask preflight failed before model evaluation: "
                f"clip={clip_index} path={clip.h264_path} details={result}"
            )
        checked_bytes += int(result["checked_bytes"])
    elapsed = time.perf_counter() - started
    print(
        f"GT syntax-mask preflight: {len(clips)}/{len(clips)} accepted "
        f"(slice_layout={slice_layout}, bytes={checked_bytes}, seconds={elapsed:.1f})",
        flush=True,
    )


def _first_qualifying_window(
    data: bytes,
    path: Path,
    nals: list[NALUnit],
    needed_frames: int,
    max_bytes: int,
    prefix_frames: int,
) -> ContinuationClip | None:
    """First IDR-anchored window holding >= needed_frames REAL frames -- not raw
    VCL-NAL/MB-slice count -- within budget. Frame boundaries are ground truth
    first_mb_in_slice == 0 (HS.slice_first_mb), the SAME primitive free_run_rollout
    uses to decide when to stop generating; clip windowing and free-run stopping can
    therefore never silently disagree on what a 'frame' is, even on a
    slice-max-mbs=1 corpus where one VCL NAL is one macroblock, not one frame."""
    n = len(nals)
    for k in range(n):
        if nals[k].nal_type != 5:
            continue
        start = k
        # Back up over the access unit's leading non-VCL NALs (SPS/PPS + any SEI/AUD
        # between them and the IDR), stopping at the previous VCL slice -- matches
        # _windows_for_video so the eval window start aligns with training.
        while start - 1 >= 0 and nals[start - 1].nal_type not in VCL_NAL_TYPES:
            start -= 1
        total = 0
        frames = 0
        prefix_end = -1
        cont_end = -1
        end = start
        while end < n:
            nal = nals[end]
            nal_len = nal.end - nal.start
            if total + nal_len > max_bytes:
                break
            total += nal_len
            if nal.nal_type in VCL_NAL_TYPES:
                nal_bytes = data[nal.start + nal.start_code_len : nal.end]
                first_mb = HS.slice_first_mb(nal_bytes)
                if first_mb is None:
                    # Unparseable slice header (corrupt/truncated) -- don't build a
                    # window past it; try the next IDR candidate instead of guessing.
                    break
                if first_mb == 0:
                    frames += 1
                    if frames == prefix_frames + 1 and prefix_end < 0:
                        prefix_end = end  # exclude this (next-frame) NAL from prefix
                    if frames == needed_frames + 1:
                        cont_end = end  # exclude this (next-frame) NAL from cont
                        break
            end += 1
        if cont_end > 0 and prefix_end > 0:
            return ContinuationClip(
                path, prefix_end, cont_end, prefix_frames, needed_frames - prefix_frames
            )
    return None


def select_intra_clips(
    rows: list[dict[str, Any]],
    nal_index: dict[str, list[NALUnit]],
    args: argparse.Namespace,
) -> list[ContinuationClip]:
    """Intra (I-frame) generation: prompt = the parameter sets preceding the first
    IDR, model generates the IDR from scratch. prefix_frames=0, cont_frames=1."""
    import random

    rng = random.Random(args.seed + 1)
    candidate_rows = list(rows)
    rng.shuffle(candidate_rows)
    clips: list[ContinuationClip] = []
    for row in candidate_rows:
        path = Path(row["h264_path"])
        nals = nal_index[str(path)]
        idr = next((k for k, nal in enumerate(nals) if nal.nal_type == 5), None)
        if idr is None:
            continue
        start = idr
        # Back up over leading non-VCL NALs (SPS/PPS + any SEI/AUD before the IDR),
        # stopping at the previous VCL slice -- matches _windows_for_video.
        while start - 1 >= 0 and nals[start - 1].nal_type not in VCL_NAL_TYPES:
            start -= 1
        if not any(
            nals[i].nal_type in PARAMETER_SET_NAL_TYPES for i in range(start, idr)
        ):
            continue  # no SPS/PPS before the IDR -> cannot decode the generated frame
        psp = sum(nals[i].end - nals[i].start for i in range(start, idr))
        idr_len = nals[idr].end - nals[idr].start
        if psp + idr_len > args.max_window_bytes:
            continue
        clips.append(
            ContinuationClip(
                path,
                prefix_end_nal=idr,
                cont_end_nal=idr + 1,
                prefix_frames=0,
                cont_frames=1,
            )
        )
        if len(clips) >= args.num_clips:
            break
    return clips


def evaluate_checkpoint(
    model: torch.nn.Module,
    clips: list[ContinuationClip],
    nal_index: dict[str, list[NALUnit]],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_name: str,
    mode: str = "continuation",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    full_count = 0
    skipped_no_budget = 0
    skipped_gt_decode_short = 0
    cont_psnr: list[float] = []
    cont_ssim: list[float] = []
    frames_made: list[int] = []
    survivals: list[int] = []
    target_byte_counts: list[int] = []
    desync_regions: Counter = Counter()  # syntax element where free-run first desyncs
    desync_categories: Counter = Counter()  # its coarse syntax category
    desync_reasons: Counter = Counter()  # parser exception class -> failure mechanism
    rollout_stop_reasons: Counter = Counter()
    mask_failure_reasons: Counter = Counter()
    mask_calls_total = 0
    mask_strict_calls_total = 0
    mask_permissive_calls_total = 0
    mask_argmax_rejected_total = 0
    mask_probability_mass_weighted_sum = 0.0
    mask_probability_mass_count = 0
    first_mask_intervention_syntaxes: Counter = Counter()
    generation_seconds: list[float] = []
    generation_bytes_per_second: list[float] = []
    strict_decode_statuses: Counter = Counter()
    strict_decode_seconds: list[float] = []
    timeout_partial_frames: list[int] = []
    details: list[dict[str, Any]] = []
    viz: dict[int, dict[str, Any]] = {}
    # Distributional-plausibility feature populations (metric 3). Appearance is
    # per-frame; motion is per frame-to-frame transition (the n-th continuation
    # frame's transition is measured against frame n-1, i.e. the prefix seam).
    real_app: list[list[float]] = []
    gen_app: list[list[float]] = []
    real_mot: list[list[float]] = []
    gen_mot: list[list[float]] = []

    for clip_idx, clip in enumerate(clips):
        print(f"  {checkpoint_name}: clip {clip_idx + 1}/{len(clips)}", flush=True)
        data = clip.h264_path.read_bytes()
        nals = nal_index[str(clip.h264_path)]
        prefix_bytes = _concat_nals(data, nals, 0, clip.prefix_end_nal)
        gt_bytes = _concat_nals(data, nals, 0, clip.cont_end_nal)
        gt_cont_len = len(gt_bytes) - len(prefix_bytes)
        # --exclude-param-sets: feed the model a prefix that starts at the first VCL NAL
        # (IDR), matching training. Survival still parses with the FULL prefix (SPS/PPS
        # kept only for the parser), so the desync location is unaffected.
        vcl_start = (
            _leading_param_bytes(nals, clip.prefix_end_nal)[0]
            if args.exclude_param_sets
            else 0
        )
        fed_prefix = _concat_nals(data, nals, vcl_start, clip.prefix_end_nal)

        if not args.survival_only:
            # The continuation benchmark is strict end to end. Ground truth should
            # already be a valid H.264 stream, so a strict-decode failure is an eval
            # input problem rather than something error concealment should hide.
            gt_frames, _, gt_decode = decode_h264(
                gt_bytes,
                args,
                strict=True,
                max_frames=clip.prefix_frames + clip.cont_frames,
            )
            if len(gt_frames) < clip.prefix_frames + clip.cont_frames:
                skipped_gt_decode_short += 1
                details.append(
                    {
                        "checkpoint": checkpoint_name,
                        "mode": mode,
                        "clip_index": clip_idx,
                        "status": "gt_decode_short",
                        "ffmpeg_decode": gt_decode,
                    }
                )
                continue

        max_gen = min(
            int(gt_cont_len * args.max_gen_multiple) + 512,
            model_max_gen(model, fed_prefix),
        )
        if max_gen <= 0:
            # Prefix fills (or exceeds) the model context: there is no room to
            # generate, so model_bytes would equal the GT prefix and score a
            # spurious perfect 100. Exclude rather than contaminate the metric.
            skipped_no_budget += 1
            details.append(
                {
                    "checkpoint": checkpoint_name,
                    "mode": mode,
                    "clip_index": clip_idx,
                    "status": "skipped_no_budget",
                    "prefix_bytes": len(prefix_bytes),
                }
            )
            continue
        target_byte_counts.append(gt_cont_len)
        rollout_trace: dict[str, Any] = {}
        generation_started = time.perf_counter()
        gen_bytes, gen_frames_emitted = generate_continuation(
            model,
            fed_prefix,
            nals,
            clip.prefix_end_nal,
            device,
            args,
            clip.cont_frames,
            max_gen,
            start_nal=vcl_start,
            mask_prefix_bytes=prefix_bytes,
            trace=rollout_trace,
        )
        generation_elapsed = time.perf_counter() - generation_started
        generation_seconds.append(generation_elapsed)
        if generation_elapsed > 0:
            generation_bytes_per_second.append(len(gen_bytes) / generation_elapsed)
        rollout_stop_reasons[rollout_trace.get("stop_reason") or "none"] += 1
        if rollout_trace.get("mask_failure_reason"):
            mask_failure_reasons[rollout_trace["mask_failure_reason"]] += 1
        mask_calls_total += int(rollout_trace.get("mask_calls") or 0)
        mask_strict_calls_total += int(rollout_trace.get("mask_strict_calls") or 0)
        mask_permissive_calls_total += int(
            rollout_trace.get("mask_permissive_calls") or 0
        )
        rejected = int(rollout_trace.get("mask_argmax_rejected") or 0)
        mask_argmax_rejected_total += rejected
        mask_calls = int(rollout_trace.get("mask_probability_mass_count") or 0)
        probability_mass = rollout_trace.get("mask_allowed_probability_mass_mean")
        if probability_mass is not None and mask_calls:
            mask_probability_mass_weighted_sum += float(probability_mass) * mask_calls
            mask_probability_mass_count += mask_calls
        first_intervention = rollout_trace.get("first_mask_intervention")
        if first_intervention:
            first_mask_intervention_syntaxes[
                first_intervention.get("syntax") or "unknown"
            ] += 1
        n, m = clip.prefix_frames, clip.cont_frames

        # Persist both streams for offline analysis (scripts/byte/eval/analyze_nal_termination.py).
        stream_gen_path = _write_stream(
            args.out_dir,
            checkpoint_name,
            clip_idx,
            mode,
            "gen",
            prefix_bytes + gen_bytes,
        )
        stream_gt_path = _write_stream(
            args.out_dir, checkpoint_name, clip_idx, mode, "gt", gt_bytes
        )
        # stop_reason names which of free_run_rollout's six exits fired; it cannot be
        # recovered from the bytes (a clean frame-target stop, a pad-run stop, a mask
        # box-in and a first_mb desync all re-parse identically).
        stream_fields = {
            "stream_gen_path": stream_gen_path,
            "stream_gt_path": stream_gt_path,
            "n_prefix_bytes": len(prefix_bytes),
            "max_gen": max_gen,
            "stop_reason": rollout_trace.get("stop_reason"),
            "mask_failure_reason": rollout_trace.get("mask_failure_reason"),
            "mask_calls": rollout_trace.get("mask_calls"),
            "mask_strict_calls": rollout_trace.get("mask_strict_calls"),
            "mask_permissive_calls": rollout_trace.get("mask_permissive_calls"),
            "mask_argmax_rejected": rollout_trace.get("mask_argmax_rejected"),
            "mask_argmax_rejection_rate": rollout_trace.get(
                "mask_argmax_rejection_rate"
            ),
            "mask_allowed_probability_mass_mean": rollout_trace.get(
                "mask_allowed_probability_mass_mean"
            ),
            "mask_decisions_measured": rollout_trace.get("mask_probability_mass_count"),
            "first_mask_intervention": rollout_trace.get("first_mask_intervention"),
            "slice_layout": args.slice_layout,
            "generation_seconds": generation_elapsed,
            "generation_bytes_per_second": (
                len(gen_bytes) / generation_elapsed if generation_elapsed > 0 else None
            ),
        }

        # Byte-level survival + desync (parse-only, no ffmpeg) -- runs in BOTH modes.
        sr = _survival_and_validity(prefix_bytes, gen_bytes, m)
        survival = sr.survival
        survivals.append(survival)
        region = _INDEX_RE.sub("", sr.desync_region) if sr.desync_region else "none"
        desync_regions[region] += 1
        category = sr.desync_category or "none"
        desync_categories[category] += 1
        desync_reasons[sr.desync_reason or "none"] += 1

        if args.survival_only:
            # Slice-level validity only; skip the ffmpeg frame decode + PSNR/Fréchet.
            full_count += int(sr.valid_cont >= m)
            details.append(
                {
                    "checkpoint": checkpoint_name,
                    "mode": mode,
                    "clip_index": clip_idx,
                    "h264_path": str(clip.h264_path),
                    "completed_bytes": survival,
                    "valid_cont_slices": sr.valid_cont,
                    "start_codes_emitted": gen_frames_emitted,
                    "desync_region": region,
                    "desync_category": category,
                    "desync_reason": sr.desync_reason or "none",
                    "gen_bytes": len(gen_bytes),
                    "target_bytes": gt_cont_len,
                    "first_desync": sr.first_desync,
                    **stream_fields,
                }
            )
            continue

        # ---- frame-based path (needs ffmpeg): decode + PSNR/Fréchet ----
        model_bytes = prefix_bytes + gen_bytes
        model_frames, model_status, model_decode = decode_h264(
            model_bytes, args, strict=True, max_frames=n + m
        )
        strict_decode_statuses[model_status] += 1
        strict_decode_seconds.append(float(model_decode["elapsed_seconds"]))
        if model_status == "timeout":
            timeout_partial_frames.append(
                int(model_decode["complete_frames_before_exit"])
            )
        produced = max(0, len(model_frames) - n)
        frames_made.append(produced)
        strict_valid = model_status == "decoded" and len(model_frames) >= n + m
        full_count += int(strict_valid)

        clip_psnr: list[float] = []
        clip_ssim: list[float] = []
        for t in range(n, n + m):
            if (
                t < len(model_frames)
                and t < len(gt_frames)
                and model_frames[t].shape == gt_frames[t].shape
            ):
                p = image_psnr(gt_frames[t], model_frames[t])
                clip_psnr.append(PSNR_PERFECT_CAP if p == float("inf") else p)
                clip_ssim.append(image_ssim(gt_frames[t], model_frames[t]))
        cont_psnr.extend(clip_psnr)
        cont_ssim.extend(clip_ssim)

        _collect_distribution(gt_frames, n, m, real_app, real_mot)
        _collect_distribution(model_frames, n, m, gen_app, gen_mot)

        details.append(
            {
                "checkpoint": checkpoint_name,
                "mode": mode,
                "clip_index": clip_idx,
                "h264_path": str(clip.h264_path),
                "status": model_status,
                "completed_frames": produced,
                "target_frames": m,
                "start_codes_emitted": gen_frames_emitted,
                "strict_valid": strict_valid,
                "completed_bytes": survival,
                "desync_region": region,
                "desync_category": category,
                "desync_reason": sr.desync_reason or "none",
                "gen_bytes": len(gen_bytes),
                "target_bytes": gt_cont_len,
                "cont_psnr_mean": mean(clip_psnr),
                "cont_ssim_mean": mean(clip_ssim),
                "first_desync": sr.first_desync,
                "ffmpeg_decode": model_decode,
                **stream_fields,
            }
        )
        if clip_idx < args.num_visualizations:
            viz[clip_idx] = {
                "gt_frames": [f.cpu() for f in gt_frames[: n + m]],
                "model_frames": [f.cpu() for f in model_frames[: n + m]],
                "prefix_frames": n,
                "h264_path": str(clip.h264_path),
            }

    attempted = len(survivals)
    summary = {
        "checkpoint": checkpoint_name,
        "mode": mode,
        "num_clips": attempted,
        "num_skipped_no_budget": skipped_no_budget,
        "num_skipped_gt_decode_short": skipped_gt_decode_short,
        "success_rate": full_count / attempted if attempted else 0.0,
        "completed_frames_mean": mean(frames_made),
        "completed_frames_total": sum(frames_made),
        "completed_bytes_mean": mean([float(s) for s in survivals]),
        "completed_bytes_median": (
            float(sorted(survivals)[len(survivals) // 2]) if survivals else None
        ),
        "completed_bytes_max": float(max(survivals)) if survivals else None,
        "completed_bytes_total": sum(survivals),
        "target_frames": clips[0].cont_frames if clips else 0,
        "target_frames_total": attempted * clips[0].cont_frames if clips else 0,
        "target_bytes_mean": mean([float(n) for n in target_byte_counts]),
        "target_bytes_median": (
            float(sorted(target_byte_counts)[len(target_byte_counts) // 2])
            if target_byte_counts
            else None
        ),
        "target_bytes_max": (
            float(max(target_byte_counts)) if target_byte_counts else None
        ),
        "target_bytes_total": sum(target_byte_counts),
        # Where free-run generation first desyncs (syntax element), pooled over clips.
        "desync_region_top": (
            desync_regions.most_common(1)[0][0] if desync_regions else None
        ),
        "desync_region_hist": dict(desync_regions.most_common()),
        # Failure mechanism (parser exception class): ValueError = codeword/VLC-table
        # miss (the nC-governed layer), KeyError/IndexError = out-of-range value
        # (within-block constraint), BitReaderError = ran off the end.
        "desync_reason_hist": dict(desync_reasons.most_common()),
        "desync_category_hist": dict(desync_categories.most_common()),
        "generation_stop_reason_hist": dict(rollout_stop_reasons.most_common()),
        "generation_seconds_mean": mean(generation_seconds),
        "generation_seconds_max": (
            max(generation_seconds) if generation_seconds else None
        ),
        "generation_bytes_per_second_mean": mean(generation_bytes_per_second),
        "mask_failure_reason_hist": dict(mask_failure_reasons.most_common()),
        "mask_calls_total": mask_calls_total if args.mask_illegal_bytes else None,
        "mask_strict_calls_total": (
            mask_strict_calls_total if args.mask_illegal_bytes else None
        ),
        "mask_permissive_calls_total": (
            mask_permissive_calls_total if args.mask_illegal_bytes else None
        ),
        "mask_strict_call_rate": (
            mask_strict_calls_total / mask_calls_total if mask_calls_total else None
        ),
        "mask_argmax_rejected_total": (
            mask_argmax_rejected_total if args.mask_illegal_bytes else None
        ),
        "mask_decisions_measured_total": (
            mask_probability_mass_count if args.mask_illegal_bytes else None
        ),
        "mask_argmax_rejection_rate": (
            mask_argmax_rejected_total / mask_probability_mass_count
            if mask_probability_mass_count
            else None
        ),
        "mask_allowed_probability_mass_mean": (
            mask_probability_mass_weighted_sum / mask_probability_mass_count
            if mask_probability_mass_count
            else None
        ),
        "mask_first_intervention_syntax_hist": (
            dict(first_mask_intervention_syntaxes.most_common())
            if args.mask_illegal_bytes
            else None
        ),
        "strict_decode_status_hist": dict(strict_decode_statuses.most_common()),
        "strict_decode_timeout_count": strict_decode_statuses.get("timeout", 0),
        "strict_decode_seconds_mean": mean(strict_decode_seconds),
        "strict_decode_seconds_max": (
            max(strict_decode_seconds) if strict_decode_seconds else None
        ),
        "timeout_complete_frames_hist": dict(Counter(timeout_partial_frames)),
        "cont_psnr_mean": mean(cont_psnr),
        "cont_ssim_mean": mean(cont_ssim),
        "prefix_frames": clips[0].prefix_frames if clips else 0,
        "temperature": args.temperature,
        "slice_layout": args.slice_layout,
    }
    summary.update(distribution_metrics(real_app, gen_app, real_mot, gen_mot))
    return summary, details, viz


_INDEX_RE = re.compile(r"\[\d+\]")

# Residual block kind -> max_coeff. Selects the total_zeros VLC family (4 == chroma DC,
# which uses the _cdc_ tables) and marks when total_zeros is absent (TotalCoeff == max).
# Mirrors the _residual_block call sites in h264_syntax._parse_residual.
_MAX_COEFF = {"luma_dc": 16, "luma_ac": 15, "luma": 16, "chroma_dc": 4, "chroma_ac": 15}


def _acc(counts: list[int]) -> float | None:
    """Pooled accuracy hits/total, or None when the bucket is empty."""
    return counts[0] / counts[1] if counts[1] else None


def _syntax_buckets(
    spans: list, prefix_len: int, total_len: int
) -> tuple[list[str], list[str]]:
    """For each continuation byte offset in [prefix_len, total_len), return its H.264
    syntax category string and normalized element name (array indices stripped).

    Bytes not covered by any span -> "untagged" (GT is valid H.264, so ~0). A byte
    straddles bit-packed elements; it is assigned to the first span (stream order)
    that covers it -- a documented approximation adequate for a coarse *byte-level*
    signal. Now used only for the unigram histogram; the honest per-field accuracy is
    _value_legal_hits (argmax-decodes-to-correct-value / -to-legal-value).
    """
    n = total_len - prefix_len
    cat = ["untagged"] * n
    elem = ["untagged"] * n
    filled = [False] * n
    for span in spans:
        b0 = max(span.byte_start, prefix_len)
        b1 = min(span.byte_end, total_len)
        if b1 <= b0:
            continue
        cname = span.category.value
        ename = _INDEX_RE.sub("", span.name)
        for off in range(b0, b1):
            i = off - prefix_len
            if not filled[i]:
                cat[i] = cname
                elem[i] = ename
                filled[i] = True
    return cat, elem


def _merge_counts(dst: dict[str, list[int]], src: dict[str, list[int]]) -> None:
    for key, (hit, tot) in src.items():
        acc = dst.setdefault(key, [0, 0])
        acc[0] += hit
        acc[1] += tot


def _argmax_bit_reader(
    pred_list: list[int], prefix_len: int, byte_start: int, rbsp_bit0: int
):
    """A reader over the model's argmax RBSP bits, addressed by absolute RBSP bit index,
    for a field whose RBSP starts at ``rbsp_bit0`` / raw byte ``byte_start``. Maps RBSP
    bit -> raw byte (byte_start + (bit>>3 - rbsp_bit0>>3), valid when no emulation-
    prevention byte lies in the field, as the caller ensures) -> argmax byte in
    ``pred_list`` (indexed by raw_offset - prefix_len). Returns 0/1 or None past the end.
    """

    def bit(b: int):
        raw_off = byte_start + ((b >> 3) - (rbsp_bit0 >> 3))
        k = raw_off - prefix_len
        if 0 <= k < len(pred_list):
            return (pred_list[k] >> (7 - (b & 7))) & 1
        return None

    return bit


def _decode_ue_bits(bit, start: int, max_zeros: int = 32):
    """Decode an exp-Golomb ue(v) from the ``bit`` reader starting at RBSP bit ``start``.
    Returns the value, or None if it ran off the end / overran ``max_zeros``."""
    b, zeros = start, 0
    while True:
        v = bit(b)
        b += 1
        if v is None:
            return None
        if v == 1:
            break
        zeros += 1
        if zeros > max_zeros:
            return None
    val = 1
    for _ in range(zeros):
        v = bit(b)
        b += 1
        if v is None:
            return None
        val = (val << 1) | v
    return val - 1


def _vlc_legal(bit, start: int, label: str, max_len: int = 32) -> bool | None:
    """Is the model's argmax codeword at ``start`` legal for VLC table ``label``?

    True  -- the bits decode to a codeword.
    False -- provably illegal: no codeword can extend the accumulated prefix.
    None  -- INCONCLUSIVE: the eval window ended mid-codeword. Returning False here
             (as this used to) counts a truncated-but-still-extendable prefix as an
             illegal one, which is the same conflation decode_vlc had. Callers skip
             None, so such spans contribute to neither legal nor illegal counts.

    Mirrors h264_cavlc_tables.decode_vlc and h264_automaton._feed's ``vlc`` branch.
    """
    from litgpt.byte import h264_cavlc_tables as T  # lazy: keep import-light

    cmap, prefixes = T.code_map(label), T.prefix_set(label)
    s, b = "", start
    for _ in range(max_len):
        v = bit(b)
        if v is None:
            return None
        s += "1" if v else "0"
        b += 1
        if s in cmap:
            return True
        if s not in prefixes:
            return False
    return False


def _leading_param_bytes(nals: list[NALUnit], end_nal: int) -> tuple[int, int]:
    """(vcl_start, ps_len): index of the first VCL NAL in [0, end_nal) and the byte length
    of the leading non-VCL NALs (SPS/PPS/SEI) before it. (0, 0) if it starts at a VCL NAL."""
    vcl_start = next(
        (i for i in range(end_nal) if nals[i].nal_type in VCL_NAL_TYPES), 0
    )
    ps_len = sum(nals[i].end - nals[i].start for i in range(vcl_start))
    return vcl_start, ps_len


def _shift_spans(spans: list, ps_len: int) -> list:
    """Map parsed spans from full-window Annex-B byte coords into fed-window coords (the
    window with ps_len leading param-set bytes dropped): byte_start/byte_end move by
    -ps_len; bit_start/bit_end are per-NAL RBSP offsets, unchanged. Param-set spans (now
    before byte 0) are dropped. Used with --exclude-param-sets so the metric, computed in
    fed coords, aligns with the model that was fed the param-set-free window."""
    if ps_len == 0:
        return spans
    out = []
    for s in spans:
        if s.byte_start < ps_len:
            continue
        s.byte_start -= ps_len
        s.byte_end -= ps_len
        out.append(s)
    return out


def _value_legal_hits(
    spans: list,
    prefix_len: int,
    total_len: int,
    gt_list: list[int],
    pred_list: list[int],
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    """The honest per-field metrics from the teacher-forced argmax:

      correct[element] = fraction of that element's occurrences where the argmax bits
          EXACTLY reproduce GT's codeword. For a prefix-free code this is identical to
          "the argmax decodes to the CORRECT value" -- no byte co-packing (unlike a
          per-byte match) and no per-bit partial credit (unlike bit accuracy).
      legal[element] = fraction where the argmax codeword decodes to a legal value for
          the field's constraint under the GT parser state.
      illegal_when_different[element] = [n_illegal, n_different] over the same spans,
          i.e. P(illegal | different from GT). Every illegal prediction is necessarily
          different from the legal GT codeword.

    Legality is defined per field against the constraint the parser would enforce:
      mb_type (<=30 P / <=25 I), coded_block_pattern (codeNum < 48), coeff_token
      (nC-selected VLC table), mvd_l0.x/.y (se(v) decodes), total_zeros (TotalCoeff-
      selected table, _cdc_ family when max_coeff == 4), run_before (min(zerosLeft, 7)
      table). Context (nC, TotalCoeff, zerosLeft) is taken from GT spans -- this is a
      teacher-forced probe, so the question is "given the true state, is the model's own
      best guess legal here".

    Spans with an interior emulation-prevention byte (raw-byte count != rbsp-byte count)
    are skipped (the local RBSP<->raw mapping is unreliable). gt_list/pred_list index by
    k = raw_offset - prefix_len.
    """
    from litgpt.byte import h264_cavlc_tables as T

    correct: dict[str, list[int]] = {}
    legal: dict[str, list[int]] = {}
    illegal_when_different: dict[str, list[int]] = {}
    slice_type = None
    # GT residual-block context, carried across spans in stream order (coeff_token ->
    # total_zeros -> run_before[i] within one block).
    blk_tc: int | None = None
    blk_max: int | None = None
    zeros_left: int | None = None
    for span in spans:
        if span.name == "slice_type":
            slice_type = span.value
        b0, b1 = span.bit_start, span.bit_end
        if b1 <= b0:
            continue
        rb0 = b0 >> 3
        n_rbsp = ((b1 - 1) >> 3) - rb0 + 1
        if n_rbsp != (span.byte_end - span.byte_start):
            continue  # emulation-prevention inside the field; local mapping unreliable

        ename = _INDEX_RE.sub("", span.name)
        all_match, any_in = True, False
        for b in range(b0, b1):
            raw_off = span.byte_start + ((b >> 3) - rb0)
            if not (prefix_len <= raw_off < total_len):
                continue
            any_in = True
            k = raw_off - prefix_len
            shift = 7 - (b & 7)  # MSB-first
            if ((gt_list[k] >> shift) & 1) != ((pred_list[k] >> shift) & 1):
                all_match = False
        if not any_in:
            continue
        c = correct.setdefault(ename, [0, 0])
        c[0] += int(all_match)
        c[1] += 1

        bit = _argmax_bit_reader(pred_list, prefix_len, span.byte_start, b0)
        base = ename.split(".")[0]
        is_legal = None
        if span.name == "mb_type":
            v = _decode_ue_bits(bit, b0)
            if v is not None and slice_type is not None:
                hi = (
                    25 if (slice_type % 5 == 2) else 30
                )  # I-slice caps at 25, else P (30)
                is_legal = 0 <= v <= hi
        elif span.name == "coded_block_pattern":
            v = _decode_ue_bits(bit, b0)
            if v is not None:
                is_legal = 0 <= v < 48  # GOLOMB_TO_*_CBP have 48 entries
        elif ename.endswith(".coeff_token"):
            nc = span.value.get("nC") if isinstance(span.value, dict) else None
            if nc is not None:
                is_legal = _vlc_legal(bit, b0, T.coeff_token_label(nc))
        elif ename in ("mvd_l0.x", "mvd_l0.y"):
            # se(v) is unbounded; the only illegality the parser can hit is an
            # exp-Golomb prefix that never terminates (read_ue caps leading zeros).
            is_legal = _decode_ue_bits(bit, b0) is not None
        elif ename.endswith(".total_zeros"):
            if blk_tc is not None and blk_max is not None and 0 < blk_tc < blk_max:
                label = (
                    f"total_zeros_cdc_{blk_tc}"
                    if blk_max == 4
                    else f"total_zeros_4x4_{blk_tc}"
                )
                is_legal = _vlc_legal(bit, b0, label)
        elif ename.endswith(".run_before"):
            if zeros_left is not None and zeros_left > 0:
                is_legal = _vlc_legal(bit, b0, f"run_before_{min(zeros_left, 7)}")
        if is_legal is not None:
            lg = legal.setdefault(ename, [0, 0])
            lg[0] += int(is_legal)
            lg[1] += 1
            if not all_match:  # condition on predictions different from GT
                iw = illegal_when_different.setdefault(ename, [0, 0])
                iw[0] += int(not is_legal)
                iw[1] += 1

        # Advance GT block context AFTER the check above (the check reads the state the
        # parser would be in when it reaches this field).
        if ename.endswith(".coeff_token"):
            blk_tc = (
                span.value.get("total_coeff") if isinstance(span.value, dict) else None
            )
            blk_max = _MAX_COEFF.get(base)
            # total_zeros is only coded when TotalCoeff < max_coeff; otherwise zerosLeft
            # is 0 and no run_before follows.
            zeros_left = 0 if (blk_tc is not None and blk_tc == blk_max) else None
        elif ename.endswith(".total_zeros") and isinstance(span.value, int):
            zeros_left = span.value
        elif (
            ename.endswith(".run_before")
            and isinstance(span.value, int)
            and zeros_left is not None
        ):
            zeros_left -= span.value
    return correct, legal, illegal_when_different


def _copy_oracle(spans: list) -> dict[str, list[int]]:
    """GT-only temporal-persistence oracle: for each per-MB element, the fraction of
    macroblocks whose value equals the same mb_addr's value in the *previous* frame.

    Frame boundary = a first_mb_in_slice span with value 0 (holds for one-slice-per-
    frame and slice-max-mbs=1). Returns {element_name: [matches, comparisons]}.
    This measures how much redundancy EXISTS in the data -- the ceiling a model that
    merely copied the previous frame would hit -- independent of the model.
    """
    frames: list[dict[int, dict[str, object]]] = []
    cur: dict[int, dict[str, object]] | None = None
    for span in spans:
        if span.name == "first_mb_in_slice" and span.value == 0:
            if cur is not None:
                frames.append(cur)
            cur = {}
        if cur is None or span.mb_addr is None:
            continue
        cur.setdefault(span.mb_addr, {})[_INDEX_RE.sub("", span.name)] = span.value
    if cur is not None:
        frames.append(cur)

    out: dict[str, list[int]] = {}
    for f in range(1, len(frames)):
        prev_frame = frames[f - 1]
        for mb, elems in frames[f].items():
            prev = prev_frame.get(mb)
            if not prev:
                continue
            for ename, val in elems.items():
                if ename in prev:
                    c = out.setdefault(ename, [0, 0])
                    c[0] += int(val == prev[ename])
                    c[1] += 1
    return out


@torch.inference_mode()
def evaluate_teacher_forced(
    model: torch.nn.Module,
    clips: list[ContinuationClip],
    nal_index: dict[str, list[NALUnit]],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Teacher-forced continuation probe on the same clips as mode=continuation.

    One full-window forward with the GROUND-TRUTH bytes always fed in (no
    autoregression), then per continuation byte compare argmax(logits) to the GT
    byte. Readings:
      * ``tf_byte_acc`` / ``tf_ce_nats`` -- overall local bit-fidelity (and a
        cross-check that the forward is faithful: tf_ce tracks training val_loss_ar).
      * ``element_correct`` / ``syntax_legal_by_element``
        -- per H.264 syntax element, whether the argmax codeword decodes to the CORRECT
        value (exact codeword match) and to a LEGAL value (would not desync). These are
        the honest per-field numbers: bit-accuracy overstates (per-bit partial credit on
        near-deterministic fields), byte-accuracy understates (co-packed neighbours), and
        neither answers "would the model's own best guess produce a legal value here" --
        which ``syntax_legal_by_element`` does.
    """
    raw = model.module if hasattr(model, "module") else model  # * model
    raw.eval()
    max_seq = int(raw.max_seq_length)

    details: list[dict[str, Any]] = []
    byte_accs: list[float] = []
    ces: list[float] = []
    # Pooled (across-clip, byte-weighted) per-syntax accuracy: bucket -> [hits, total].
    # Honest per-field metrics: does the model's argmax codeword decode to the CORRECT
    # value (exact codeword match), and to a LEGAL value (would not desync)?
    elem_correct: dict[str, list[int]] = {}
    elem_legal: dict[str, list[int]] = {}
    elem_illegal_different: dict[str, list[int]] = {}
    # Unigram/chance floor: GT byte histogram per bucket (accuracy of always guessing
    # the bucket's most common byte).
    cat_hist: dict[str, Counter] = {}
    elem_hist: dict[str, Counter] = {}
    # Temporal-copy oracle (GT-only): per-element previous-frame persistence.
    copy_acc: dict[str, list[int]] = {}
    skipped = 0
    attempted = 0

    for clip_idx, clip in enumerate(clips):
        with open(clip.h264_path, "rb") as handle:
            data = handle.read()
        nals = nal_index[str(clip.h264_path)]
        prefix_bytes = _concat_nals(data, nals, 0, clip.prefix_end_nal)
        gt_bytes = _concat_nals(data, nals, 0, clip.cont_end_nal)
        prefix_len = len(prefix_bytes)
        total_len = len(gt_bytes)
        if total_len - prefix_len <= 0:
            continue
        # --exclude-param-sets: feed a window starting at the first VCL NAL (IDR), like the
        # training windows. The full window is still parsed below (SPS/PPS kept for the
        # parser); spans are shifted into fed coords. ps_len = leading param-set bytes.
        vcl_start, ps_len = (
            _leading_param_bytes(nals, clip.cont_end_nal)
            if args.exclude_param_sets
            else (0, 0)
        )
        fed_bytes = gt_bytes[ps_len:]
        fed_prefix_len = prefix_len - ps_len
        fed_total_len = total_len - ps_len
        # One BOS + all fed bytes must fit the context window.
        if fed_total_len + 1 > max_seq:
            skipped += 1
            details.append(
                {
                    "checkpoint": checkpoint_name,
                    "clip_index": clip_idx,
                    "status": "skipped_too_long",
                    "window_bytes": fed_total_len,
                }
            )
            continue
        attempted += 1
        # * preparing for inputs and auxiliary inputs (region_ids, offset_ids)
        prompt_ids, region_ids, offset_ids = _prompt_tensors(
            fed_bytes, nals, clip.cont_end_nal, start_nal=vcl_start
        )
        prompt_ids = prompt_ids.to(device).unsqueeze(0)
        region_ids = region_ids.to(device).unsqueeze(0)
        offset_ids = offset_ids.to(device).unsqueeze(0)
        seq_len = prompt_ids.size(1)  # BOS + total_len

        cache_dtype = (
            torch.bfloat16 if device.type == "cuda" else next(raw.parameters()).dtype
        )
        raw.set_kv_cache(
            batch_size=1, max_seq_length=seq_len, device=device, dtype=cache_dtype
        )
        try:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = raw(
                    prompt_ids,
                    input_pos=torch.arange(seq_len, device=device),
                    input_pos_maxp1=seq_len,
                    region_ids=region_ids,
                    offset_ids=offset_ids,
                )
        finally:
            raw.clear_kv_cache()

        # token = [BOS, gt_0, ..., gt_{L-1}]; logits[:, j] predicts token[j+1] = gt_j.
        # Continuation bytes are gt_j for j in [prefix_len, total_len-1].
        gt_ids = bytes_to_ids(fed_bytes).to(device)
        cont_logits = logits[0, fed_prefix_len:fed_total_len, :BYTE_VOCAB_SIZE].float()
        cont_targets = gt_ids[fed_prefix_len:fed_total_len]
        pred_ids = cont_logits.argmax(dim=-1)
        acc = float((pred_ids == cont_targets).float().mean())
        ce = float(torch.nn.functional.cross_entropy(cont_logits, cont_targets))
        byte_accs.append(acc)
        ces.append(ce)

        # Parse the FULL GT window once (SPS/PPS needed to resolve pps_id), then shift the
        # spans into fed coords so they align with the (param-set-free) window the model saw.
        spans = HS.parse_stream(gt_bytes, parse_slice_data=True).all_spans()
        spans = _shift_spans(spans, ps_len)
        gt_list = cont_targets.tolist()
        pred_list = pred_ids.tolist()

        # Unigram histogram per bucket (chance floor).
        cat_of, elem_of = _syntax_buckets(spans, fed_prefix_len, fed_total_len)
        for k in range(len(gt_list)):
            cat_hist.setdefault(cat_of[k], Counter())[gt_list[k]] += 1
            elem_hist.setdefault(elem_of[k], Counter())[gt_list[k]] += 1

        # Honest per-field metrics (correct-value / legal-value) + temporal-copy oracle.
        cor, leg, ilw = _value_legal_hits(
            spans, fed_prefix_len, fed_total_len, gt_list, pred_list
        )
        _merge_counts(elem_correct, cor)
        _merge_counts(elem_legal, leg)
        _merge_counts(elem_illegal_different, ilw)
        _merge_counts(copy_acc, _copy_oracle(spans))

        details.append(
            {
                "checkpoint": checkpoint_name,
                "clip_index": clip_idx,
                "h264_path": str(clip.h264_path),
                "cont_bytes": total_len - prefix_len,
                "tf_byte_acc": acc,
                "tf_ce_nats": ce,
            }
        )

    def _uni(counter: Counter) -> float | None:
        tot = sum(counter.values())
        return max(counter.values()) / tot if tot else None

    summary = {
        "checkpoint": checkpoint_name,
        "mode": "teacher_forced",
        "num_clips": attempted,
        "num_skipped_too_long": skipped,
        "tf_byte_acc_mean": mean(byte_accs),
        "tf_ce_nats_mean": mean(ces),
        # Unigram/chance floor: accuracy of always guessing the bucket's most common byte.
        "unigram_mb_type": _uni(elem_hist.get("mb_type", Counter())),
        "unigram_mb_pred": _uni(cat_hist.get("mb_pred", Counter())),
        "unigram_residual": _uni(
            cat_hist.get("residual_luma", Counter())
            + cat_hist.get("residual_chroma", Counter())
        ),
        # Temporal-copy oracle (GT-only): does the field persist frame-to-frame?
        "copy_oracle_mb_type": _acc(copy_acc.get("mb_type", [0, 0])),
        # Full per-element breakdowns.
        "element_correct": {
            e: {"acc": h / t, "n": t}
            for e, (h, t) in sorted(elem_correct.items())
            if t >= 16
        },
        "syntax_legal_by_element": {
            e: {"legal_rate": h / t, "count": t}
            for e, (h, t) in sorted(elem_legal.items())
            if t >= 16
        },
        # P(illegal | different from GT) per element. Deliberately not count-filtered:
        # the denominator is the ERROR count, which is ~1e-3 of the occurrence count, so
        # a threshold would empty this dict. Raw counts keep uncertainty visible.
        "syntax_illegal_when_different_by_element": {
            e: {"illegal_rate": h / t, "illegal_count": h, "different_count": t,}
            for e, (h, t) in sorted(elem_illegal_different.items())
            if t > 0
        },
        "unigram_acc": {
            e: {"acc": _uni(c), "n": sum(c.values())}
            for e, c in sorted(elem_hist.items())
            if sum(c.values()) >= 32
        },
        "copy_oracle": {
            e: {"acc": h / t, "n": t}
            for e, (h, t) in sorted(copy_acc.items())
            if t >= 16
        },
        "prefix_frames": clips[0].prefix_frames if clips else 0,
        "cont_frames": clips[0].cont_frames if clips else 0,
        "temperature": 0.0,
    }
    return summary, details


def model_max_gen(model: torch.nn.Module, prefix_bytes: bytes) -> int:
    raw = model.module if hasattr(model, "module") else model
    return max(0, int(raw.max_seq_length) - len(prefix_bytes) - 1)


@torch.inference_mode()
def generate_continuation(
    model: torch.nn.Module,
    prefix_bytes: bytes,
    nals: list[NALUnit],
    prefix_end_nal: int,
    device: torch.device,
    args: argparse.Namespace,
    cont_frames: int,
    max_gen: int,
    start_nal: int = 0,
    mask_prefix_bytes: bytes | None = None,
    trace: dict | None = None,
) -> tuple[bytes, int]:
    """Free-run byte generation, stopping after cont_frames complete real frames
    (detected via first_mb_in_slice == 0 on each closed VCL NAL -- see
    HS.slice_first_mb / free_run_rollout, NOT a raw NAL/start-code count) or the byte
    budget. Offsets reset per NAL by detecting start codes; region is REGION_TARGET
    for all generated bytes. ``start_nal`` > 0 (--exclude-param-sets) means
    ``prefix_bytes`` = concat of nals[start_nal:end]. ``mask_prefix_bytes`` retains
    the full SPS/PPS-bearing stream for the syntax mask while the model sees the
    parameter-set-free prompt.
    """
    raw = model.module if hasattr(model, "module") else model
    raw.eval()

    prompt_ids, prompt_region, prompt_offset = _prompt_tensors(
        prefix_bytes, nals, prefix_end_nal, start_nal=start_nal
    )
    prompt_ids = prompt_ids.to(device).unsqueeze(0)
    prompt_region = prompt_region.to(device).unsqueeze(0)
    prompt_offset = prompt_offset.to(device).unsqueeze(0)
    # Shared rollout with the in-loop val_freerun probe -- do not reimplement here.
    return free_run_rollout(
        raw,
        prompt_ids,
        prompt_region,
        prompt_offset,
        device,
        cont_frames,
        max_gen,
        args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_pad_run=args.stop_pad_run,
        constrain=args.mask_illegal_bytes,
        mask_residual_only=args.mask_residual_only,
        prefix_bytes=mask_prefix_bytes or prefix_bytes,
        trace=trace,
        slice_layout=args.slice_layout,
    )


def _prompt_tensors(
    prompt_bytes: bytes, nals: list[NALUnit], end_nal: int, start_nal: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """Build (prompt_ids, region_ids, offset_ids) for ``prompt_bytes`` = concat of
    nals[start_nal:end_nal]. ``start_nal`` > 0 drops leading NALs (e.g. SPS/PPS/SEI) to
    match training windows that begin at the IDR -- offset_ids are per-NAL either way."""
    region_chunks: list[Tensor] = []
    offset_chunks: list[Tensor] = []
    for nal in nals[start_nal:end_nal]:
        length = nal.end - nal.start
        region = (
            REGION_META if nal.nal_type in PARAMETER_SET_NAL_TYPES else REGION_TARGET
        )
        region_chunks.append(torch.full((length,), region, dtype=torch.long))
        offset_chunks.append(torch.arange(length, dtype=torch.long))
    raw_region = torch.cat(region_chunks)
    raw_offset = torch.cat(offset_chunks)
    bos = torch.tensor([SLICE_BOS_ID], dtype=torch.long)
    prompt_ids = torch.cat((bos, bytes_to_ids(prompt_bytes)))
    region_ids = torch.cat(
        (torch.tensor([REGION_TARGET], dtype=torch.long), raw_region)
    )
    offset_ids = torch.cat((torch.tensor([0], dtype=torch.long), raw_offset))
    return prompt_ids, region_ids, offset_ids


def _concat_nals(data: bytes, nals: list[NALUnit], start: int, end: int) -> bytes:
    return b"".join(data[nals[i].start : nals[i].end] for i in range(start, end))


def decode_h264(
    stream: bytes,
    args: argparse.Namespace,
    *,
    strict: bool,
    max_frames: int | None = None,
) -> tuple[list[Tensor], str, dict[str, Any]]:
    command = [args.ffmpeg_binary, "-hide_banner", "-loglevel", "error"]
    if strict:
        command.extend(
            ["-ec", "0", "-err_detect", "explode+bitstream+buffer+compliant"]
        )
    command.extend(
        [
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vsync",
            "0",
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
        ]
    )
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append("pipe:1")
    started = time.monotonic()

    def diagnostics(
        *,
        status: str,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int | None = None,
    ) -> dict[str, Any]:
        complete_frames = len(parse_ppm_sequence(stdout)) if stdout else 0
        return {
            "status": status,
            "strict": strict,
            "input_bytes": len(stream),
            "max_output_frames": max_frames,
            "timeout_limit_seconds": args.timeout_sec,
            "elapsed_seconds": time.monotonic() - started,
            "returncode": returncode,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stderr_head": stderr.decode("utf-8", errors="replace")[:2000],
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-4000:],
            "complete_frames_before_exit": complete_frames,
            "command": command,
        }

    try:
        result = subprocess.run(
            command, input=stream, capture_output=True, timeout=args.timeout_sec
        )
    except FileNotFoundError:
        info = diagnostics(status="ffmpeg_not_found")
        return [], "ffmpeg_not_found", info
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, str):
            stdout = stdout.encode()
        if isinstance(stderr, str):
            stderr = stderr.encode()
        info = diagnostics(status="timeout", stdout=stdout, stderr=stderr)
        return [], "timeout", info
    info = diagnostics(
        status="completed",
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )
    if strict and result.returncode != 0:
        info["status"] = "decoder_error"
        return [], "decoder_error", info
    if not result.stdout:
        info["status"] = "no_frame"
        return [], "no_frame", info
    frames = parse_ppm_sequence(result.stdout)
    status = "decoded" if frames else "no_frame"
    info["status"] = status
    return frames, status, info


def parse_ppm_sequence(data: bytes) -> list[Tensor]:
    frames: list[Tensor] = []
    cursor = 0
    n = len(data)
    while cursor < n:
        if data[cursor : cursor + 2] != b"P6":
            break
        # header: P6 <w> <h> <maxval> then one whitespace byte, then w*h*3 bytes
        fields: list[int] = []
        idx = cursor + 2
        while len(fields) < 3 and idx < n:
            while idx < n and data[idx] in b" \t\n\r":
                idx += 1
            start = idx
            while idx < n and data[idx] not in b" \t\n\r":
                idx += 1
            if start == idx:
                break
            try:
                fields.append(int(data[start:idx]))
            except ValueError:
                break
        if len(fields) != 3 or idx >= n:
            break
        width, height, _ = fields
        idx += 1  # single whitespace after maxval
        payload = idx + width * height * 3
        if payload > n:
            break
        frames.append(parse_ppm(data[cursor:payload]))
        cursor = payload
    return frames


def save_continuation_videos(
    viz: dict[int, dict[str, Any]],
    frame_dir: Path,
    checkpoint_name: str,
    args: argparse.Namespace,
    tag: str = "continuation",
) -> None:
    if not viz:
        return
    print(f"Saving {len(viz)} {tag} videos for {checkpoint_name}", flush=True)
    for clip_idx, item in viz.items():
        gt = item["gt_frames"]
        model = item["model_frames"]
        if not gt:
            continue
        out_path = frame_dir / f"clip_{clip_idx:04d}_{tag}.mp4"
        saved = save_comparison_video(
            reference_frames=gt,
            result_frames=model,
            out_path=out_path,
            ffmpeg_binary=args.ffmpeg_binary,
            fps=args.viz_fps,
            timeout_sec=args.timeout_sec,
            columns=("AR input", "model output"),
            left_blank_from=item["prefix_frames"],
            thumbnail_frame=item["prefix_frames"],
            metadata={
                "checkpoint": checkpoint_name,
                "clip_index": clip_idx,
                "prefix_frames": item["prefix_frames"],
                "generation_starts_at_frame": item["prefix_frames"],
                "black_left_tile": "unknown continuation; not given to the model",
                "h264_path": item["h264_path"],
            },
        )
        if not saved:
            print(f"  continuation video failed for clip {clip_idx}", flush=True)


def _write_stream(
    out_dir: Path,
    checkpoint_name: str,
    clip_idx: int,
    mode: str,
    kind: str,
    data: bytes,
) -> str:
    """Persist one Annex-B byte stream and return its path.

    Both streams are written with the SAME full prefix that ``_survival_and_validity``
    parses, so ``gen`` and ``gt`` are byte-aligned over the prefix and their NALs can be
    paired by index. Without this the generated bytes are unrecoverable -- the .mp4s are
    mpeg4 re-encodes of DECODED frames, not the bitstream -- and every post-hoc question
    (termination cause, codeword-length mismatch, parser-state mismatch) needs a fresh
    GPU run. A few KB per clip buys offline re-analysis of all of them.
    """
    stream_dir = out_dir / "streams" / checkpoint_name
    stream_dir.mkdir(parents=True, exist_ok=True)
    path = stream_dir / f"clip_{clip_idx:04d}_{mode}_{kind}.h264"
    path.write_bytes(data)
    return str(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row)) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    # Union of keys across summaries: continuation/intra and teacher_forced report
    # different columns, so keying off summaries[0] alone would drop or error.
    fieldnames: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


# --- Distributional plausibility (metric 3) -------------------------------------
# A self-contained FVD/MAUVE analog: compare the *population* of real continuation
# frames to the population of generated ones via a Fréchet distance in a cheap,
# dependency-free feature space. No per-sample reference (generation invents a new
# future), and no I3D/Inception download. Swap the two feature fns for a learned
# embedding to get a published-comparable FVD/FID.


def _luma(frame: Tensor) -> Tensor:
    """Rec.601 luma of an [H, W, 3] frame in [0, 1]."""
    f = frame.float()
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def appearance_features(frame: Tensor) -> list[float]:
    """Per-frame appearance: luma mean/std + gradient energy (blur collapses it)."""
    y = _luma(frame)
    dx = (y[:, 1:] - y[:, :-1]).abs().mean()
    dy = (y[1:, :] - y[:-1, :]).abs().mean()
    return [float(y.mean()), float(y.std()), float(0.5 * (dx + dy))]


def motion_features(curr: Tensor, prev: Tensor) -> list[float] | None:
    """Per-transition motion: mean/std of |luma_t - luma_{t-1}| (flow-magnitude
    proxy). ~0 => frozen, >> real => runaway. None if shapes differ."""
    if curr.shape != prev.shape:
        return None
    d = (_luma(curr) - _luma(prev)).abs()
    return [float(d.mean()), float(d.std())]


def _collect_distribution(
    frames: list[Tensor],
    start: int,
    count: int,
    app_acc: list[list[float]],
    mot_acc: list[list[float]],
) -> None:
    """Accumulate appearance/motion features over continuation frames [start, start+count).
    Motion at the first continuation frame is measured against frame start-1 (the seam).
    """
    for t in range(start, start + count):
        if t >= len(frames):
            break
        app_acc.append(appearance_features(frames[t]))
        if t - 1 >= 0 and t - 1 < len(frames):
            mf = motion_features(frames[t], frames[t - 1])
            if mf is not None:
                mot_acc.append(mf)


def _cov(x: Tensor) -> Tensor:
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.t() @ xc) / max(x.shape[0] - 1, 1)
    return cov + 1e-6 * torch.eye(x.shape[1], dtype=x.dtype)


def _sqrtm_psd(m: Tensor) -> Tensor:
    vals, vecs = torch.linalg.eigh(m)
    return (vecs * vals.clamp_min(0).sqrt()) @ vecs.t()


def _frechet(real: list[list[float]], gen: list[list[float]]) -> float | None:
    """Fréchet distance between two Gaussians fit to small feature populations.
    trace((AB)^{1/2}) is computed as trace((A^{1/2} B A^{1/2})^{1/2})."""
    dim = len(real[0]) if real else 0
    if len(real) < dim + 1 or len(gen) < dim + 1:
        return None
    r = torch.tensor(real, dtype=torch.float64)
    g = torch.tensor(gen, dtype=torch.float64)
    diff = r.mean(0) - g.mean(0)
    cov_r, cov_g = _cov(r), _cov(g)
    sa = _sqrtm_psd(cov_r)
    covmean = _sqrtm_psd(sa @ cov_g @ sa)
    fid = float(diff.dot(diff) + torch.trace(cov_r + cov_g - 2 * covmean))
    return max(fid, 0.0)


def distribution_metrics(
    real_app: list[list[float]],
    gen_app: list[list[float]],
    real_mot: list[list[float]],
    gen_mot: list[list[float]],
) -> dict[str, Any]:
    # Keys are kept uniform across modes (None when unavailable) so the summary
    # CSV writer, which fixes columns from the first row, never sees a key drift
    # between continuation (has motion) and intra (no transitions) summaries.
    return {
        "frechet_appearance": _frechet(real_app, gen_app),
        "frechet_motion": _frechet(real_mot, gen_mot),
        "dist_frames_real": len(real_app),
        "dist_frames_gen": len(gen_app),
        "grad_energy_real_mean": mean([f[2] for f in real_app]),
        "grad_energy_gen_mean": mean([f[2] for f in gen_app]),
        "motion_energy_real_mean": mean([f[0] for f in real_mot]),
        "motion_energy_gen_mean": mean([f[0] for f in gen_mot]),
    }


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


if __name__ == "__main__":
    main()
