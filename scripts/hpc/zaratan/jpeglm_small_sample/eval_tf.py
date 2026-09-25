#!/usr/bin/env python3
"""Fixed-hole teacher-forced FIM benchmark for the isolated small-sample runs.

The benchmark preserves the checkpoint's MEGABYTE patch phase and scores only
the deleted byte targets. EOS is reported separately. It never uses free-run
generation or GT-length stopping as a success metric.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte.data import BYTE_VOCAB_SIZE, IGNORE_INDEX, SEQ_EOS_ID  # noqa: E402
from litgpt.byte.megabyte_inference import megabyte_teacher_forced_sample  # noqa: E402
from litgpt.byte.reconstruction import _unwrap_model  # noqa: E402
from scripts.byte.eval import eval_fim_avclm as FIM  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import load_model  # noqa: E402
from scripts.hpc.zaratan.jpeglm_small_sample.submit import load_config, load_evaluation_config  # noqa: E402


STRUCTURAL_SYNTAX_CATEGORIES = frozenset({
    HS.Category.START_CODE,
    HS.Category.NAL_HEADER,
    HS.Category.EMULATION_PREVENTION,
    HS.Category.SPS,
    HS.Category.PPS,
    HS.Category.SEI,
    HS.Category.SLICE_HEADER,
    HS.Category.RBSP_TRAILING,
})
CONTENT_DEPENDENT_CATEGORIES = frozenset({
    HS.Category.MB_HEADER,
    HS.Category.MB_PRED,
    HS.Category.CBP,
    HS.Category.MB_QP_DELTA,
    HS.Category.RESIDUAL_LUMA,
    HS.Category.RESIDUAL_CHROMA,
})
SYNTAX_BUCKETS = (
    "structural_syntax",
    "content_dependent",
    "mixed",
    "unclassified",
)
EVAL_PROTOCOL_ID = "feasible_35_holes_v1"
# The frozen run configs request 50 holes. On this 256-video subset, none of
# the 60 held-out GOP windows has a P-frame eligible for a 256-byte cut (and
# therefore not for 400/600 bytes). Keep the feasible train/val comparison
# identical without changing either run's frozen training configuration.
EVAL_STRATA = (
    ("idr", 64), ("idr", 128), ("idr", 256), ("idr", 400), ("idr", 600),
    ("p", 64), ("p", 128),
)


def evaluation_strata(evaluation: dict) -> tuple[list[tuple[str, int]], list[dict[str, Any]]]:
    requested = [
        (frame_type, length)
        for frame_type in evaluation["frame_types"]
        for length in evaluation["corruption_lengths"]
    ]
    if requested != [
        (frame_type, length)
        for frame_type in ("idr", "p")
        for length in (64, 128, 256, 400, 600)
    ]:
        raise ValueError(
            f"{EVAL_PROTOCOL_ID} requires the frozen IDR/P, 64/128/256/400/600B "
            "evaluation config; use a different protocol for other settings"
        )
    if evaluation["samples_per_length"] != 5:
        raise ValueError(f"{EVAL_PROTOCOL_ID} requires five holes per stratum")
    omitted = [
        {"frame_type": frame_type, "corruption_length_bytes": length,
         "reason": "no eligible held-out P-frame window in the fixed 256-video subset"}
        for frame_type, length in requested if (frame_type, length) not in EVAL_STRATA
    ]
    return list(EVAL_STRATA), omitted


def syntax_bucket(categories: set[HS.Category]) -> str:
    """Assign a whole byte only when all overlapping bit fields agree."""
    if not categories or categories - STRUCTURAL_SYNTAX_CATEGORIES - CONTENT_DEPENDENT_CATEGORIES:
        return "unclassified"
    structural = bool(categories & STRUCTURAL_SYNTAX_CATEGORIES)
    content = bool(categories & CONTENT_DEPENDENT_CATEGORIES)
    if structural and content:
        return "mixed"
    return "structural_syntax" if structural else "content_dependent"


def sample_identity(sample: FIM.WindowFimSample, split: str) -> dict[str, Any]:
    return {
        "eval_split": split,
        "h264_path": str(sample.h264_path),
        "start_nal": sample.start_nal,
        "end_nal": sample.end_nal,
        "frame_lo": sample.frame_lo,
        "frame_hi": sample.frame_hi,
        "split_byte": sample.split,
        "gap": sample.gap,
        "frame_type": sample.corruption_frame_type,
        "target_sha256": hashlib.sha256(sample.target_bytes).hexdigest(),
    }


def build_samples(values: dict[str, str], evaluation: dict, split: str) -> list[FIM.WindowFimSample]:
    samples = []
    per_length = evaluation["samples_per_length"]
    strata, _ = evaluation_strata(evaluation)
    for frame_type, length in strata:
        # Require only the requested cut to fit. Reusing a source window
        # across severities is intentional: each severity remains five
        # distinct windows, while cross-severity comparisons are paired.
        args = argparse.Namespace(
            manifest=Path(values["MANIFEST"]),
            nal_index_path=Path(values["NAL_INDEX"]),
            train_split_file=Path(values["OUT_DIR"]) / "train_split.json",
            eval_split=split,
            max_manifest_rows=int(values["MAX_ROWS"]),
            max_window_bytes=int(values["RAW_CONTEXT_BYTES"]) - 1,
            window_min_frames=int(values["WINDOW_MIN_FRAMES"]),
            window_unit=values["WINDOW_UNIT"],
            val_fraction=float(values["VAL_FRACTION"]),
            split_by_video=True,
            seed=evaluation["seed"],
            fim_format=values["FIM_FORMAT"],
            fim_loss_scope=values["FIM_LOSS_SCOPE"],
            use_eos=True,
            fim_min_gap=int(values["FIM_MIN_GAP"]),
            fim_max_gap=int(values["FIM_MAX_GAP"]),
            slice_header_guard_bytes=int(values["SLICE_HEADER_GUARD_BYTES"]),
            hole_placement="corrupt_gen_frame",
            hole_set="sampled",
            corr_pos=evaluation["corruption_position"],
            corr_len_bytes=None,
            corr_len_bytes_list=[length],
            corr_samples_per_length=per_length,
            corr_eligibility_bytes=length,
            corr_header_guard_bytes=evaluation["corruption_header_guard_bytes"],
            corr_frame_type=frame_type,
            num_clips=per_length,
        )
        try:
            selection = FIM.build_eval_sample_selection(args)
        except RuntimeError as error:
            raise RuntimeError(
                f"Cannot select {per_length} distinct {split}/{frame_type}/"
                f"{length}B evaluation windows: {error}"
            ) from error
        if len(selection.samples) != per_length:
            raise RuntimeError(
                f"{split}/{frame_type}/{length}B: expected {per_length} "
                f"samples, got {len(selection.samples)}; no silent dropping is allowed"
            )
        if any(
            s.corruption_frame_type != frame_type or s.gap != length
            for s in selection.samples
        ):
            raise AssertionError(
                f"{split}/{frame_type}/{length}B: wrong corruption class selected"
            )
        samples.extend(selection.samples)
    if len(samples) != 35:
        raise AssertionError(f"{EVAL_PROTOCOL_ID} selected {len(samples)} holes, expected 35")
    return samples


def verify_shared_sample_set(samples: list[FIM.WindowFimSample], evaluation: dict, split: str) -> None:
    path = Path(evaluation["sample_set_dir"]) / EVAL_PROTOCOL_ID / f"{split}.json"
    identities = [sample_identity(sample, split) for sample in samples]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(identities, indent=2, sort_keys=True) + "\n"
    with (path.parent / ".sample-set.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != identities:
                raise RuntimeError(
                    f"Evaluation samples differ from the shared fixed set: {path}. "
                    "Check the corpus, training split, and sample-selection settings."
                )
        else:
            with path.open("x", encoding="utf-8") as file:
                file.write(payload)
    print(f"Verified {len(samples)} fixed evaluation holes: {path}", flush=True)


@torch.inference_mode()
def score_sample(model: torch.nn.Module, sample: FIM.WindowFimSample, device: torch.device) -> dict:
    raw = _unwrap_model(model)
    patch_size = int(raw.config.byte_patch_size)
    if patch_size > 1:
        patched = megabyte_teacher_forced_sample(
            sample.teacher_input_ids,
            sample.teacher_labels,
            sample.teacher_region_ids,
            sample.teacher_offset_ids,
            patch_size,
        )
        inputs = patched["input_ids"]
        labels = patched["labels"].to(device)
        regions = patched["region_ids"]
        offsets = patched["offset_ids"]
        extra = {"patch_targets": labels}
    else:
        inputs = sample.teacher_input_ids.unsqueeze(0)
        labels = sample.teacher_labels.unsqueeze(0).to(device)
        regions = sample.teacher_region_ids.unsqueeze(0)
        offsets = sample.teacher_offset_ids.unsqueeze(0)
        extra = {}
    if inputs.size(1) > raw.max_seq_length:
        raise RuntimeError(f"FIM sample exceeds model context: {inputs.size(1)} > {raw.max_seq_length}")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = raw(
            inputs.to(device),
            region_ids=regions.to(device),
            offset_ids=offsets.to(device),
            **extra,
        )
    mask = FIM._teacher_forced_span_mask(labels, sample.target_length)
    if mask is None:
        raise RuntimeError("Missing supervised FIM target positions")
    target = labels[mask]
    selected = logits[mask].float()
    expected = torch.tensor(
        list(sample.target_bytes) + [SEQ_EOS_ID], device=device, dtype=target.dtype
    )
    if not torch.equal(target, expected):
        raise RuntimeError("FIM target labels differ from the deleted bytes + EOS")
    byte_logits = selected[:-1]
    byte_labels = target[:-1]
    nll = F.cross_entropy(byte_logits, byte_labels, reduction="none")
    eos_logits = selected[-1]
    eos_prob = float(torch.softmax(eos_logits, dim=-1)[SEQ_EOS_ID])
    eos_rank = 1 + int((eos_logits > eos_logits[SEQ_EOS_ID]).sum())
    return {
        "byte_nll_bits": (nll / math.log(2)).cpu().tolist(),
        "byte_correct": int((byte_logits.argmax(dim=-1) == byte_labels).sum()),
        "eos_probability": eos_prob,
        "eos_rank": eos_rank,
    }


def syntax_annotations(sample: FIM.WindowFimSample) -> list[dict[str, str]]:
    """Record all parser spans touching each deleted Annex-B byte."""
    try:
        spans = HS.parse_stream(sample.gt_truncated_stream, parse_slice_data=True).all_spans()
    except Exception:
        return [
            {"owners": "parser unavailable", "bucket": "unclassified"}
            for _ in range(sample.target_length)
        ]
    names: list[list[str]] = [[] for _ in range(sample.target_length)]
    categories: list[set[HS.Category]] = [set() for _ in range(sample.target_length)]
    for span in spans:
        lo = max(0, span.byte_start - sample.split)
        hi = min(sample.target_length, span.byte_end - sample.split)
        for index in range(lo, hi):
            if span.name not in names[index]:
                names[index].append(span.name)
            categories[index].add(span.category)
    return [
        {
            "owners": ", ".join(names[index]) if names[index] else "unattributed",
            "bucket": syntax_bucket(categories[index]),
        }
        for index in range(sample.target_length)
    ]


def write_heatmap(path: Path, sample: FIM.WindowFimSample, scores: dict, annotations: list[dict[str, str]]) -> None:
    cells = []
    for index, (bit_loss, annotation) in enumerate(zip(scores["byte_nll_bits"], annotations)):
        clipped = min(max(float(bit_loss), 0.0), 8.0) / 8.0
        red = int(245 - 55 * clipped)
        green = int(245 - 190 * clipped)
        blue = int(245 - 190 * clipped)
        label = (
            f"byte {index} | stream offset {sample.split + index} | "
            f"GT {sample.target_bytes[index]:02x} | {bit_loss:.3f} bits | "
            f"{annotation['bucket']} | {annotation['owners']}"
        )
        cells.append(
            f'<span class="byte" style="background:rgb({red},{green},{blue})" '
            f'title="{html.escape(label, quote=True)}">{sample.target_bytes[index]:02x}</span>'
        )
    title = html.escape(f"{sample.h264_path.name} | {sample.corruption_frame_type} | {sample.gap} B")
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>FIM byte loss</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:1100px}"
        ".grid{display:grid;grid-template-columns:repeat(32,1.9rem);gap:3px}"
        ".byte{font:12px monospace;text-align:center;padding:4px 1px}"
        "</style>"
        f"<h1>{title}</h1><p>Teacher-forced loss per deleted byte. "
        "Pale = low loss; red = high loss. Hover for GT byte and syntax ownership.</p>"
        f"<div class='grid'>{''.join(cells)}</div>",
        encoding="utf-8",
    )


def aggregate(rows: list[dict]) -> dict:
    total_bytes = sum(row["target_bytes"] for row in rows)
    total_bits = sum(row["byte_loss_bits_sum"] for row in rows)
    bits_per_byte = total_bits / total_bytes if total_bytes else None
    bucket_totals = {bucket: {"bytes": 0, "loss_bits_sum": 0.0} for bucket in SYNTAX_BUCKETS}
    for row in rows:
        for bucket, values in row["syntax_buckets"].items():
            bucket_totals[bucket]["bytes"] += values["bytes"]
            bucket_totals[bucket]["loss_bits_sum"] += values["loss_bits_sum"]
    if sum(values["bytes"] for values in bucket_totals.values()) != total_bytes:
        raise AssertionError("Syntax buckets do not cover every scored byte")
    return {
        "samples": len(rows),
        "target_bytes": total_bytes,
        "span_bits_per_byte": bits_per_byte,
        "span_perplexity": 2**bits_per_byte if bits_per_byte is not None else None,
        "span_byte_accuracy": sum(row["byte_correct"] for row in rows) / total_bytes if total_bytes else None,
        "eos_probability_mean": statistics.mean(row["eos_probability"] for row in rows) if rows else None,
        "eos_rank_median": statistics.median(row["eos_rank"] for row in rows) if rows else None,
        "eos_top1_rate": sum(row["eos_rank"] == 1 for row in rows) / len(rows) if rows else None,
        "by_syntax_bucket": {
            bucket: {
                "bytes": values["bytes"],
                "byte_fraction": values["bytes"] / total_bytes if total_bytes else None,
                "ce_bits_per_byte": values["loss_bits_sum"] / values["bytes"] if values["bytes"] else None,
                "contribution_bits_per_target_byte": values["loss_bits_sum"] / total_bytes if total_bytes else None,
            }
            for bucket, values in bucket_totals.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--checkpoint", default="final",
        help="Checkpoint directory under the run (final or step-XXXXXXXX)",
    )
    args = parser.parse_args()
    values = load_config(args.config)
    evaluation = load_evaluation_config(args.config)
    run_dir = Path(values["OUT_DIR"])
    record_path = run_dir / "small_sample_config.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    values["SMALL_SAMPLE_CONFIG"] = str(run_dir / "small_sample_config.yaml")
    if record != {"training_environment": values, "evaluation": evaluation}:
        raise RuntimeError(
            f"Evaluation config differs from the frozen training config: {record_path}"
        )
    split = args.split
    if args.checkpoint != "final" and not (
        args.checkpoint.startswith("step-")
        and len(args.checkpoint) == 13
        and args.checkpoint[5:].isdigit()
    ):
        raise ValueError("--checkpoint must be final or step-XXXXXXXX")
    strata, omitted_strata = evaluation_strata(evaluation)
    print(
        f"Evaluation protocol {EVAL_PROTOCOL_ID}: {len(strata) * evaluation['samples_per_length']} "
        f"holes per split; {len(omitted_strata)} requested strata not evaluated",
        flush=True,
    )
    checkpoint = run_dir / args.checkpoint
    if not (checkpoint / "lit_model.pth").is_file():
        raise FileNotFoundError(f"Final checkpoint missing: {checkpoint / 'lit_model.pth'}")
    out = run_dir / "eval_small_sample" / args.checkpoint / EVAL_PROTOCOL_ID / split
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Evaluation output already exists; refusing overwrite: {out}")
    samples = build_samples(values, evaluation, split)
    verify_shared_sample_set(samples, evaluation, split)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = load_model(checkpoint, device, dtype=torch.bfloat16)
    rows = []
    with (out / "samples.jsonl").open("w", encoding="utf-8") as sample_file, (
        out / "bytes.jsonl"
    ).open("w", encoding="utf-8") as byte_file:
        heatmaps = out / "heatmaps"
        heatmaps.mkdir()
        for sample_id, sample in enumerate(samples):
            scores = score_sample(model, sample, device)
            annotations = syntax_annotations(sample)
            bits = scores["byte_nll_bits"]
            if len(annotations) != len(bits):
                raise AssertionError("Syntax annotations do not match scored bytes")
            syntax_buckets = {bucket: {"bytes": 0, "loss_bits_sum": 0.0} for bucket in SYNTAX_BUCKETS}
            for bit_loss, annotation in zip(bits, annotations):
                bucket = syntax_buckets[annotation["bucket"]]
                bucket["bytes"] += 1
                bucket["loss_bits_sum"] += bit_loss
            identity = sample_identity(sample, split)
            row = {
                "sample_id": sample_id,
                **identity,
                "target_bytes": len(bits),
                "byte_loss_bits_sum": sum(bits),
                "span_bits_per_byte": statistics.mean(bits),
                "byte_correct": scores["byte_correct"],
                "eos_probability": scores["eos_probability"],
                "eos_rank": scores["eos_rank"],
                "syntax_buckets": syntax_buckets,
            }
            rows.append(row)
            sample_file.write(json.dumps(row, sort_keys=True) + "\n")
            for index, (bit_loss, annotation) in enumerate(zip(bits, annotations)):
                byte_file.write(json.dumps({
                    "sample_id": sample_id,
                    "byte_index": index,
                    "stream_offset": sample.split + index,
                    "gt_byte": sample.target_bytes[index],
                    "nll_bits": bit_loss,
                    "syntax_owners": annotation["owners"],
                    "syntax_bucket": annotation["bucket"],
                }, sort_keys=True) + "\n")
            if sample_id < evaluation["num_heatmaps"]:
                write_heatmap(heatmaps / f"sample-{sample_id:03d}.html", sample, scores, annotations)
            print(
                f"[{split}] {sample_id + 1}/{len(samples)} "
                f"{sample.corruption_frame_type} {sample.gap}B "
                f"CE={row['span_bits_per_byte']:.3f} bits/byte "
                f"EOS p={row['eos_probability']:.4f} rank={row['eos_rank']}",
                flush=True,
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[f"{row['frame_type']}/{row['gap']}B"].append(row)
    summary = {
        "checkpoint": str(checkpoint),
        "eval_split": split,
        "eval_protocol": EVAL_PROTOCOL_ID,
        "requested_strata": [
            {"frame_type": frame_type, "corruption_length_bytes": length}
            for frame_type in evaluation["frame_types"]
            for length in evaluation["corruption_lengths"]
        ],
        "evaluated_strata": [
            {"frame_type": frame_type, "corruption_length_bytes": length,
             "samples": evaluation["samples_per_length"]}
            for frame_type, length in strata
        ],
        "not_evaluated_strata": omitted_strata,
        "syntax_bucket_definition": {
            "structural_syntax": sorted(category.value for category in STRUCTURAL_SYNTAX_CATEGORIES),
            "content_dependent": sorted(category.value for category in CONTENT_DEPENDENT_CATEGORIES),
            "mixed": "byte overlaps both structural and content-dependent fields",
            "unclassified": "no parsed span, opaque slice data, unknown category, or parser failure",
        },
        "overall": aggregate(rows),
        "by_frame_type_and_length": {key: aggregate(group) for key, group in sorted(grouped.items())},
    }
    torch.cuda.synchronize(device)
    summary["eval_wall_seconds"] = time.perf_counter() - started
    summary["peak_gpu_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    summary["peak_gpu_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[{split}] overall: {json.dumps(summary['overall'])}", flush=True)
    print(f"[{split}] output: {out}", flush=True)


if __name__ == "__main__":
    main()
