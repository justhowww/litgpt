#!/usr/bin/env python3
"""How much does a MEGABYTE checkpoint rely on its global transformer?

Teacher-forced CE on held-out FIM windows under four global-signal conditions,
applied with a forward hook on ``megabyte_global_to_local`` (the only path from
the global model into the local byte model):

  baseline  unchanged
  zero      global conditioning removed; the local model sees only in-patch bytes
  mean      every patch gets the window's average conditioning (no position info)
  roll      conditioning shifted by half the window (right statistics, wrong place)

It also records baseline CE by byte offset inside a patch (0..P-1). If the
local model does most of the work, removing the global signal costs little and
CE falls steeply across each patch as in-patch context accumulates.

Per-field breakdown: every scored byte whose original-order prefix the model
has seen (context, prefix, middle; not the orphan) is labelled with the H.264
syntax category that owns it (h264_syntax spans; "mixed" if a byte straddles
categories) and with its frame's index in the window. Slice headers repeat or
increment from frame to frame, so a working global model should make them
nearly free once earlier frames are visible ("slice_header" by frame index),
and zeroing the global signal should hurt them sharply.

A plain byte transformer (byte_patch_size == 1, e.g. run F) has no global
signal: only the baseline condition runs, with the same per-field and per-frame
breakdowns, so its numbers line up with the MEGABYTE baselines.

Windows come from videos NOT in the run's train_split.json (held out), built
with the run's own window unit/FIM settings; holes are deterministic per
window, and FIM frames are drawn 50/50 IDR/P so both frame types are scored.

    python scripts/byte/eval/megabyte_global_ablation.py RUN/step-00200000 \
        --train-split-file RUN/train_split.json --manifest DATA/manifest.jsonl \
        --num-videos 200 --out RUN/eval_global_ablation/step-00200000.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litgpt.byte.data import (  # noqa: E402
    BYTE_VOCAB_SIZE,
    IGNORE_INDEX,
    REGION_BRIDGE,
    ByteStreamWindowDataset,
    load_manifest_rows,
    patch_byte_sample,
)
from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte.reconstruction import _unwrap_model  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import load_model  # noqa: E402

CONDITIONS = ("baseline", "zero", "mean", "roll")


class GlobalSignalHook:
    """Rewrites the global->local conditioning ``[B, T, P, D_local]``."""

    def __init__(self) -> None:
        self.mode = "baseline"

    def __call__(self, module, inputs, output):
        if self.mode == "baseline":
            return output
        if self.mode == "zero":
            return torch.zeros_like(output)
        if self.mode == "mean":
            return output.mean(dim=1, keepdim=True).expand_as(output)
        if self.mode == "roll":
            shift = max(1, output.size(1) // 2)
            return torch.roll(output, shifts=shift, dims=1)
        raise ValueError(self.mode)


def window_labels(dataset, item):
    """Per-label (category, frame index) for byte labels, aligned with score_window."""
    meta = item["sample_meta"]
    path = meta["h264_path"]
    nals = dataset.nal_index[str(path)]
    start, end = int(meta["start_nal"]), int(meta["end_nal"])
    base = nals[start].start
    window = Path(path).read_bytes()[base : nals[end - 1].end]
    owners: list[set] = [set() for _ in range(len(window))]
    try:
        for span in HS.parse_stream(window, parse_slice_data=True).all_spans():
            for b in range(max(0, span.byte_start), min(len(window), span.byte_end)):
                owners[b].add(span.category.value)
    except Exception:  # noqa: BLE001 - unparsable window: leave bytes unclassified
        pass
    category = [
        next(iter(o)) if len(o) == 1 else ("mixed" if o else "unclassified") for o in owners
    ]
    frame = [-1] * len(window)  # -1 = parameter sets / SEI before the first slice
    k = -1
    for nal in nals[start:end]:
        if nal.nal_type in (1, 5):
            k += 1
        for b in range(nal.start - base, nal.end - base):
            frame[b] = k
    labels = item["labels"]
    keep = (labels != IGNORE_INDEX) & (labels < BYTE_VOCAB_SIZE)
    gap, split = int(meta["fim_gap"]), int(meta["fim_split"])
    frame_lo, frame_hi = int(meta["frame_lo"]), int(meta["frame_hi"])
    offsets = dataset._fim_label_window_offsets(
        frame_lo, split, gap, split - frame_lo, frame_hi - split - gap,
        gap + (1 if dataset.use_eos else 0), labels.numel(),
    )
    out_cat, out_frame = [], []
    for off in offsets[keep].tolist():
        if off < 0:
            out_cat.append("orphan")
            out_frame.append(-2)
        else:
            out_cat.append(category[off])
            out_frame.append(frame[off])
    return out_cat, out_frame


def heldout_rows(manifest: Path, split_file: Path, num_videos: int, seed: int):
    split = json.loads(split_file.read_text(encoding="utf-8"))
    train_videos = {str(Path(v)) for v in split.get("videos", [])}
    if not train_videos:
        raise RuntimeError(f"{split_file} lists no training videos")
    rows = load_manifest_rows(manifest)
    held = [r for r in rows if str(Path(r["h264_path"])) not in train_videos]
    if not held:
        raise RuntimeError("no held-out videos: every manifest row is a training video")
    random.Random(seed).shuffle(held)
    return held[:num_videos], split, len(rows), len(held)


@torch.inference_mode()
def score_window(model, item, patch_size, device):
    """Per-target NLL (nats) with category tags, for one teacher-forced window."""
    patched = patch_byte_sample(item, patch_size)
    labels = patched["labels"].unsqueeze(0).to(device)
    logits = model(
        patched["input_ids"].unsqueeze(0).to(device),
        region_ids=patched["region_ids"].unsqueeze(0).to(device),
        offset_ids=patched["offset_ids"].unsqueeze(0).to(device),
        patch_targets=labels,
    ).float()
    supervised = (labels != IGNORE_INDEX) & (labels < BYTE_VOCAB_SIZE)  # bytes only
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    offsets = torch.arange(patch_size, device=device).view(1, 1, -1).expand_as(labels)
    middle = patched["target_region_ids"].unsqueeze(0).to(device) == REGION_BRIDGE
    return nll[supervised], offsets[supervised], middle[supervised]


@torch.inference_mode()
def score_window_plain(model, item, device):
    """Same as score_window for a plain byte transformer (byte_patch_size == 1)."""
    raw = _unwrap_model(model)
    labels = item["labels"].unsqueeze(0).to(device)
    kwargs = {}
    if raw.config.use_region_id:
        kwargs["region_ids"] = item["region_ids"].unsqueeze(0).to(device)
    if raw.config.use_offset_id:
        kwargs["offset_ids"] = item["offset_ids"].unsqueeze(0).to(device)
    logits = model(item["input_ids"].unsqueeze(0).to(device), **kwargs).float()
    supervised = (labels != IGNORE_INDEX) & (labels < BYTE_VOCAB_SIZE)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    offsets = torch.zeros_like(labels)
    # The input at a middle position (FIM_END, then generated bytes) predicts a middle byte.
    middle = item["region_ids"].unsqueeze(0).to(device) == REGION_BRIDGE
    return nll[supervised], offsets[supervised], middle[supervised]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="checkpoint dir (lit_model.pth + model_config.yaml)")
    ap.add_argument("--train-split-file", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None, help="default: manifest recorded in the split file")
    ap.add_argument("--num-videos", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    split = json.loads(args.train_split_file.read_text(encoding="utf-8"))
    manifest = args.manifest or Path(split["manifest"])
    rows, split, n_rows, n_held = heldout_rows(manifest, args.train_split_file, args.num_videos, args.seed)
    print(f"held-out videos: {n_held:,} of {n_rows:,}; using {len(rows)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.dtype == "bf16" and device.type == "cuda") else torch.float32
    model = load_model(args.checkpoint, device, dtype=dtype)
    raw = _unwrap_model(model)
    patch_size = int(raw.config.byte_patch_size)
    megabyte = patch_size > 1 and hasattr(raw, "megabyte_global_to_local")
    # A plain byte transformer has no global signal to ablate: baseline (and the
    # per-field / per-frame breakdowns) only, comparable with MEGABYTE baselines.
    conditions = CONDITIONS if megabyte else ("baseline",)
    field_conditions = tuple(c for c in ("baseline", "zero") if c in conditions)
    budget = int(raw.config.block_size) * patch_size - 1

    dataset = ByteStreamWindowDataset(
        rows,
        max_seq_length=budget,
        min_frames=2,
        p_fim=1.0,
        fim_format=split.get("fim_format", "psm"),
        fim_loss_scope=split.get("fim_loss_scope", "full"),
        use_eos=bool(split.get("use_eos", True)),
        frame_guard_bytes=0,
        window_unit=split.get("window_unit", "gop"),
        resample_fim=False,  # deterministic hole per window
        fim_idr_sampling_probability=0.5,
        seed=args.seed,
    )
    hook = GlobalSignalHook()
    handle = raw.megabyte_global_to_local.register_forward_hook(hook) if megabyte else None

    sums = {c: defaultdict(float) for c in CONDITIONS}
    counts = defaultdict(int)
    offset_sum = torch.zeros(patch_size, dtype=torch.float64)
    offset_cnt = torch.zeros(patch_size, dtype=torch.float64)
    offset_sum_zero = torch.zeros(patch_size, dtype=torch.float64)
    field_sum = {c: defaultdict(float) for c in ("baseline", "zero")}
    field_cnt = defaultdict(int)
    header_sum = {c: defaultdict(float) for c in ("baseline", "zero")}
    header_cnt = defaultdict(int)
    frame_sum = defaultdict(float)
    frame_cnt = defaultdict(int)
    started = time.perf_counter()
    for index in range(len(dataset)):
        item = dataset[index]
        frame = {5: "idr", 1: "p"}.get(item["sample_meta"].get("fim_frame_nal_type"), "other")
        if item["input_ids"].numel() > budget + 8:
            continue
        cats, frames = window_labels(dataset, item)
        for condition in conditions:
            hook.mode = condition
            if megabyte:
                nll, offs, middle = score_window(model, item, patch_size, device)
            else:
                nll, offs, middle = score_window_plain(model, item, device)
            nll = nll.double().cpu(); offs = offs.cpu(); middle = middle.cpu()
            groups = {"all": torch.ones_like(middle), "middle": middle, "context": ~middle,
                      f"middle_{frame}": middle}
            for name, mask in groups.items():
                sums[condition][name] += float(nll[mask].sum())
                if condition == "baseline":
                    counts[name] += int(mask.sum())
            if condition in ("baseline", "zero"):
                if len(cats) != nll.numel():
                    raise RuntimeError(f"label alignment: {len(cats)} labels vs {nll.numel()} scores")
                for value, cat, fr in zip(nll.tolist(), cats, frames):
                    field_sum[condition][cat] += value
                    if cat == "slice_header":
                        header_sum[condition][fr] += value
                    if condition == "baseline":
                        field_cnt[cat] += 1
                        frame_sum[fr] += value
                        frame_cnt[fr] += 1
                        if cat == "slice_header":
                            header_cnt[fr] += 1
                target = offset_sum if condition == "baseline" else offset_sum_zero
                target.index_add_(0, offs, nll)
                if condition == "baseline":
                    offset_cnt.index_add_(0, offs, torch.ones_like(nll))
        if (index + 1) % 25 == 0:
            print(f"[{index + 1}/{len(dataset)}] windows, {time.perf_counter() - started:.0f}s", flush=True)
    if handle is not None:
        handle.remove()

    ce = {c: {g: sums[c][g] / counts[g] for g in counts if counts[g]} for c in conditions}
    result = {
        "checkpoint": str(args.checkpoint),
        "patch_size": patch_size,
        "windows": len(dataset),
        "heldout_videos_used": len(rows),
        "target_bytes": dict(counts),
        "ce_nats_per_byte": ce,
        "ce_increase_vs_baseline": {
            c: {g: ce[c][g] - ce["baseline"][g] for g in ce["baseline"]} for c in conditions if c != "baseline"
        },
        "relative_increase": {
            c: {g: ce[c][g] / ce["baseline"][g] - 1 for g in ce["baseline"]} for c in conditions if c != "baseline"
        },
        "ce_by_field": {
            c: {k: field_sum[c][k] / field_cnt[k] for k in sorted(field_cnt)} for c in field_conditions
        },
        "bytes_by_field": dict(sorted(field_cnt.items())),
        "slice_header_ce_by_frame_index": {
            c: {str(k): header_sum[c][k] / header_cnt[k] for k in sorted(header_cnt)} for c in field_conditions
        },
        "slice_header_bytes_by_frame_index": {str(k): v for k, v in sorted(header_cnt.items())},
        "baseline_ce_by_frame_index": {str(k): frame_sum[k] / frame_cnt[k] for k in sorted(frame_cnt)},
        "baseline_ce_by_patch_offset": (offset_sum / offset_cnt.clamp_min(1)).tolist(),
        "zero_global_ce_by_patch_offset": (offset_sum_zero / offset_cnt.clamp_min(1)).tolist(),
        "bytes_by_patch_offset": offset_cnt.tolist(),
        "seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    print(f"\nCE (nats/byte) on {len(dataset)} held-out windows, patch {patch_size}")
    groups = [g for g in ("all", "context", "middle", "middle_idr", "middle_p") if g in counts]
    print(f"{'condition':10s}" + "".join(f"{g:>13s}" for g in groups))
    for c in conditions:
        print(f"{c:10s}" + "".join(f"{ce[c][g]:>13.3f}" for g in groups))
    print("relative increase vs baseline:")
    for c in conditions[1:]:
        print(f"{c:10s}" + "".join(f"{100 * result['relative_increase'][c][g]:>12.1f}%" for g in groups))
    prof = result["baseline_ce_by_patch_offset"]
    marks = sorted({0, 1, 2, 4, 8, 16, 32, 64, 128, patch_size - 1} & set(range(patch_size)))
    print("baseline CE by offset in patch: " + ", ".join(f"{k}:{prof[k]:.2f}" for k in marks))
    print("\nCE by syntax field (nats/byte): baseline -> zero-global   [bytes]")
    for k in sorted(field_cnt, key=lambda k: -field_cnt[k]):
        b = result["ce_by_field"]["baseline"][k]
        if not megabyte:
            print(f"  {k:18s} {b:6.3f}  [{field_cnt[k]:,}]")
            continue
        z = result["ce_by_field"]["zero"][k]
        print(f"  {k:18s} {b:6.3f} -> {z:6.3f}  ({100 * (z / b - 1):+6.1f}%)  [{field_cnt[k]:,}]")
    print("slice-header CE by frame index in window (baseline / zero-global):")
    for k in sorted(header_cnt):
        zero = f" / {header_sum['zero'][k] / header_cnt[k]:6.3f}" if megabyte else ""
        print(f"  frame {k:>3}: {header_sum['baseline'][k] / header_cnt[k]:6.3f}{zero}  [{header_cnt[k]:,} bytes]")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
