#!/usr/bin/env python3
"""Extract hidden states + parser-state labels for linear probes (one checkpoint).

Rebuilds the same SPM FIM test samples as the overfit-ladder test evaluation
(fixed --test-rows, deterministic held-out holes), runs one teacher-forced forward
pass per sample on the frozen model, and records hidden states of a few layers at
positions in four regions of the hole's neighbourhood:

  context  last bytes of earlier frames (normal left-to-right reading)
  suffix   received bytes after the hole (shown first in SPM order)
  prefix   bytes of the damaged frame before the hole
  middle   the hole itself, teacher-forced

The hidden state at the input position of window byte ``w`` is paired with the
parser state *after* that byte (``litgpt.byte.probe_labels``, index ``w + 1``).
``--random-init`` runs the same architecture with untrained weights (probe control).

Output: one .npz with X_<layer> (float16, N x d) and label/metadata arrays.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litgpt.byte.data import FIM_BEGIN_ID, FIM_HOLE_ID, source_video_id  # noqa: E402
from litgpt.byte.probe_labels import parser_state_labels  # noqa: E402
from litgpt.model import GPT, Config  # noqa: E402
from scripts.byte.eval import eval_fim_avclm as FIM  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import load_model  # noqa: E402

REGIONS = ("context", "suffix", "prefix", "middle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--nal-index-path", type=Path, required=True)
    p.add_argument("--test-rows", type=int, nargs=2, default=(70000, 70400))
    p.add_argument("--num-clips", type=int, default=300)
    p.add_argument("--fim-max-gap", type=int, default=600)
    p.add_argument("--max-window-bytes", type=int, default=16384)
    p.add_argument("--layers", type=str, default="quartiles",
                   help="'quartiles' or comma-separated block indices")
    p.add_argument("--max-per-region", type=int, default=256)
    p.add_argument("--context-bytes", type=int, default=400)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def build_samples(args: argparse.Namespace):
    """Same sample construction as the overfit-ladder test evaluation."""
    argv = [
        "eval_fim_avclm.py", str(args.manifest),
        "--nal-index-path", str(args.nal_index_path),
        "--checkpoint-dirs", str(args.run_dir / args.checkpoint),
        "--out-dir", str(args.out.parent / "_unused"),
        "--train-split-file", str(args.run_dir / "train_split.json"),
        "--max-window-bytes", str(args.max_window_bytes),
        "--window-min-frames", "2", "--window-unit", "gop",
        "--split-by-video", "--seed", "42",
        "--fim-format", "spm", "--fim-loss-scope", "full", "--use-eos",
        "--fim-min-gap", "64", "--fim-max-gap", str(args.fim_max_gap),
        "--slice-header-guard-bytes", "0",
        "--eval-split", "test", "--test-rows", *map(str, args.test_rows),
        "--hole-set", "heldout", "--heldout-holes-per-window", "4",
        "--num-clips", str(args.num_clips), "--slice-layout", "frame",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        fim_args = FIM.parse_args()
    finally:
        sys.argv = saved
    return FIM.build_eval_sample_selection(fim_args).samples


def region_positions(sample) -> dict[str, list[tuple[int, int]]]:
    """(input index, window byte offset) pairs per region, for the SPM layout:
    [context, FIM_HOLE, suffix, FIM_BEGIN, prefix, middle]."""
    lo, hi, split, gap = sample.frame_lo, sample.frame_hi, sample.split, sample.gap
    ids = sample.teacher_input_ids
    n_suffix = hi - (split + gap)
    hole_at, begin_at = lo, lo + 1 + n_suffix
    if int(ids[hole_at]) != FIM_HOLE_ID or int(ids[begin_at]) != FIM_BEGIN_ID:
        raise RuntimeError("sample is not in the expected SPM layout")
    window = sample.window_bytes
    regions = {
        "context": [(w, w) for w in range(max(0, lo - 400), lo)],
        "suffix": [(hole_at + 1 + j, split + gap + j) for j in range(n_suffix)],
        "prefix": [(begin_at + 1 + j, lo + j) for j in range(split - lo)],
        "middle": [(begin_at + 1 + (split - lo) + j, split + j) for j in range(gap)],
    }
    for pairs in regions.values():  # the input token at each index is that window byte
        for idx, w in pairs:
            if int(ids[idx]) != window[w]:
                raise RuntimeError("input/window byte mismatch while mapping positions")
    return regions


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.run_dir / args.checkpoint
    if args.random_init:
        model = GPT(Config.from_file(ckpt / "model_config.yaml")).to(device).eval()
    else:
        model = load_model(ckpt, device)
    n_layer = len(model.transformer.h)
    layers = (
        sorted({max(0, n_layer // 4 - 1), n_layer // 2 - 1, 3 * n_layer // 4 - 1, n_layer - 1})
        if args.layers == "quartiles" else [int(x) for x in args.layers.split(",")]
    )
    captured: dict[int, torch.Tensor] = {}
    for li in layers:
        model.transformer.h[li].register_forward_hook(
            lambda _m, _i, out, li=li: captured.__setitem__(li, out[0] if isinstance(out, tuple) else out)
        )

    samples = build_samples(args)
    print(f"[probe] {len(samples)} samples, layers {layers}, random_init={args.random_init}", flush=True)
    feats = {li: [] for li in layers}
    meta = {k: [] for k in ("element", "bits_into", "mb_x", "mb_y", "region", "sample", "source", "hole_offset")}
    sources: dict[str, int] = {}
    for si, sample in enumerate(samples):
        labels = parser_state_labels(sample.window_bytes)
        regions = region_positions(sample)
        regions["context"] = regions["context"][-args.context_bytes:]
        chosen = []
        for r, pairs in regions.items():
            if len(pairs) > args.max_per_region:
                pairs = sorted(rng.sample(pairs, args.max_per_region))
            chosen += [(REGIONS.index(r), idx, w) for idx, w in pairs]
        if not chosen:
            continue
        ids = sample.teacher_input_ids.to(device)[None]
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(ids, region_ids=sample.teacher_region_ids.to(device)[None],
                  offset_ids=sample.teacher_offset_ids.to(device)[None])
        index = torch.tensor([idx for _, idx, _ in chosen], device=device)
        for li in layers:
            feats[li].append(captured[li][0].index_select(0, index).float().cpu().numpy().astype(np.float16))
        src = sources.setdefault(source_video_id(str(sample.h264_path)), len(sources))
        for r, _idx, w in chosen:
            k = w + 1  # state after window byte w
            meta["element"].append(labels.element[k]); meta["bits_into"].append(labels.bits_into[k])
            meta["mb_x"].append(labels.mb_x[k]); meta["mb_y"].append(labels.mb_y[k])
            meta["region"].append(r); meta["sample"].append(si); meta["source"].append(src)
            meta["hole_offset"].append(w - sample.split)
        if (si + 1) % 25 == 0:
            print(f"[probe] {si + 1}/{len(samples)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        layers=np.array(layers),
        **{f"X_{li}": np.concatenate(feats[li]) for li in layers},
        **{k: np.array(v, dtype=np.int32) for k, v in meta.items()},
    )
    print(f"[probe] saved {sum(len(f) for f in feats[layers[0]])} positions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
