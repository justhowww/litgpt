#!/usr/bin/env python3
"""Experiment E: regenerate from a cut to the end of its slice.

Cut a frame (one slice per frame) at a relative position, drop everything from
the cut to the end of that slice, and generate the remainder. The bytes after
the hole start at the next start code, where parser state resets, so any
complete legal slice reconnects: this is the JPEG-LM / AVC-LM continuation
setting, isolated from the mid-slice junction problem.

The prompt is the training FIM layout with an empty orphan
([context, BEGIN, prefix, HOLE, END] -> middle, EOS); holes ending exactly at
the frame end occur in training whenever ``split + gap == frame_hi``.

Conditions (every hole):

  unmasked      model, learned EOS, no syntax mask
  masked_old    model, litgpt/byte/h264_mask; EOS forced when the slice completes
  masked_new    model, syntax_mask.MaskStream; EOS forced when the slice completes
  random_legal  no model: uniform over bytes legal under the new mask
  concealment   FFmpeg concealment of the truncated slice (reference row per hole)

Model conditions run greedy once plus ``--samples-per-hole`` sampled draws;
random_legal runs the sampled draws only. Every repaired GOP is strictly decoded;
content metrics use the strict frames when valid and FFmpeg's lenient decode
otherwise (both are recorded). With ``--cross-check-masks`` each masked run
also steps the other mask and logs where the two disagree.

Holes are pinned: the first run writes ``--holes-file``; later runs (other
checkpoints, other runs) load it, so every model is scored on the same holes.

    python scripts/byte/eval/eval_slice_end.py RUN/latest \
        --train-split-file RUN/train_split.json \
        --holes-file eval_sets/slice_end_v1.json \
        --out-dir RUN/eval_slice_end/latest

Outputs (resumable; existing rows are skipped): references.jsonl,
generations.jsonl, summary.json, summary.csv, viz/*.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litgpt.byte.data import (  # noqa: E402
    BYTE_VOCAB_SIZE,
    IGNORE_INDEX,
    REGION_BRIDGE,
    SEQ_EOS_ID,
    ByteStreamWindowDataset,
    load_manifest_rows,
)
from litgpt.byte.megabyte_inference import (  # noqa: E402
    MegabyteInference,
    megabyte_max_new_bytes,
)
from litgpt.byte.reconstruction import _unwrap_model, image_psnr, image_ssim  # noqa: E402
from scripts.byte.eval import eval_ar_continuation as AR  # noqa: E402
from scripts.byte.eval import eval_fim_avclm as FIM  # noqa: E402
from scripts.byte.eval.helpers import slice_end as SE  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import (  # noqa: E402
    jsonable,
    load_model,
)

MODEL_CONDITIONS = ("unmasked", "masked_old", "masked_new")
CONDITIONS = (*MODEL_CONDITIONS, "random_legal")
HOLES_VERSION = 1


# ---------------------------------------------------------------------------
# holes
# ---------------------------------------------------------------------------
def build_dataset(rows, split_meta: dict, budget: int, max_remainder: int, seed: int):
    """The run's window layout; gap range opened up to admit every slice tail."""
    return ByteStreamWindowDataset(
        rows,
        max_seq_length=budget,
        min_frames=2,
        p_fim=1.0,
        fim_format=split_meta.get("fim_format", "psm"),
        fim_loss_scope=split_meta.get("fim_loss_scope", "full"),
        use_eos=bool(split_meta.get("use_eos", True)),
        fim_min_gap=1,
        fim_max_gap=max_remainder,
        frame_guard_bytes=0,
        window_unit=split_meta.get("window_unit", "gop"),
        resample_fim=False,
        seed=seed,
    )


def window_bytes(dataset, idx: int) -> bytes:
    sample = dataset.samples[idx]
    window, _region, _offset = dataset._window_tensors(sample, sample.h264_path.read_bytes())
    return bytes(window.tolist())


def select_holes(args, rows, split_meta: dict, budget: int) -> dict:
    """Pinned hole set: one hole per video per stratum, distinct videos per stratum."""
    dataset = build_dataset(rows, split_meta, budget, args.max_remainder, args.seed)
    by_video: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        by_video[str(sample.h264_path)].append(idx)
    cache: dict[int, tuple[bytes, list[tuple[int, int, str]]]] = {}

    def frames_of(idx: int):
        if idx not in cache:
            sample = dataset.samples[idx]
            data = sample.h264_path.read_bytes()
            window = window_bytes(dataset, idx)
            cands = [
                (lo, hi, SE.frame_type_at(window, lo))
                for lo, hi in dataset._fim_candidates(sample, data)
            ]
            cache[idx] = (window, cands)
        return cache[idx]

    holes, skips = [], Counter()
    videos = sorted(by_video)
    for ftype in args.frame_types:
        for cut in args.cut_positions:
            rng = random.Random(f"{args.seed}:{ftype}:{cut}")
            order = videos[:]
            rng.shuffle(order)
            chosen = 0
            for video in order:
                if chosen == args.holes_per_stratum:
                    break
                options = []
                for idx in by_video[video]:
                    window, cands = frames_of(idx)
                    for lo, hi, kind in cands:
                        if kind != ftype:
                            continue
                        body = hi - lo - args.header_guard_bytes
                        split = lo + args.header_guard_bytes + int(body * cut)
                        gap = hi - split
                        if body <= 1 or not 1 <= gap <= args.max_remainder:
                            skips[f"{ftype}:remainder_out_of_range"] += 1
                            continue
                        options.append((idx, lo, hi, split, gap))
                if not options:
                    skips[f"{ftype}:no_frame_in_video"] += 1
                    continue
                idx, lo, hi, split, gap = rng.choice(options)
                sample = dataset.samples[idx]
                window = cache[idx][0]
                holes.append(
                    {
                        "hole_id": len(holes),
                        "h264_path": str(sample.h264_path),
                        "start_nal": int(sample.start_nal),
                        "end_nal": int(sample.end_nal),
                        "frame_lo": lo,
                        "frame_hi": hi,
                        "split": split,
                        "gap": gap,
                        "frame_type": ftype,
                        "cut_pos": cut,
                        "target_frame_index": SE.frame_index_at(window, lo),
                        "window_frames": SE.frame_index_at(window, len(window)),
                    }
                )
                chosen += 1
            if chosen < args.holes_per_stratum:
                print(f"[holes] {ftype} cut={cut}: only {chosen} holes", flush=True)
    return {
        "version": HOLES_VERSION,
        "split": args.split,
        "seed": args.seed,
        "frame_types": args.frame_types,
        "cut_positions": args.cut_positions,
        "holes_per_stratum": args.holes_per_stratum,
        "header_guard_bytes": args.header_guard_bytes,
        "max_remainder": args.max_remainder,
        "window_budget_bytes": budget,
        "window_unit": split_meta.get("window_unit", "gop"),
        "skips": dict(skips),
        "holes": holes,
    }


def candidate_rows(args, manifest: Path, split_meta: dict) -> list[dict]:
    train_videos = {str(Path(v)) for v in split_meta.get("videos", [])}
    if not train_videos:
        raise SystemExit("train_split.json lists no training videos")
    rows = load_manifest_rows(manifest)
    if args.split == "val":
        rows = [r for r in rows if str(Path(r["h264_path"])) not in train_videos]
    else:
        rows = [r for r in rows if str(Path(r["h264_path"])) in train_videos]
    random.Random(args.seed).shuffle(rows)
    return rows[: args.num_videos]


def materialize(args, holes_doc: dict, manifest: Path, split_meta: dict, budget: int):
    """Hole dicts -> WindowFimSample with the exact training prompt layout."""
    wanted = {h["h264_path"] for h in holes_doc["holes"]}
    rows = [r for r in load_manifest_rows(manifest) if str(Path(r["h264_path"])) in wanted]
    dataset = build_dataset(rows, split_meta, budget, holes_doc["max_remainder"], args.seed)
    index = {
        (str(s.h264_path), int(s.start_nal), int(s.end_nal)): i
        for i, s in enumerate(dataset.samples)
    }
    ns = argparse.Namespace(use_eos=bool(split_meta.get("use_eos", True)))
    policy = FIM.EvalSamplePolicy(
        fim_loss_scope=split_meta.get("fim_loss_scope", "full"),
        window_unit=split_meta.get("window_unit", "gop"),
        hole_placement="training_random",
        hole_set="sampled",
        min_gap=1,
        max_gap=holes_doc["max_remainder"],
        frame_guard_bytes=0,
        corruption_eligibility_bytes=holes_doc["max_remainder"],
    )
    out = []
    for hole in holes_doc["holes"]:
        key = (hole["h264_path"], hole["start_nal"], hole["end_nal"])
        if key not in index:
            raise RuntimeError(f"pinned window not rebuilt: {key}")
        spec = (hole["frame_lo"], hole["frame_hi"], hole["split"], hole["gap"])
        sample, _verified, reason = FIM._materialize_fim_sample(
            ns, dataset, policy, FIM.HoleRequest(index[key], spec, None)
        )
        if sample is None:
            raise RuntimeError(f"hole {hole['hole_id']} not materialized: {reason}")
        if sample.bytes_after_hole:
            raise AssertionError("slice-end hole must have an empty in-frame suffix")
        out.append((hole, sample))
    return out


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
class ModelStepper:
    """Next-token logits for one FIM prompt (MEGABYTE or plain byte model)."""

    def __init__(self, raw, sample, device) -> None:
        self.raw, self.device = raw, device
        self.prompt = sample.prompt_ids.to(device).unsqueeze(0)
        regions = sample.prompt_region_ids.to(device).unsqueeze(0)
        offsets = sample.prompt_offset_ids.to(device).unsqueeze(0)
        self.prompt_len = self.prompt.size(1)
        supervised = (sample.teacher_labels != IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        start = int(supervised[0])
        self.megabyte = None
        if int(raw.config.byte_patch_size) > 1:
            self.max_new = megabyte_max_new_bytes(raw, self.prompt_len, supervision_start=start)
            self.megabyte = MegabyteInference(
                raw, self.prompt, regions, offsets, device, supervision_start=start
            )
        else:
            self.max_new = megabyte_max_new_bytes(raw, self.prompt_len)
            dtype = torch.bfloat16 if device.type == "cuda" else next(raw.parameters()).dtype
            raw.set_kv_cache(batch_size=1, max_seq_length=raw.max_seq_length, device=device, dtype=dtype)
            with self._autocast():
                self.logits = raw(
                    self.prompt,
                    input_pos=torch.arange(self.prompt_len, device=device, dtype=torch.long),
                    input_pos_maxp1=self.prompt_len,
                    region_ids=regions,
                    offset_ids=offsets,
                )[0, -1]

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        )

    def next_logits(self):
        return self.megabyte.next_logits()[0] if self.megabyte is not None else self.logits

    def append(self, token: int, step: int) -> None:
        if self.megabyte is not None:
            self.megabyte.append(token, REGION_BRIDGE, step + 1)
            return
        position = self.prompt_len + step
        t = lambda v: torch.tensor([[v]], device=self.device, dtype=torch.long)  # noqa: E731
        with self._autocast():
            self.logits = self.raw(
                t(token),
                input_pos=torch.tensor([position], device=self.device, dtype=torch.long),
                input_pos_maxp1=position + 1,
                region_ids=t(REGION_BRIDGE),
                offset_ids=t(step + 1),
            )[0, -1]

    def close(self) -> None:
        if self.megabyte is not None:
            self.megabyte.close()
        else:
            self.raw.clear_kv_cache()


@torch.inference_mode()
def generate_model(raw, sample, device, *, condition, greedy, args, seed) -> dict:
    temperature, top_k, top_p = (0.0, 0, 1.0) if greedy else (args.temperature, args.top_k, args.top_p)
    mask = cross = None
    if condition != "unmasked":
        kind = "old" if condition == "masked_old" else "new"
        mask = SE.make_mask(kind, sample.bytes_before_hole)
        shadow = (
            SE.make_mask("new" if kind == "old" else "old", sample.bytes_before_hole)
            if args.cross_check_masks
            else None
        )
        cross = SE.CrossCheck(shadow)
    torch.manual_seed(seed)
    stepper = ModelStepper(raw, sample, device)
    max_new = min(args.max_gen_bytes, stepper.max_new)
    out: list[int] = []
    stop = "budget"
    eos_prob_at_end = None
    rejected = 0
    mass = []
    started = time.perf_counter()
    try:
        for step in range(max_new):
            logits = stepper.next_logits()
            with_eos = torch.cat((logits[:BYTE_VOCAB_SIZE], logits[SEQ_EOS_ID : SEQ_EOS_ID + 1])).float()
            if mask is not None:
                if mask.slice_complete():
                    eos_prob_at_end = float(F.softmax(with_eos, dim=-1)[-1])
                    stop = "slice_end"
                    break
                allowed = mask.allowed()
                cross.compare(step, allowed, mask.where())
                if not any(allowed):
                    stop = "mask_boxed_in"
                    break
                allowed_t = torch.tensor(allowed, dtype=torch.bool, device=logits.device)
                byte_logits = logits[:BYTE_VOCAB_SIZE].float().clone()
                mass.append(
                    float(torch.exp(torch.logsumexp(byte_logits[allowed_t], 0) - torch.logsumexp(byte_logits, 0)))
                )
                if not allowed[int(byte_logits.argmax())]:
                    rejected += 1
                byte_logits[~allowed_t] = float("-inf")
                token = FIM._sample_token(byte_logits, temperature, top_k, top_p)
                try:
                    mask.advance(token)
                except SE.MaskRejected as exc:
                    stop = f"mask_rejected:{str(exc)[:80]}"
                    break
                cross.advance(token)
            else:
                token = FIM._sample_token(with_eos, temperature, top_k, top_p)
                if token == BYTE_VOCAB_SIZE:
                    stop = "eos"
                    break
            out.append(token)
            if step == max_new - 1:
                break
            stepper.append(token, step)
        if stop == "budget" and mask is not None and mask.slice_complete():
            stop = "slice_end"
    finally:
        stepper.close()
    return {
        "generated": bytes(out),
        "stop_reason": stop,
        "eos_prob_at_slice_end": eos_prob_at_end,
        "mask_argmax_rejected": rejected,
        "mask_allowed_mass_mean": statistics.fmean(mass) if mass else None,
        "cross_check": cross.report() if cross is not None and cross.steps else None,
        "seconds": time.perf_counter() - started,
    }


def generate_random(sample, *, args, seed) -> dict:
    started = time.perf_counter()
    res = SE.random_legal_completion(
        sample.bytes_before_hole, random.Random(seed), max_bytes=args.random_max_bytes
    )
    return {
        "generated": res.data,
        "stop_reason": res.stop_reason,
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# decoding and scoring
# ---------------------------------------------------------------------------
def _psnr(a, b) -> float:
    value = image_psnr(a, b)
    return AR.PSNR_PERFECT_CAP if value == float("inf") else value


def frame_metrics(frames, gt_frames, t: int) -> dict:
    out: dict[str, Any] = {"decoded_frames": len(frames)}
    if t < len(frames) and frames[t].shape == gt_frames[t].shape:
        out["target_psnr"] = _psnr(gt_frames[t], frames[t])
        out["target_ssim"] = image_ssim(gt_frames[t], frames[t])
        if t > 0:
            out["target_vs_prev_gt_psnr"] = _psnr(gt_frames[t - 1], frames[t])
    drift = [
        _psnr(gt_frames[k], frames[k])
        for k in range(t + 1, min(len(frames), len(gt_frames)))
        if frames[k].shape == gt_frames[k].shape
    ]
    if drift:
        out["drift_psnr_mean"] = statistics.fmean(drift)
        out["drift_psnr_last"] = drift[-1]
    return out


def decode_and_score(args, stream: bytes, gt_frames, t: int) -> tuple[dict, list]:
    frames, status, _info = AR.decode_h264(stream, args, strict=True)
    strict_valid = status == "decoded" and len(frames) == len(gt_frames)
    row: dict[str, Any] = {"strict_status": status, "strict_valid": strict_valid}
    if not strict_valid:
        frames, lenient_status, _ = AR.decode_h264(
            stream, args, strict=False, keep_partial_on_error=True
        )
        row["lenient_status"] = lenient_status
    row.update(frame_metrics(frames, gt_frames, t))
    return row, frames


def to_uint8(frame):
    return (frame.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()


def save_viz(path: Path, panels: list[tuple[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    images = [(name, Image.fromarray(to_uint8(f))) for name, f in panels if f is not None]
    if not images:
        return
    w, h = images[0][1].size
    canvas = Image.new("RGB", (w * len(images), h + 14), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, img) in enumerate(images):
        canvas.paste(img.resize((w, h)), (i * w, 14))
        draw.text((i * w + 2, 1), name, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return statistics.fmean(vals) if vals else None


def _rate(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else None


def summarize(out_dir: Path) -> list[dict]:
    refs = {r["hole_id"]: r for r in read_jsonl(out_dir / "references.jsonl") if r.get("ok")}
    gens = [g for g in read_jsonl(out_dir / "generations.jsonl") if g["hole_id"] in refs]
    for g in gens:
        conceal = refs[g["hole_id"]].get("concealment", {}).get("target_psnr")
        if g.get("target_psnr") is not None and conceal is not None:
            g["psnr_lift_vs_concealment"] = g["target_psnr"] - conceal
        drift_c = refs[g["hole_id"]].get("concealment", {}).get("drift_psnr_mean")
        if g.get("drift_psnr_mean") is not None and drift_c is not None:
            g["drift_lift_vs_concealment"] = g["drift_psnr_mean"] - drift_c
        g["gen_len_ratio"] = g["gen_len"] / max(1, refs[g["hole_id"]]["gap"])

    def row_for(name: dict, rows: list[dict]) -> dict:
        lifts = [r["psnr_lift_vs_concealment"] for r in rows if r.get("psnr_lift_vs_concealment") is not None]
        cross = [r["cross_check"] for r in rows if r.get("cross_check")]
        return {
            **name,
            "n": len(rows),
            "strict_valid_rate": _rate(rows, lambda r: r.get("strict_valid")),
            "all_legal_rate": _rate(rows, lambda r: r.get("all_legal")),
            "ends_at_slice_end_rate": _rate(rows, lambda r: r.get("ends_at_slice_end")),
            "stop_reasons": dict(Counter(r["stop_reason"].split(":")[0] for r in rows)),
            "gen_len_ratio_median": statistics.median([r["gen_len_ratio"] for r in rows]) if rows else None,
            "eos_prob_at_slice_end_mean": _mean(rows, "eos_prob_at_slice_end"),
            "mask_allowed_mass_mean": _mean(rows, "mask_allowed_mass_mean"),
            "target_psnr_mean": _mean(rows, "target_psnr"),
            "target_ssim_mean": _mean(rows, "target_ssim"),
            "target_psnr_strict_mean": _mean([r for r in rows if r.get("strict_valid")], "target_psnr"),
            "target_vs_prev_gt_psnr_mean": _mean(rows, "target_vs_prev_gt_psnr"),
            "drift_psnr_mean": _mean(rows, "drift_psnr_mean"),
            "psnr_lift_vs_concealment_mean": statistics.fmean(lifts) if lifts else None,
            "beats_concealment_rate": (sum(1 for x in lifts if x > 0) / len(lifts)) if lifts else None,
            "drift_lift_vs_concealment_mean": _mean(rows, "drift_lift_vs_concealment"),
            "cross_check_disagree_step_rate": (
                sum(c["disagree_steps"] for c in cross) / max(1, sum(c["steps"] for c in cross))
                if cross else None
            ),
            "cross_check_runs_with_disagreement": sum(1 for c in cross if c["disagree_steps"]) if cross else None,
            "seconds_mean": _mean(rows, "seconds"),
        }

    table = []
    groups: dict[tuple, list] = defaultdict(list)
    for g in gens:
        groups[(g["condition"], g["decoding"], g["frame_type"], "all")].append(g)
        groups[(g["condition"], g["decoding"], g["frame_type"], g["cut_pos"])].append(g)
    for (cond, dec, ftype, cut), rows in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        table.append(row_for({"condition": cond, "decoding": dec, "frame_type": ftype, "cut_pos": cut}, rows))
    # Concealment and GT reference rows on the same holes.
    ref_groups: dict[tuple, list] = defaultdict(list)
    for r in refs.values():
        ref_groups[(r["frame_type"], "all")].append(r)
        ref_groups[(r["frame_type"], r["cut_pos"])].append(r)
    for (ftype, cut), rows in sorted(ref_groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        conceal = [r["concealment"] for r in rows]
        table.append(
            {
                "condition": "concealment",
                "decoding": "-",
                "frame_type": ftype,
                "cut_pos": cut,
                "n": len(rows),
                "target_psnr_mean": _mean(conceal, "target_psnr"),
                "target_ssim_mean": _mean(conceal, "target_ssim"),
                "target_vs_prev_gt_psnr_mean": _mean(conceal, "target_vs_prev_gt_psnr"),
                "drift_psnr_mean": _mean(conceal, "drift_psnr_mean"),
                "gt_target_vs_prev_gt_psnr_mean": _mean(rows, "gt_target_vs_prev_gt_psnr"),
                "gap_bytes_median": statistics.median([r["gap"] for r in rows]),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(jsonable(table), indent=1) + "\n", encoding="utf-8")
    fields = sorted({k for row in table for k in row})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()})
    return table


def print_table(table: list[dict]) -> None:
    print(f"{'condition':13s} {'dec':6s} {'type':4s} {'n':>4s} {'strict':>7s} {'endOK':>6s} "
          f"{'PSNR':>6s} {'lift':>6s} {'win':>5s} {'drift':>6s} {'len':>5s}")
    for r in table:
        if r["cut_pos"] != "all":
            continue
        f = lambda k, w=6, p=2: (f"{r[k]:{w}.{p}f}" if isinstance(r.get(k), (int, float)) else f"{'-':>{w}s}")  # noqa: E731
        print(f"{r['condition']:13s} {r['decoding']:6s} {r['frame_type']:4s} {r['n']:4d} "
              f"{f('strict_valid_rate', 7)} {f('ends_at_slice_end_rate')} {f('target_psnr_mean')} "
              f"{f('psnr_lift_vs_concealment_mean')} {f('beats_concealment_rate', 5)} "
              f"{f('drift_psnr_mean')} {f('gen_len_ratio_median', 5)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row)) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, nargs="?", help="checkpoint dir; omit with --summarize-only")
    ap.add_argument("--train-split-file", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None, help="default: manifest recorded in the split file")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--holes-file", type=Path, required=True, help="pinned holes; created if missing")
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--num-videos", type=int, default=600, help="candidate videos when creating holes")
    ap.add_argument("--holes-per-stratum", type=int, default=20)
    ap.add_argument("--frame-types", nargs="+", default=["idr", "p"], choices=("idr", "p"))
    ap.add_argument("--cut-positions", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    ap.add_argument("--header-guard-bytes", type=int, default=4)
    ap.add_argument("--max-remainder", type=int, default=1400, help="largest regenerated tail (training max gap)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--samples-per-hole", type=int, default=3)
    ap.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-gen-bytes", type=int, default=4096)
    ap.add_argument("--random-max-bytes", type=int, default=200000)
    ap.add_argument("--cross-check-masks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit-holes", type=int, default=0, help="smoke test: first N holes only")
    ap.add_argument("--num-visualizations", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--ffmpeg-binary", default="ffmpeg")
    ap.add_argument("--timeout-sec", type=int, default=60)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only and args.checkpoint is None:
        ap.error("checkpoint is required unless --summarize-only")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print_table(summarize(args.out_dir))
        return

    split_meta = json.loads(args.train_split_file.read_text(encoding="utf-8"))
    manifest = args.manifest or Path(split_meta["manifest"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.dtype == "bf16" and device.type == "cuda") else torch.float32
    needs_model = any(c in MODEL_CONDITIONS for c in args.conditions)
    model = load_model(args.checkpoint, device, dtype=dtype)
    raw = _unwrap_model(model)
    raw.eval()
    budget = int(raw.config.block_size) * int(raw.config.byte_patch_size) - 1
    if not needs_model:
        del model
        raw = None

    if args.holes_file.exists():
        holes_doc = json.loads(args.holes_file.read_text(encoding="utf-8"))
        print(f"[holes] loaded {len(holes_doc['holes'])} pinned holes from {args.holes_file}", flush=True)
        if holes_doc["window_budget_bytes"] != budget:
            print(f"[holes] WARNING: holes built for a {holes_doc['window_budget_bytes']}-byte window, "
                  f"model budget is {budget}", flush=True)
    else:
        rows = candidate_rows(args, manifest, split_meta)
        holes_doc = select_holes(args, rows, split_meta, budget)
        args.holes_file.parent.mkdir(parents=True, exist_ok=True)
        args.holes_file.write_text(json.dumps(holes_doc, indent=1) + "\n", encoding="utf-8")
        print(f"[holes] wrote {len(holes_doc['holes'])} holes to {args.holes_file}; skips={holes_doc['skips']}",
              flush=True)
    samples = materialize(args, holes_doc, manifest, split_meta, budget)
    if args.limit_holes:
        samples = samples[: args.limit_holes]
    (args.out_dir / "config.json").write_text(
        json.dumps(jsonable({**vars(args), "window_budget_bytes": budget, "device": str(device)}), indent=2) + "\n",
        encoding="utf-8",
    )

    refs_path = args.out_dir / "references.jsonl"
    gens_path = args.out_dir / "generations.jsonl"
    refs_done = {r["hole_id"]: r for r in read_jsonl(refs_path)}
    gens_done = {(g["hole_id"], g["condition"], g["decoding"], g["sample_idx"]) for g in read_jsonl(gens_path)}
    viz_budget = args.num_visualizations
    started = time.perf_counter()

    for n, (hole, sample) in enumerate(samples):
        hid = hole["hole_id"]
        t = hole["target_frame_index"]
        window = sample.window_bytes
        prefix, tail = window[: sample.split], window[sample.frame_hi :]
        gt_frames, gt_status, _ = AR.decode_h264(window, args, strict=True)
        if hid not in refs_done:
            ref = {k: hole[k] for k in ("hole_id", "frame_type", "cut_pos", "gap", "target_frame_index")}
            ref["ok"] = gt_status == "decoded" and len(gt_frames) == hole["window_frames"]
            ref["gt_status"] = gt_status
            if ref["ok"]:
                ref["concealment"], _ = decode_and_score(args, prefix + tail, gt_frames, t)
                if t > 0:
                    ref["gt_target_vs_prev_gt_psnr"] = _psnr(gt_frames[t - 1], gt_frames[t])
            append_jsonl(refs_path, ref)
            refs_done[hid] = ref
        if not refs_done[hid]["ok"]:
            print(f"[hole {hid}] skipped: GT window does not strictly decode ({gt_status})", flush=True)
            continue

        viz_frames: dict[str, Any] = {}
        for condition in args.conditions:
            runs = [("greedy", 0)] if (args.greedy and condition in MODEL_CONDITIONS) else []
            runs += [("sample", i) for i in range(args.samples_per_hole)]
            for decoding, sample_idx in runs:
                key = (hid, condition, decoding, sample_idx)
                if key in gens_done:
                    continue
                seed = (args.seed * 1_000_003 + hid * 1_009 + sample_idx) & 0x7FFFFFFF
                if condition == "random_legal":
                    gen = generate_random(sample, args=args, seed=seed)
                else:
                    gen = generate_model(
                        raw, sample, device, condition=condition,
                        greedy=decoding == "greedy", args=args, seed=seed,
                    )
                data = gen.pop("generated")
                row = {
                    "hole_id": hid,
                    "frame_type": hole["frame_type"],
                    "cut_pos": hole["cut_pos"],
                    "condition": condition,
                    "decoding": decoding,
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "gen_len": len(data),
                    "gap": hole["gap"],
                    **gen,
                    **SE.replay_legality(prefix, data),
                }
                scored, frames = decode_and_score(args, prefix + data + tail, gt_frames, t)
                row.update(scored)
                row["generated_hex"] = data.hex()
                append_jsonl(gens_path, row)
                gens_done.add(key)
                if sample_idx == 0 and decoding == "sample" and t < len(frames):
                    viz_frames[condition] = frames[t]
        if viz_budget > 0 and viz_frames:
            conceal_frames = AR.decode_h264(prefix + tail, args, strict=False, keep_partial_on_error=True)[0]
            panels = [("GT", gt_frames[t]), ("concealment", conceal_frames[t] if t < len(conceal_frames) else None)]
            panels += [(c, viz_frames.get(c)) for c in args.conditions]
            save_viz(args.out_dir / "viz" / f"hole{hid:03d}_{hole['frame_type']}_cut{hole['cut_pos']}.png", panels)
            viz_budget -= 1
        elapsed = time.perf_counter() - started
        print(f"[{n + 1}/{len(samples)}] hole {hid} {hole['frame_type']} cut={hole['cut_pos']} "
              f"gap={hole['gap']}B elapsed={elapsed / 60:.1f} min", flush=True)

    print_table(summarize(args.out_dir))


if __name__ == "__main__":
    main()
