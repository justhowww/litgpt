"""AVC-LM window-FIM span reconstruction evaluation.

This is the Phase-2 FIM-side probe for the AVC-LM per-macroblock-slice corpus.
It evaluates the exact window-FIM training format:

    context, FIM_BEGIN, prefix, FIM_HOLE, orphan, FIM_END -> missing span [, EOS]

The default stopping mode is learned EOS because Phase 2 is trained with
``use_eos=True``. The optional oracle-length mode is a diagnostic that separates
byte-content reconstruction from learned termination.

This is deliberately not the BSCV artifact-repair benchmark. On this corpus the
hole can span many tiny per-MB NALs, so the result should be read as "can the
AVC-LM FIM objective reconstruct held-out spans under its own data layout?", not
as a one-slice-per-frame corruption-recovery product metric.

Usage:
    python scripts/byte/eval/eval_fim_avclm.py MANIFEST \
        --nal-index-path DATA/nal_index.sqlite \
        --checkpoint-dirs RUN/step-XXXX [...] \
        --train-split-file RUN/train_split.json \
        --out-dir RUN/eval_fim/train/step-XXXX
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte.data import (  # noqa: E402
    BYTE_VOCAB_SIZE,
    FIM_FORMATS,
    IGNORE_INDEX,
    REGION_BRIDGE,
    SEQ_EOS_ID,
    ByteStreamWindowDataset,
    default_nal_index_path,
    load_manifest_rows,
    load_nal_index,
)
from litgpt.byte.reconstruction import _unwrap_model  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import (  # noqa: E402
    jsonable,
    load_model,
)


@dataclass(frozen=True)
class WindowFimSample:
    sample_index: int
    h264_path: Path
    start_nal: int
    end_nal: int
    frame_lo: int
    frame_hi: int
    split: int
    gap: int
    prompt_ids: Tensor
    prompt_region_ids: Tensor
    prompt_offset_ids: Tensor
    teacher_input_ids: Tensor
    teacher_region_ids: Tensor
    teacher_offset_ids: Tensor
    teacher_labels: Tensor
    target_bytes: bytes
    window_bytes: bytes

    @property
    def target_length(self) -> int:
        return len(self.target_bytes)

    @property
    def gt_truncated_stream(self) -> bytes:
        return self.window_bytes[: self.frame_hi]

    def repaired_stream(self, generated: bytes) -> bytes:
        return (
            self.window_bytes[: self.split]
            + generated
            + self.window_bytes[self.split + self.gap : self.frame_hi]
        )


@dataclass(frozen=True)
class GenerationResult:
    data: bytes
    eos_stopped: bool
    stop_reason: str
    steps: int


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
    parser.add_argument("--train-split-file", type=Path, default=None)
    parser.add_argument(
        "--eval-split",
        choices=("train", "val", "all"),
        default="train",
        help=(
            "Subset to evaluate when --train-split-file is absent. With a split file, "
            "'train' uses the dumped train windows exactly."
        ),
    )
    parser.add_argument("--num-clips", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-window-bytes", type=int, default=16384)
    parser.add_argument("--window-min-frames", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--split-by-video", action="store_true")
    parser.add_argument("--fim-format", choices=FIM_FORMATS, default="psm")
    parser.add_argument(
        "--use-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use learned SEQ_EOS termination. Phase 2 defaults to true.",
    )
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument(
        "--stop-modes",
        nargs="+",
        choices=("learned_eos", "oracle_len"),
        default=["learned_eos", "oracle_len"],
        help="Run learned-EOS generation, oracle-length generation, or both.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-gen-multiple", type=float, default=2.0)
    parser.add_argument("--max-gen-extra", type=int, default=16)
    parser.add_argument(
        "--save-streams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write GT-truncated and repaired H.264 streams for each sample.",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Also run ffmpeg decode on GT-truncated and repaired streams.",
    )
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--timeout-sec", type=int, default=30)
    return parser.parse_args()


def _load_train_split(path: Path) -> tuple[set[tuple[str, int, int]], set[str]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    windows = {
        (str(Path(w["h264_path"])), int(w["start_nal"]), int(w["end_nal"]))
        for w in split.get("windows", [])
    }
    videos = {str(Path(v)) for v in split.get("videos", [])}
    return windows, videos


def _split_indices(dataset: ByteStreamWindowDataset, args: argparse.Namespace) -> list[int]:
    all_indices = list(range(len(dataset)))
    if args.eval_split == "all":
        return all_indices
    if args.split_by_video:
        videos = sorted({str(s.h264_path) for s in dataset.samples})
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(videos), generator=generator).tolist()
        n_val = max(1, int(len(videos) * args.val_fraction))
        val_videos = {videos[i] for i in perm[:n_val]}
        if args.eval_split == "val":
            return [i for i, s in enumerate(dataset.samples) if str(s.h264_path) in val_videos]
        return [i for i, s in enumerate(dataset.samples) if str(s.h264_path) not in val_videos]

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Need at least two usable windows for train/val split")
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    return perm[train_size:] if args.eval_split == "val" else perm[:train_size]


def build_eval_samples(args: argparse.Namespace) -> list[WindowFimSample]:
    rows = load_manifest_rows(
        args.manifest,
        max_rows=args.max_manifest_rows or None,
        report_progress=True,
    )
    index_path = args.nal_index_path or default_nal_index_path(args.manifest)
    nal_index = load_nal_index(index_path, args.manifest, rows) if index_path.is_file() else None
    if args.nal_index_path is not None and nal_index is None:
        raise FileNotFoundError(f"NAL index does not exist: {index_path}")

    dataset = ByteStreamWindowDataset(
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

    if args.train_split_file is not None:
        train_windows, train_videos = _load_train_split(args.train_split_file)
        if train_windows:
            indices = [
                i
                for i, s in enumerate(dataset.samples)
                if (str(s.h264_path), int(s.start_nal), int(s.end_nal)) in train_windows
            ]
            split_desc = f"{len(indices)} exact train windows from {args.train_split_file}"
        else:
            indices = [
                i for i, s in enumerate(dataset.samples) if str(s.h264_path) in train_videos
            ]
            split_desc = f"{len(indices)} train videos from {args.train_split_file}"
        if not indices:
            raise RuntimeError(
                "No dataset windows matched -- check manifest/max rows against train_split.json"
            )
        print(f"train-split filter: {split_desc}", flush=True)
    else:
        indices = _split_indices(dataset, args)
        print(
            f"split filter: eval_split={args.eval_split} selected {len(indices)}/{len(dataset)} windows",
            flush=True,
        )

    selected: list[WindowFimSample] = []
    for idx in indices:
        item = dataset[idx]
        if item["sample_meta"].get("task") != "fim":
            continue
        labels: Tensor = item["labels"]
        supervised = labels != IGNORE_INDEX
        if not bool(supervised.any()):
            continue
        target_ids = labels[supervised].tolist()
        if args.use_eos:
            if not target_ids or target_ids[-1] != SEQ_EOS_ID:
                continue
            target_ids = target_ids[:-1]
        if not target_ids or any(t < 0 or t >= BYTE_VOCAB_SIZE for t in target_ids):
            continue

        first_supervised = int(supervised.nonzero()[0])
        prompt_end = first_supervised + 1
        meta = item["sample_meta"]
        window_bytes = bytes(dataset.ar_item(idx)["labels"].tolist())
        selected.append(
            WindowFimSample(
                sample_index=idx,
                h264_path=Path(meta["h264_path"]),
                start_nal=int(meta["start_nal"]),
                end_nal=int(meta["end_nal"]),
                frame_lo=int(meta["frame_lo"]),
                frame_hi=int(meta["frame_hi"]),
                split=int(meta["fim_split"]),
                gap=int(meta["fim_gap"]),
                prompt_ids=item["input_ids"][:prompt_end].clone(),
                prompt_region_ids=item["region_ids"][:prompt_end].clone(),
                prompt_offset_ids=item["offset_ids"][:prompt_end].clone(),
                teacher_input_ids=item["input_ids"].clone(),
                teacher_region_ids=item["region_ids"].clone(),
                teacher_offset_ids=item["offset_ids"].clone(),
                teacher_labels=item["labels"].clone(),
                target_bytes=bytes(target_ids),
                window_bytes=window_bytes,
            )
        )
        if len(selected) >= args.num_clips:
            break

    if not selected:
        raise RuntimeError(
            "No FIM samples selected. Lower gap/guard constraints or check that window-FIM is reachable."
        )
    print(f"Selected {len(selected)} FIM AVC-LM samples", flush=True)
    return selected


def _sample_token(logits: Tensor, temperature: float, top_k: int, top_p: float) -> int:
    if temperature <= 0 and top_k <= 0 and top_p >= 1.0:
        return int(logits.argmax(dim=-1))
    logits = logits.float()
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove_sorted = cumulative > top_p
        remove_sorted[0] = False
        remove = torch.zeros_like(remove_sorted).scatter(0, sorted_idx, remove_sorted)
        logits = torch.where(remove, torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


@torch.inference_mode()
def generate_span(
    model: nn.Module,
    sample: WindowFimSample,
    device: torch.device,
    *,
    stop_mode: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_gen_multiple: float,
    max_gen_extra: int,
) -> GenerationResult | None:
    raw = _unwrap_model(model)
    raw.eval()
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0)
    prompt_len = prompt.size(1)
    if prompt_len >= raw.max_seq_length:
        return None
    if stop_mode == "oracle_len":
        max_new = sample.target_length
    elif stop_mode == "learned_eos":
        max_new = min(
            max(sample.target_length + max_gen_extra, int(sample.target_length * max_gen_multiple)),
            raw.max_seq_length - prompt_len + 1,
        )
    else:
        raise ValueError(f"unknown stop_mode: {stop_mode}")
    if max_new <= 0:
        return None

    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw.parameters()).dtype
    raw.set_kv_cache(
        batch_size=1,
        max_seq_length=raw.max_seq_length,
        device=device,
        dtype=cache_dtype,
    )
    generated: list[int] = []
    eos_stopped = False
    stop_reason = "budget"
    try:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = raw(
                prompt,
                input_pos=torch.arange(prompt_len, device=device, dtype=torch.long),
                input_pos_maxp1=prompt_len,
                region_ids=region_ids,
                offset_ids=offset_ids,
            )
        for generated_idx in range(max_new):
            next_logits = logits[0, -1]
            if stop_mode == "learned_eos":
                allowed = torch.cat(
                    (next_logits[:BYTE_VOCAB_SIZE], next_logits[SEQ_EOS_ID : SEQ_EOS_ID + 1])
                )
                token = _sample_token(allowed, temperature, top_k, top_p)
                token = SEQ_EOS_ID if token == BYTE_VOCAB_SIZE else token
                if token == SEQ_EOS_ID:
                    eos_stopped = True
                    stop_reason = "eos"
                    break
            else:
                token = _sample_token(next_logits[:BYTE_VOCAB_SIZE], temperature, top_k, top_p)

            if token < 0 or token >= BYTE_VOCAB_SIZE:
                stop_reason = f"invalid_token_{token}"
                break
            generated.append(token)
            if generated_idx == max_new - 1:
                stop_reason = "oracle_len" if stop_mode == "oracle_len" else "budget"
                break

            position = prompt_len + generated_idx
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = raw(
                    torch.tensor([[token]], device=device, dtype=torch.long),
                    input_pos=torch.tensor([position], device=device, dtype=torch.long),
                    input_pos_maxp1=position + 1,
                    region_ids=torch.tensor([[REGION_BRIDGE]], device=device, dtype=torch.long),
                    offset_ids=torch.tensor([[generated_idx + 1]], device=device, dtype=torch.long),
                )
    finally:
        raw.clear_kv_cache()
    return GenerationResult(bytes(generated), eos_stopped, stop_reason, len(generated))


@torch.inference_mode()
def teacher_forced_span_metrics(
    model: nn.Module,
    sample: WindowFimSample,
    device: torch.device,
) -> dict[str, Any] | None:
    """Teacher-forced accuracy on the exact same FIM target span used for generation."""
    raw = _unwrap_model(model)
    raw.eval()
    if sample.teacher_input_ids.numel() > raw.max_seq_length:
        return None
    idx = sample.teacher_input_ids.to(device).unsqueeze(0)
    labels = sample.teacher_labels.to(device).unsqueeze(0)
    region_ids = sample.teacher_region_ids.to(device).unsqueeze(0)
    offset_ids = sample.teacher_offset_ids.to(device).unsqueeze(0)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = raw(idx, region_ids=region_ids, offset_ids=offset_ids)
    supervised = labels != IGNORE_INDEX
    supervised_labels = labels[supervised]
    if supervised_labels.numel() == 0:
        return None
    pred = logits.argmax(dim=-1)[supervised]
    byte_mask = supervised_labels < BYTE_VOCAB_SIZE
    eos_mask = supervised_labels == SEQ_EOS_ID
    byte_total = int(byte_mask.sum())
    eos_total = int(eos_mask.sum())
    byte_correct = int((pred[byte_mask] == supervised_labels[byte_mask]).sum()) if byte_total else 0
    eos_correct = int((pred[eos_mask] == SEQ_EOS_ID).sum()) if eos_total else 0
    exact_span = bool(byte_total == sample.target_length and byte_correct == byte_total)
    exact_tail = bool(int((pred == supervised_labels).sum()) == int(supervised_labels.numel()))
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    return {
        "tf_loss": float(loss),
        "tf_target_tokens": int(supervised_labels.numel()),
        "tf_byte_total": byte_total,
        "tf_byte_correct": byte_correct,
        "tf_byte_acc": byte_correct / max(byte_total, 1),
        "tf_eos_total": eos_total,
        "tf_eos_correct": eos_correct,
        "tf_eos_acc": eos_correct / max(eos_total, 1) if eos_total else None,
        "tf_exact_span_bytes": exact_span,
        "tf_exact_tail_with_eos": exact_tail,
        "tf_argmax_ids": [int(x) for x in pred.detach().cpu().tolist()],
    }


def first_mismatch_diagnostic(
    generated: bytes,
    target: bytes,
    tf_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare the first free-generation mismatch to the TF argmax.

    At the first mismatch position k, generated[:k] still equals target[:k], so
    free generation and teacher forcing should be conditioning on the same byte
    prefix. If TF predicts the GT byte at k but free generation emits something
    else, the generation path is not equivalent to the teacher-forced path.
    """
    limit = min(len(generated), len(target))
    mismatch_pos: int | None = None
    for i in range(limit):
        if generated[i] != target[i]:
            mismatch_pos = i
            break
    if mismatch_pos is None and len(generated) != len(target):
        mismatch_pos = limit
    if mismatch_pos is None:
        return {
            "first_mismatch_pos": None,
            "first_mismatch_kind": "none",
            "tf_correct_at_first_mismatch": None,
            "free_matches_tf_argmax_at_first_mismatch": None,
        }

    if mismatch_pos >= len(generated):
        kind = "generated_too_short"
    elif mismatch_pos >= len(target):
        kind = "generated_too_long"
    else:
        kind = "byte_mismatch"
    gt_byte = target[mismatch_pos] if mismatch_pos < len(target) else None
    gen_byte = generated[mismatch_pos] if mismatch_pos < len(generated) else None
    tf_argmax_ids = list(tf_metrics.get("tf_argmax_ids", [])) if tf_metrics else []
    tf_argmax = tf_argmax_ids[mismatch_pos] if mismatch_pos < len(tf_argmax_ids) else None
    return {
        "first_mismatch_pos": mismatch_pos,
        "first_mismatch_kind": kind,
        "first_mismatch_gt_byte": gt_byte,
        "first_mismatch_gen_byte": gen_byte,
        "first_mismatch_tf_argmax": tf_argmax,
        "tf_correct_at_first_mismatch": (
            bool(tf_argmax == gt_byte) if gt_byte is not None and tf_argmax is not None else None
        ),
        "free_matches_tf_argmax_at_first_mismatch": (
            bool(tf_argmax == gen_byte) if gen_byte is not None and tf_argmax is not None else None
        ),
    }


def byte_accuracy(a: bytes, b: bytes) -> float:
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b))
    hits = sum(x == y for x, y in zip(a, b))
    return hits / denom


def parse_summary(stream: bytes) -> dict[str, Any]:
    try:
        parsed = HS.parse_stream(stream, parse_slice_data=True)
    except Exception as exc:  # defensive: parser diagnostics should not kill eval
        return {"parse_ok": False, "parse_error": type(exc).__name__, "nals": 0}
    statuses = [getattr(n.status, "name", str(n.status)) for n in parsed.nals]
    return {
        "parse_ok": all(s == "OK" for s in statuses),
        "parse_status_hist": {s: statuses.count(s) for s in sorted(set(statuses))},
        "nals": len(parsed.nals),
    }


def decode_ok(
    stream: bytes,
    ffmpeg_binary: str,
    timeout_sec: int,
    tmp_path: Path,
) -> tuple[bool, str]:
    tmp_path.write_bytes(stream)
    cmd = [
        ffmpeg_binary,
        "-v",
        "error",
        "-f",
        "h264",
        "-i",
        str(tmp_path),
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    return proc.returncode == 0, stderr[:1000]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(details)
    if n == 0:
        return {}
    out = {
        "n": n,
        "exact_match_rate": mean(1.0 if r["exact_match"] else 0.0 for r in details),
        "byte_acc_mean": mean(r["byte_acc"] for r in details),
        "length_match_rate": mean(1.0 if r["length_delta"] == 0 else 0.0 for r in details),
        "length_delta_mean": mean(r["length_delta"] for r in details),
        "eos_stop_rate": mean(1.0 if r["eos_stopped"] else 0.0 for r in details),
        "parse_ok_rate": mean(1.0 if r["model_parse_ok"] else 0.0 for r in details),
        "decode_ok_rate": (
            mean(1.0 if r.get("model_decode_ok") else 0.0 for r in details)
            if "model_decode_ok" in details[0]
            else None
        ),
    }
    tf_rows = [r for r in details if "tf_byte_acc" in r]
    if tf_rows:
        first_mismatch_rows = [
            r for r in tf_rows if r.get("tf_correct_at_first_mismatch") is not None
        ]
        out.update(
            {
                "tf_byte_acc_mean": mean(r["tf_byte_acc"] for r in tf_rows),
                "tf_exact_span_rate": mean(
                    1.0 if r["tf_exact_span_bytes"] else 0.0 for r in tf_rows
                ),
                "tf_exact_tail_with_eos_rate": mean(
                    1.0 if r["tf_exact_tail_with_eos"] else 0.0 for r in tf_rows
                ),
                "tf_eos_acc_mean": mean(
                    r["tf_eos_acc"] for r in tf_rows if r.get("tf_eos_acc") is not None
                ),
                "tf_loss_mean": mean(r["tf_loss"] for r in tf_rows),
            }
        )
        if first_mismatch_rows:
            out.update(
                {
                    "first_mismatch_tf_correct_rate": mean(
                        1.0 if r["tf_correct_at_first_mismatch"] else 0.0
                        for r in first_mismatch_rows
                    ),
                    "first_mismatch_free_matches_tf_rate": mean(
                        1.0 if r["free_matches_tf_argmax_at_first_mismatch"] else 0.0
                        for r in first_mismatch_rows
                    ),
                    "first_mismatch_pos_mean": mean(
                        r["first_mismatch_pos"] for r in first_mismatch_rows
                    ),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    samples = build_eval_samples(args)
    (args.out_dir / "config.json").write_text(
        json.dumps(jsonable(vars(args)), indent=2) + "\n", encoding="utf-8"
    )
    sample_manifest = [
        {
            "sample_id": i,
            "dataset_index": s.sample_index,
            "h264_path": str(s.h264_path),
            "start_nal": s.start_nal,
            "end_nal": s.end_nal,
            "frame_lo": s.frame_lo,
            "frame_hi": s.frame_hi,
            "split": s.split,
            "gap": s.gap,
            "target_length": s.target_length,
        }
        for i, s in enumerate(samples)
    ]
    (args.out_dir / "samples.json").write_text(
        json.dumps(jsonable(sample_manifest), indent=2) + "\n", encoding="utf-8"
    )

    metrics_path = args.out_dir / "metrics.jsonl"
    details_path = args.out_dir / "sample_details.jsonl"
    summaries: list[dict[str, Any]] = []

    for checkpoint_dir in args.checkpoint_dirs:
        model = load_model(checkpoint_dir, device)
        ckpt_name = checkpoint_dir.name
        stream_dir = args.out_dir / "streams" / ckpt_name
        if args.save_streams:
            stream_dir.mkdir(parents=True, exist_ok=True)
        for stop_mode in args.stop_modes:
            print(f"[{ckpt_name}] FIM AVC-LM stop_mode={stop_mode}", flush=True)
            rows: list[dict[str, Any]] = []
            for sample_id, sample in enumerate(samples):
                tf_metrics = teacher_forced_span_metrics(model, sample, device)
                result = generate_span(
                    model,
                    sample,
                    device,
                    stop_mode=stop_mode,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    max_gen_multiple=args.max_gen_multiple,
                    max_gen_extra=args.max_gen_extra,
                )
                if result is None:
                    row = {
                        "checkpoint": ckpt_name,
                        "stop_mode": stop_mode,
                        "sample_id": sample_id,
                        "dataset_index": sample.sample_index,
                        "error": "prompt_or_budget_too_long",
                    }
                    rows.append(row)
                    continue

                model_stream = sample.repaired_stream(result.data)
                gt_stream = sample.gt_truncated_stream
                gt_parse = parse_summary(gt_stream)
                model_parse = parse_summary(model_stream)
                row = {
                    "checkpoint": ckpt_name,
                    "stop_mode": stop_mode,
                    "sample_id": sample_id,
                    "dataset_index": sample.sample_index,
                    "h264_path": str(sample.h264_path),
                    "start_nal": sample.start_nal,
                    "end_nal": sample.end_nal,
                    "frame_lo": sample.frame_lo,
                    "frame_hi": sample.frame_hi,
                    "split": sample.split,
                    "gap": sample.gap,
                    "target_length": sample.target_length,
                    "generated_length": len(result.data),
                    "length_delta": len(result.data) - sample.target_length,
                    "eos_stopped": result.eos_stopped,
                    "stop_reason": result.stop_reason,
                    "byte_acc": byte_accuracy(result.data, sample.target_bytes),
                    "exact_match": result.data == sample.target_bytes,
                    "gt_parse_ok": gt_parse.get("parse_ok", False),
                    "model_parse_ok": model_parse.get("parse_ok", False),
                    "gt_parse": gt_parse,
                    "model_parse": model_parse,
                }
                row.update(first_mismatch_diagnostic(result.data, sample.target_bytes, tf_metrics))
                if tf_metrics is not None:
                    row.update(tf_metrics)
                if args.save_streams:
                    stem = f"sample_{sample_id:04d}_{stop_mode}"
                    gt_path = stream_dir / f"{stem}_gt.h264"
                    model_path = stream_dir / f"{stem}_model.h264"
                    missing_path = stream_dir / f"{stem}_missing_gt.bin"
                    gen_path = stream_dir / f"{stem}_missing_model.bin"
                    gt_path.write_bytes(gt_stream)
                    model_path.write_bytes(model_stream)
                    missing_path.write_bytes(sample.target_bytes)
                    gen_path.write_bytes(result.data)
                    row.update(
                        {
                            "gt_stream_path": str(gt_path),
                            "model_stream_path": str(model_path),
                            "gt_missing_path": str(missing_path),
                            "model_missing_path": str(gen_path),
                        }
                    )
                if args.decode:
                    tmp_gt = args.out_dir / f".tmp_{ckpt_name}_{stop_mode}_{sample_id}_gt.h264"
                    tmp_model = args.out_dir / f".tmp_{ckpt_name}_{stop_mode}_{sample_id}_model.h264"
                    gt_ok, gt_err = decode_ok(gt_stream, args.ffmpeg_binary, args.timeout_sec, tmp_gt)
                    model_ok, model_err = decode_ok(
                        model_stream, args.ffmpeg_binary, args.timeout_sec, tmp_model
                    )
                    tmp_gt.unlink(missing_ok=True)
                    tmp_model.unlink(missing_ok=True)
                    row.update(
                        {
                            "gt_decode_ok": gt_ok,
                            "gt_decode_error": gt_err,
                            "model_decode_ok": model_ok,
                            "model_decode_error": model_err,
                        }
                    )
                rows.append(row)
            write_jsonl(details_path, rows)
            summary = {
                "checkpoint": ckpt_name,
                "checkpoint_dir": str(checkpoint_dir),
                "mode": "fim_avclm",
                "stop_mode": stop_mode,
                **summarize([r for r in rows if "error" not in r]),
            }
            write_jsonl(metrics_path, [summary])
            summaries.append(summary)
            print(
                f"[{ckpt_name}/{stop_mode}] exact={summary.get('exact_match_rate', 0):.3f} "
                f"byte_acc={summary.get('byte_acc_mean', 0):.4f} "
                f"len_match={summary.get('length_match_rate', 0):.3f} "
                f"parse_ok={summary.get('parse_ok_rate', 0):.3f}",
                flush=True,
            )

    if summaries:
        fieldnames = sorted({k for row in summaries for k in row.keys()})
        with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(jsonable(row))


if __name__ == "__main__":
    main()
