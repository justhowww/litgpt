"""Per-macroblock residual length-consistency check for free-run generation.

Auxiliary to eval_stream_continuation.py. Under slice-max-mbs=1 each VCL NAL is one
macroblock ending byte-aligned at the next start code, which makes MB validity a LOCAL,
checkable property. For every MB the model generates we walk the CAVLC cascade

    mb_type -> coded_block_pattern -> coeff_token(s) -> levels / total_zeros / runs

and ask the one thing that is NOT enforced by construction: does the declared residual
terminate exactly at the byte-aligned rbsp_trailing boundary?

  * aligned            slice_data ends at the stop bit (trailing <= 8 bits) -> well-formed MB
  * underrun           slice_data ends early, leaving a large "trailing" span
                       (model emitted more slice bytes than its declared residual filled)
  * overrun            residual ran off the RBSP end (BitReaderError) -> slice too short
                       for the residual the model declared
  * illegal@<region>   VLC-table / out-of-range miss mid-decode (ValueError/KeyError/IndexError)

The counts of coeff_tokens-per-cbp and levels-per-coeff_token are TAUTOLOGICAL (the parser
reads exactly what each level declares), so they cannot mismatch; the informative signal is
the boundary delta (its sign = the direction of the mis-sizing) plus the stage of any
illegal codeword. Per-MB records go to mb_length_clips.jsonl; a verdict histogram + mean
signed boundary delta go to summary.json.

Usage:
    python scripts/byte/eval/mb_length_check.py MANIFEST --nal-index-path NAL.sqlite \
        --checkpoint-dirs RUN/step-XXXX --out-dir OUT \
        --num-clips 20 --prefix-frames 288 --cont-frames 144 --max-window-bytes 16384 \
        --temperature 1.0 [--top-k 50 --top-p 0.9]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte.data import (  # noqa: E402
    PARAMETER_SET_NAL_TYPES,
    REGION_META,
    REGION_TARGET,
    SLICE_BOS_ID,
    VCL_NAL_TYPES,
    bytes_to_ids,
    default_nal_index_path,
    load_manifest_rows,
    load_nal_index,
    parse_annexb_nals,
)
from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte.free_run_eval import _desync_info, free_run_rollout  # noqa: E402
from scripts.byte.eval.eval_checkpoints import load_model  # noqa: E402
from scripts.byte.eval.eval_stream_continuation import (  # noqa: E402
    model_max_gen,
    select_continuation_clips,
)

# A byte-aligned per-MB slice ends with rbsp_stop_one_bit + zero padding: at most 8 bits.
# A larger trailing span means slice_data stopped early -> residual underran the slice.
_MAX_VALID_TRAILING_BITS = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--nal-index-path", type=Path, default=None)
    p.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-clips", type=int, default=20)
    p.add_argument("--max-manifest-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix-frames", type=int, default=288)
    p.add_argument("--cont-frames", type=int, default=144)
    p.add_argument("--max-window-bytes", type=int, default=16384)
    p.add_argument("--max-gen-multiple", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=1.0, help="0 = greedy.")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    return p.parse_args()


def _prompt_from_stream(stream: bytes, device: torch.device):
    """(prompt_ids, region_ids, offset_ids) for an Annex-B stream, mirroring the training
    ByteStreamWindowDataset: SLICE_BOS prepended, region META for parameter sets else
    TARGET, per-NAL byte offset reset."""
    nals = parse_annexb_nals(stream)
    region_chunks, offset_chunks = [], []
    for nal in nals:
        length = nal.end - nal.start
        region = REGION_META if nal.nal_type in PARAMETER_SET_NAL_TYPES else REGION_TARGET
        region_chunks.append(torch.full((length,), region, dtype=torch.long))
        offset_chunks.append(torch.arange(length, dtype=torch.long))
    raw_region = torch.cat(region_chunks) if region_chunks else torch.empty(0, dtype=torch.long)
    raw_offset = torch.cat(offset_chunks) if offset_chunks else torch.empty(0, dtype=torch.long)
    prompt_ids = torch.cat((torch.tensor([SLICE_BOS_ID], dtype=torch.long), bytes_to_ids(stream)))
    region_ids = torch.cat((torch.tensor([REGION_TARGET], dtype=torch.long), raw_region))
    offset_ids = torch.cat((torch.tensor([0], dtype=torch.long), raw_offset))
    return (prompt_ids.to(device).unsqueeze(0),
            region_ids.to(device).unsqueeze(0),
            offset_ids.to(device).unsqueeze(0))


def _read_prefix(clip) -> bytes:
    data = clip.h264_path.read_bytes()
    nals = parse_annexb_nals(data)
    return b"".join(data[nals[i].start:nals[i].end] for i in range(clip.prefix_end_nal))


def _span_value(spans, name):
    for s in spans:
        if s.name == name:
            return s.value
    return None


def analyze_mb(nal) -> dict:
    """Per-MB length-consistency verdict from a parsed VCL NAL (= one macroblock)."""
    spans = nal.spans
    mb_type = _span_value(spans, "mb_type")
    cbp = _span_value(spans, "coded_block_pattern")

    coeff_tokens = [s for s in spans if s.name.endswith(".coeff_token")]
    total_coeffs = []
    zero_coeff_blocks = 0
    for s in coeff_tokens:
        v = s.value if isinstance(s.value, dict) else {}
        tc = v.get("total_coeff")
        total_coeffs.append(tc)
        if tc == 0:
            zero_coeff_blocks += 1  # NB: legal for a 4x4 sub-block; only suggestive

    trailing = next((s for s in spans if s.name == "rbsp_trailing_bits"), None)
    trailing_bits = (trailing.bit_end - trailing.bit_start) if trailing is not None else None

    if nal.status == HS.ParseStatus.DESYNC:
        region, _cat, reason_kind = _desync_info(nal)
        if reason_kind == "BitReaderError":
            verdict = "overrun"           # residual ran off the RBSP end
        else:
            verdict = f"illegal@{region}"  # VLC miss / out-of-range mid-decode
        delta = None
    elif nal.status == HS.ParseStatus.UNSUPPORTED:
        verdict, delta = "unsupported", None
    else:  # OK
        if trailing_bits is None:
            verdict, delta = "aligned", 0  # consumed to the exact bit-end, no padding span
        elif trailing_bits > _MAX_VALID_TRAILING_BITS:
            verdict = "underrun"           # slice_data ended early, big leftover
            delta = -(trailing_bits - _MAX_VALID_TRAILING_BITS)  # signed: negative = short
        else:
            verdict, delta = "aligned", 0

    return {
        "mb_type": mb_type,
        "cbp": cbp,
        "n_coeff_tokens": len(coeff_tokens),
        "total_coeffs": total_coeffs,
        "zero_coeff_blocks": zero_coeff_blocks,
        "trailing_bits": trailing_bits,
        "consumed_bits": nal.consumed_bits,
        "rbsp_bits_total": nal.rbsp_bits_total,
        "boundary_delta_bits": delta,
        "status": nal.status.value,
        "verdict": verdict,
    }


def analyze_generated_mbs(prefix: bytes, generated: bytes) -> list[dict]:
    """Parse prefix+generated and return per-MB analysis for the VCL NALs that begin in
    the generated region (byte offset >= len(prefix))."""
    n_prefix = len(prefix)
    parsed = HS.parse_stream(prefix + generated, parse_slice_data=True)
    out = []
    for idx, nal in enumerate(parsed.nals):
        if nal.nal.nal_type not in VCL_NAL_TYPES:
            continue
        if nal.nal.start_code_start < n_prefix:
            continue
        rec = analyze_mb(nal)
        rec["mb_idx"] = idx
        out.append(rec)
    return out


@torch.inference_mode()
def evaluate_checkpoint(raw, clips, device, args, name) -> tuple[dict, list]:
    clip_records = []
    verdicts = Counter()
    underrun_deltas, overrun_ct, illegal_ct = [], 0, 0
    total_mbs, aligned_mbs = 0, 0
    mbs_per_clip = []
    for idx, clip in enumerate(clips):
        prefix = _read_prefix(clip)
        p_ids, r_ids, o_ids = _prompt_from_stream(prefix, device)
        max_gen = model_max_gen(raw, prefix)
        gen, _ = free_run_rollout(raw, p_ids, r_ids, o_ids, device, args.cont_frames, max_gen,
                                  args.temperature, top_k=args.top_k, top_p=args.top_p)
        mbs = analyze_generated_mbs(prefix, gen)
        mbs_per_clip.append(len(mbs))
        for m in mbs:
            v = m["verdict"]
            verdicts[v.split("@")[0]] += 1
            total_mbs += 1
            if v == "aligned":
                aligned_mbs += 1
            elif v == "underrun":
                underrun_deltas.append(m["boundary_delta_bits"])
            elif v == "overrun":
                overrun_ct += 1
            elif v.startswith("illegal"):
                illegal_ct += 1
        last = mbs[-1]["verdict"] if mbs else "none"
        print(f"  clip {idx:2d}: {len(mbs):3d} generated MBs | verdicts="
              f"{dict(Counter(m['verdict'].split('@')[0] for m in mbs))} | last={last}", flush=True)
        clip_records.append({
            "checkpoint": name,
            "clip_index": idx,
            "h264_path": str(clip.h264_path),
            "n_generated_mbs": len(mbs),
            "mbs": mbs,
        })

    n = total_mbs or 1
    summary = {
        "meta": {"checkpoint": name, "n_clips": len(clips), "temperature": args.temperature,
                 "top_k": args.top_k, "top_p": args.top_p},
        "generated_mbs": {
            "total": total_mbs,
            "per_clip_mean": statistics.mean(mbs_per_clip) if mbs_per_clip else 0,
            "per_clip_median": statistics.median(mbs_per_clip) if mbs_per_clip else 0,
        },
        "verdict_hist": dict(verdicts.most_common()),
        "rates": {
            "aligned_rate": aligned_mbs / n,
            "underrun_rate": len(underrun_deltas) / n,
            "overrun_rate": overrun_ct / n,
            "illegal_rate": illegal_ct / n,
        },
        "underrun_delta_bits": {
            "n": len(underrun_deltas),
            "mean": statistics.mean(underrun_deltas) if underrun_deltas else None,
            "median": statistics.median(underrun_deltas) if underrun_deltas else None,
            "min": min(underrun_deltas) if underrun_deltas else None,
        },
    }
    return summary, clip_records


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows = load_manifest_rows(args.manifest, max_rows=args.max_manifest_rows or None, report_progress=True)
    index_path = args.nal_index_path or default_nal_index_path(args.manifest)
    nal_index = load_nal_index(index_path, args.manifest, rows)
    clips = select_continuation_clips(rows, nal_index, args)
    print(f"Selected {len(clips)} continuation clips", flush=True)

    clips_path = args.out_dir / "mb_length_clips.jsonl"
    summaries = []
    with clips_path.open("w", encoding="utf-8") as fh:
        for ckpt in args.checkpoint_dirs:
            print(f"Loading checkpoint: {ckpt}", flush=True)
            model = load_model(ckpt, device)
            raw = model.module if hasattr(model, "module") else model
            summary, records = evaluate_checkpoint(raw, clips, device, args, ckpt.name)
            for r in records:
                fh.write(json.dumps(r) + "\n")
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
            del model
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"MB length check complete: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
