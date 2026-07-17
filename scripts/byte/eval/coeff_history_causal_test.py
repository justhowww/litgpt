#!/usr/bin/env python3
"""Randomized preceding-coefficient-block causal test.

For each persisted continuation whose first failure is BitReaderError, locate the
last fully decoded CAVLC block before that failure.  Replace exactly that block
(coeff_token through run_before) by a randomly generated valid block under the
same incoming nC, then resume greedy generation from the altered byte prefix.

If the predecessor is in the failing NAL, this tests within-slice compounding:
changed TotalCoeff -> changed nC -> changed table for the next block.  If it is in
an earlier NAL, the bit reader cannot carry alignment across the start code; that
case tests only whether altered coefficient bytes change the model's later output.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte.data import VCL_NAL_TYPES  # noqa: E402
from litgpt.byte.free_run_eval import _survival_and_validity, free_run_rollout  # noqa: E402
from scripts.byte.eval.cavlc_coeff_sanity import random_block_bits  # noqa: E402
from scripts.byte.eval.eval_checkpoints import load_model  # noqa: E402
from scripts.byte.eval.eval_stream_continuation import model_max_gen  # noqa: E402
from scripts.byte.eval.rescue_test import _init_offset, _prompt_from_stream  # noqa: E402


def _bits(data: bytes) -> list[int]:
    return [(b >> (7 - i)) & 1 for b in data for i in range(8)]


def _pack(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise ValueError("bit prefix must end at a byte boundary")
    out = bytearray(len(bits) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def _escape_rbsp(rbsp: bytes) -> bytes:
    out = bytearray()
    zeros = 0
    for b in rbsp:
        if zeros >= 2 and b <= 0x03:
            out.append(0x03)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def _max_coeff(name: str) -> int:
    if name.startswith("chroma_dc["):
        return 4
    if name.startswith("luma_ac[") or name.startswith("chroma_ac["):
        return 15
    return 16


def _completed_coeff_spans(parse: HS.NALParse):
    return [
        s for s in parse.spans
        if s.name.endswith(".coeff_token")
        and s.bit_start >= 0
        and isinstance(s.value, dict)
        and not s.value.get("failed")
    ]


def _generated_nals(stream: bytes, n_prefix: int):
    parsed = HS.parse_stream(stream, parse_slice_data=True)
    return [p for p in parsed.nals if p.nal.start_code_start >= n_prefix]


def _find_predecessor(nals, failure_j: int):
    """Return (NAL index, parse, completed coeff_token span) immediately before failure."""
    target = nals[failure_j]
    limit = target.desync_bit if target.desync_bit is not None else 1 << 60
    in_target = [s for s in _completed_coeff_spans(target) if s.bit_end <= limit]
    if in_target:
        return failure_j, target, max(in_target, key=lambda s: s.bit_end)
    for j in range(failure_j - 1, -1, -1):
        spans = _completed_coeff_spans(nals[j])
        if spans:
            return j, nals[j], max(spans, key=lambda s: s.bit_end)
    return None


def _intervention(stream: bytes, selected, new_block: list[int]) -> tuple[bytes, list[int]]:
    """Return a byte prompt before the block and the exact EBSP bits to force.

    The syntax location is in RBSP bit space, while the model emits escaped EBSP
    bytes.  Escape the altered RBSP prefix first, then force all complete bytes plus
    only the meaningful high bits of its final partial byte.
    """
    _selected_j, parse, token = selected
    nal = parse.nal
    rbsp, _, _ = HS.unescape_rbsp(stream, nal.payload_start + 1, nal.payload_end)
    old = _bits(rbsp)
    meaningful_rbsp_bits = token.bit_start + len(new_block)
    altered = old[:token.bit_start] + new_block
    altered.extend([0] * ((-len(altered)) % 8))
    desired = _escape_rbsp(_pack(altered))
    original = stream[nal.payload_start + 1:nal.payload_end]

    common = 0
    while common < min(len(desired), len(original)) and desired[common] == original[common]:
        common += 1
    rem = meaningful_rbsp_bits % 8
    if rem and common == len(desired):
        # Never put the artificial low-bit zero padding into the teacher-forced prompt.
        common -= 1
    meaningful_ebsp_bits = 8 * len(desired) if rem == 0 else 8 * (len(desired) - 1) + rem
    forced = _bits(desired)[8 * common:meaningful_ebsp_bits]
    if not forced:
        raise ValueError(f"randomized {token.name} produced no changed forced bits")
    prompt = stream[:nal.payload_start + 1] + desired[:common]
    return prompt, forced


def _frame_starts(stream: bytes, n_prefix: int) -> int:
    count = 0
    for p in HS.parse_stream(stream, parse_slice_data=False).nals:
        if p.nal.start_code_start < n_prefix or p.nal.nal_type not in VCL_NAL_TYPES:
            continue
        raw = stream[p.nal.payload_start:p.nal.payload_end]
        if HS.slice_first_mb(raw) == 0:
            count += 1
    return count


def _first_failure(stream: bytes, n_prefix: int) -> dict | None:
    for j, p in enumerate(_generated_nals(stream, n_prefix)):
        if p.status == HS.ParseStatus.DESYNC:
            return {
                "nal_index": j,
                "kind": p.reason_kind,
                "element": p.failure_element,
                "consumed_bits": p.consumed_bits,
                "total_bits": p.rbsp_bits_total,
            }
    return None


def _latest_rows(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row.get("stream_gen_path") and row.get("frames_target") == 2:
            rows[int(row["clip_index"])] = row
    return rows


@torch.inference_mode()
def _resume(
    raw, prompt: bytes, forced_bits: list[int], clean_prefix: bytes,
    original_max_gen: int, device,
) -> dict:
    used = len(prompt) - len(clean_prefix)
    budget = min(max(1, original_max_gen - used), model_max_gen(raw, prompt))
    p, r, o = _prompt_from_stream(prompt, device)
    remaining_frames = max(0, 2 - _frame_starts(prompt, len(clean_prefix)))
    tail, _ = free_run_rollout(
        raw, p, r, o, device, remaining_frames, budget, 0.0,
        forced_bits=forced_bits, init_offset=_init_offset(prompt),
    )
    continuation = prompt[len(clean_prefix):] + tail
    sr = _survival_and_validity(clean_prefix, continuation, 2)
    full = clean_prefix + continuation
    return {
        "survival": sr.survival,
        "desync_reason": sr.desync_reason,
        "first_parser_failure": _first_failure(full, len(clean_prefix)),
        "generated_bytes": len(continuation),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=32,
                    help="model rollouts per failed clip; unlike isolated CAVLC, these are expensive")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = _latest_rows(args.results_dir / "clip_details.jsonl")
    terms = {
        int(json.loads(line)["clip_index"]): json.loads(line)
        for line in (args.results_dir / "nal_termination_clips.jsonl").open()
    }
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.checkpoint_dir, device)
    raw = model.module if hasattr(model, "module") else model
    rng = random.Random(args.seed)
    output = []

    for clip, term in sorted(terms.items()):
        if term.get("desync_reason") != "BitReaderError" or clip not in rows:
            continue
        row = rows[clip]
        stream = Path(row["stream_gen_path"]).read_bytes()
        n_prefix = int(row["n_prefix_bytes"])
        nals = _generated_nals(stream, n_prefix)
        failure_j = int(term["first_desync_nal_index"])
        selected = _find_predecessor(nals, failure_j)
        if selected is None:
            print(f"clip={clip}: no preceding completed coefficient block")
            continue
        selected_j, parse, token = selected
        base = token.name.removesuffix(".coeff_token")
        nc = int(token.value["nC"])
        max_coeff = _max_coeff(base)
        old_pair = (token.value.get("total_coeff"), token.value.get("trailing_ones"))
        original_element = next(
            (n.get("failure_element") for n in term["nals"] if n.get("is_first_desync_nal")),
            None,
        )
        baseline_survival = int(row["survival_bytes"])
        trials = []
        for _ in range(args.trials):
            for _attempt in range(100):
                block, meta = random_block_bits(rng, nc, max_coeff)
                if (meta["total_coeff"], meta["trailing_ones"]) == old_pair:
                    continue
                try:
                    prompt, forced = _intervention(stream, selected, block)
                except ValueError:
                    continue
                break
            else:
                raise RuntimeError(f"clip {clip}: could not sample a distinct forced block")
            result = _resume(
                raw, prompt, forced, stream[:n_prefix], int(row["max_gen"]), device
            )
            result.update(
                sampled_total_coeff=meta["total_coeff"],
                sampled_trailing_ones=meta["trailing_ones"],
                forced_bits=len(forced),
                survival_gain=result["survival"] - baseline_survival,
                same_failure=(
                    (result["first_parser_failure"] or {}).get("kind") == "BitReaderError"
                    and (result["first_parser_failure"] or {}).get("element") == original_element
                ),
            )
            trials.append(result)

        hist = Counter(
            ((t["first_parser_failure"] or {}).get("kind") or "none") for t in trials
        )
        site_hist = Counter(
            f"{(t['first_parser_failure'] or {}).get('kind') or 'none'}:"
            f"{(t['first_parser_failure'] or {}).get('element') or 'none'}"
            for t in trials
        )
        same_nal = selected_j == failure_j
        record = {
            "clip_index": clip,
            "original_failure_nal": failure_j,
            "original_failure_element": original_element,
            "baseline_survival": baseline_survival,
            "predecessor_nal": selected_j,
            "predecessor_block": base,
            "incoming_nC": nc,
            "max_coeff": max_coeff,
            "mechanism": "within_nal_nC_chain" if same_nal else "cross_nal_model_history",
            "downstream_bits_borrowed": 0,
            "failure_kind_hist": dict(hist),
            "failure_site_hist": dict(site_hist),
            "same_original_failure": sum(t["same_failure"] for t in trials),
            "trials": trials,
        }
        output.append(record)
        print(
            f"clip={clip} failure={record['original_failure_element']} prev={base} "
            f"mechanism={record['mechanism']} failures={dict(hist)}",
            flush=True,
        )

    out = args.results_dir / "coeff_history_causal.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
