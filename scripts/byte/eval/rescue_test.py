"""Free-run failure pinpoint: mb_type rescue test (Exp 1) + backward rescue (Exp 2).

Free-run generation desyncs (usually at mb_type). Two causes look identical from the
crash location alone:
  (A) p(mb_type | generated prefix) is bad -- a local conditional failure, OR
  (B/C) an earlier field pushed the generated parser STATE off-manifold and mb_type is
        only where the stream first becomes illegal (the canary).

The reference is NOT GT equality -- it is "under the generated state S, does a legal
continuation exist that the model should have found?". So at the crash we enumerate the
LEGAL mb_type values, force each into the stream, let the model CONTINUE, and measure
survival:
  * some legal mb_type rescues (model continues far)  -> Case A  (fix mb_type)
  * none rescue                                        -> Case B/C -> Exp 2 walks backward
    field-by-field to find the earlier field whose repair restores decodability.

Outputs (clear, per-experiment -- see plan): rescue_clips.jsonl (one sectioned record
per clip) + summary.json (grouped by experiment, headline metric first).

Usage:
    python scripts/byte/eval/rescue_test.py MANIFEST --nal-index-path NAL.sqlite \
        --checkpoint-dirs RUN/step-XXXX [...] --out-dir OUT \
        --prefix-frames 8 --cont-frames 4 --num-clips 20 --temperature 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

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
    bytes_to_ids,
    default_nal_index_path,
    load_manifest_rows,
    load_nal_index,
    parse_annexb_nals,
)
from litgpt.byte import h264_syntax as HS  # noqa: E402
from litgpt.byte import h264_encode as HE  # noqa: E402
from litgpt.byte.free_run_eval import _survival_and_validity, free_run_rollout  # noqa: E402
from scripts.byte.eval.eval_checkpoints import load_model  # noqa: E402
from scripts.byte.eval.eval_stream_continuation import (  # noqa: E402
    model_max_gen,
    select_continuation_clips,
)

# Fields Exp 2 walks backward over (closest-to-crash first is applied at runtime).
_BACKWARD_FIELDS = (
    "mb_qp_delta", "cbp", "coded_block_pattern", "sub_mb_type", "ref_idx_l0",
    "mvd_l0.x", "mvd_l0.y", "coeff_token", "total_zeros", "run_before", "level",
)


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
    p.add_argument("--prefix-frames", type=int, default=8)
    p.add_argument("--cont-frames", type=int, default=4)
    p.add_argument("--max-window-bytes", type=int, default=16384)
    p.add_argument("--max-gen-multiple", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=1.0, help="match the survival sweep; 0=greedy")
    p.add_argument("--rescue-gain-bytes", type=int, default=256,
                   help="a legal mb_type 'rescues' if it survives this many bytes past the original desync")
    p.add_argument("--max-backward-fields", type=int, default=8)
    p.add_argument("--self-test", action="store_true",
                   help="validate the sub-byte forcing path on one mb_type-desync clip, then exit")
    return p.parse_args()


def _prompt_from_stream(stream: bytes, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (prompt_ids, region_ids, offset_ids) for an arbitrary Annex-B byte stream,
    mirroring _prompt_tensors: BOS prepended, region = META for SPS/PPS else TARGET,
    offset = per-NAL byte position (arange over each NAL, including its start code)."""
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


@torch.inference_mode()
def _byte_dist_at_end(raw, stream: bytes, device: torch.device) -> list[float]:
    """Model's probability distribution over the NEXT byte after ``stream`` (i.e. the
    distribution over the byte at position len(stream)). Single teacher-forced forward."""
    prompt_ids, region_ids, offset_ids = _prompt_from_stream(stream, device)
    # Plain full-sequence forward (NO input_pos): passing input_pos routes the model
    # into its KV-cache path (model.py:134), which requires set_kv_cache(). This is a
    # one-shot teacher-forced forward, so run cache-free -- the causal mask makes the
    # last position's logits identical to the cached path. free_run_rollout may have
    # left a cache allocated; input_pos=None means attention ignores it.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = raw(prompt_ids, region_ids=region_ids, offset_ids=offset_ids)
    row = logits[0, -1, :BYTE_VOCAB_SIZE].float()
    return torch.softmax(row, dim=-1).tolist()


def _failed_mb_type_span(prefix: bytes, generated: bytes):
    """Reparse prefix+generated and return the mb_type SyntaxSpan at the first desync in
    the generated region (the illegal ue was recorded as a span before it was rejected),
    plus the slice_type of that NAL. Returns (span, slice_type) or (None, None)."""
    n_prefix = len(prefix)
    parsed = HS.parse_stream(prefix + generated, parse_slice_data=True)
    for nal in parsed.nals:
        if nal.status == HS.ParseStatus.OK:
            continue
        if nal.desync_byte is None or nal.desync_byte < n_prefix:
            continue
        slice_type = None
        mb_type_span = None
        for s in nal.spans:
            if s.name == "slice_type":
                slice_type = s.value
            if s.name == "mb_type":
                mb_type_span = s  # last mb_type before the crash is the illegal one
        return mb_type_span, slice_type
    return None, None


def _survival(prefix: bytes, generated: bytes, cont_frames: int):
    return _survival_and_validity(prefix, generated, cont_frames)


def _init_offset(prompt_stream: bytes) -> int:
    """Per-NAL byte offset the first *generated* byte should take when continuing
    mid-slice: len(prompt) minus the start of the NAL the prompt ends inside."""
    nals = parse_annexb_nals(prompt_stream)
    if not nals:
        return 0
    return len(prompt_stream) - nals[-1].start


def exp1_rescue(raw, prefix: bytes, generated: bytes, sr, device, args) -> dict:
    """Exp 1: at the mb_type crash, P_legal + substitute each legal mb_type and continue."""
    span, slice_type = _failed_mb_type_span(prefix, generated)
    out: dict[str, Any] = {"p_legal": None, "exp1_rescue": None}
    if span is None or sr.desync_region != "mb_type":
        return out  # not an mb_type crash -> Exp 1 not applicable this clip

    stream = prefix + generated
    p_bit = span.byte_start * 8 + (span.bit_start & 7)  # raw bit where mb_type begins
    byte_b = p_bit >> 3
    bit_off = p_bit & 7
    fixed_high = HE.read_bits_int(stream, byte_b * 8, bit_off)
    emitted_byte = stream[byte_b]
    legal_values = HE.legal_mb_type(slice_type if slice_type is not None else HE.SLICE_TYPE_P)

    # P_legal from the model's distribution over byte B (feed bytes up to byte B).
    byte_probs = _byte_dist_at_end(raw, stream[:byte_b], device)
    pl = HE.legal_prob_mass(byte_probs, bit_off, fixed_high, legal_values, emitted_byte)
    order = sorted(range(256), key=lambda b: byte_probs[b], reverse=True)
    rank_of = {b: i for i, b in enumerate(order)}
    out["p_legal"] = {
        "field": "mb_type", "slice_type": slice_type,
        "p_legal": pl["p_legal"], "best_legal_value": pl["best_value"],
        "best_legal_rank": pl["best_legal_rank"], "best_legal_prob": pl["best_prob"],
        "illegal_value": span.value, "illegal_prob": pl["illegal_prob"],
        "illegal_rank": rank_of.get(emitted_byte),
    }

    # Rescue: force each legal mb_type (ranked by model prob) and continue.
    prompt = stream[:byte_b]  # byte-aligned; forced_bits reconstruct byte B onward
    base_gen = generated[: byte_b - len(prefix)]  # generated bytes before byte B
    max_gen = min(int(len(generated) * args.max_gen_multiple) + 512, model_max_gen(raw, prompt))
    ranked = sorted(legal_values, key=lambda v: byte_probs_for_value(byte_probs, v, bit_off, fixed_high), reverse=True)
    io = _init_offset(prompt)
    candidates = []
    orig_survival = sr.survival
    best_gain = -1
    for v in ranked[:8]:  # top-8 legal by model mass is plenty to detect a rescue
        forced = HE.int_to_bits(fixed_high, bit_off) + HE.encode_ue(v)
        p_ids, r_ids, o_ids = _prompt_from_stream(prompt, device)
        gen2, _ = free_run_rollout(raw, p_ids, r_ids, o_ids, device, args.cont_frames, max_gen,
                                   args.temperature, forced_bits=forced, init_offset=io)
        sr2 = _survival(prefix, base_gen + gen2, args.cont_frames)
        candidates.append({
            "value": v, "codeword": "".join(map(str, HE.encode_ue(v))),
            "model_logprob": _logprob_for_value(byte_probs, v, bit_off, fixed_high),
            "rank": _rank_for_value(byte_probs, v, bit_off, fixed_high, order),
            "survival": sr2.survival, "next_crash_region": sr2.desync_region,
        })
        best_gain = max(best_gain, sr2.survival - orig_survival)

    verdict = "A" if best_gain >= args.rescue_gain_bytes else "BC"
    out["exp1_rescue"] = {
        "verdict": verdict,
        "best_rescue_survival": max((c["survival"] for c in candidates), default=orig_survival),
        "candidates": candidates,
    }
    return out


def byte_probs_for_value(byte_probs, v, bit_off, fixed_high) -> float:
    """Summed prob over bytes consistent with fixed_high whose tail begins codeword v."""
    cw = HE.encode_ue(v)
    tail_len = 8 - bit_off
    pref = tuple(cw[: min(len(cw), tail_len)])
    total = 0.0
    for b in range(256):
        if (b >> tail_len) != fixed_high:
            continue
        tail = [(b >> (tail_len - 1 - i)) & 1 for i in range(tail_len)]
        if tuple(tail[: len(pref)]) == pref:
            total += byte_probs[b]
    return total


def _logprob_for_value(byte_probs, v, bit_off, fixed_high) -> float:
    p = byte_probs_for_value(byte_probs, v, bit_off, fixed_high)
    return math.log(p) if p > 0 else float("-inf")


def _rank_for_value(byte_probs, v, bit_off, fixed_high, order) -> int | None:
    cw = HE.encode_ue(v)
    tail_len = 8 - bit_off
    pref = tuple(cw[: min(len(cw), tail_len)])
    for i, b in enumerate(order):
        if (b >> tail_len) != fixed_high:
            continue
        tail = [(b >> (tail_len - 1 - j)) & 1 for j in range(tail_len)]
        if tuple(tail[: len(pref)]) == pref:
            return i
    return None


def exp2_backward(raw, prefix: bytes, generated: bytes, sr, device, args) -> dict:
    """Exp 2: when no mb_type rescue works, walk backward over prior enumerable fields,
    substitute a legal alternative, continue, and find the field whose repair restores
    survival = the causal field. Context-dependent legal sets (coeff_token needs nC) fall
    back to the model's top-k byte values consistent with the field's byte alignment."""
    n_prefix = len(prefix)
    parsed = HS.parse_stream(prefix + generated, parse_slice_data=True)
    stream = prefix + generated
    # prior fields in the crashed NAL, closest-to-crash first
    prior = []
    for nal in parsed.nals:
        if nal.status == HS.ParseStatus.OK or (nal.desync_byte or 0) < n_prefix:
            continue
        for s in nal.spans:
            base = s.name.split("[")[0]
            if base in _BACKWARD_FIELDS and s.byte_start >= n_prefix:
                prior.append(s)
        break
    prior = list(reversed(prior))[: args.max_backward_fields]

    max_gen = min(int(len(generated) * args.max_gen_multiple) + 512, model_max_gen(raw, stream[: -1]))
    fields_out = []
    best = {"field": None, "survival": sr.survival}
    for s in prior:
        p_bit = s.byte_start * 8 + (s.bit_start & 7)
        byte_b = p_bit >> 3
        bit_off = p_bit & 7
        fixed_high = HE.read_bits_int(stream, byte_b * 8, bit_off)
        prompt = stream[:byte_b]
        base_gen = generated[: byte_b - n_prefix]
        orig_byte = stream[byte_b]
        # Repair candidates: the model's top consistent bytes OTHER than the one it
        # originally generated (first-cut proxy for a "legal alternative"; enumerable
        # fields with a true legal set are a future refinement).
        byte_probs = _byte_dist_at_end(raw, prompt, device)
        consistent = [b for b in range(256) if (b >> (8 - bit_off)) == fixed_high and b != orig_byte]
        alts = sorted(consistent, key=lambda b: byte_probs[b], reverse=True)[:3]
        io = _init_offset(prompt)
        field_best = {"alt": None, "survival": sr.survival}
        for alt_byte in alts:
            forced = HE.int_to_bits(alt_byte, 8)
            p_ids, r_ids, o_ids = _prompt_from_stream(prompt, device)
            gen2, _ = free_run_rollout(raw, p_ids, r_ids, o_ids, device, args.cont_frames, max_gen,
                                       args.temperature, forced_bits=forced, init_offset=io)
            sr2 = _survival(prefix, base_gen + gen2, args.cont_frames)
            if sr2.survival > field_best["survival"]:
                field_best = {"alt": alt_byte, "survival": sr2.survival}
        fields_out.append({
            "field": s.name.split("[")[0], "mb_addr": s.mb_addr,
            "gen_value": _jsonable(s.value), "gen_len": s.bit_end - s.bit_start,
            "alt_value": field_best["alt"], "survival_after_repair": field_best["survival"],
        })
        if field_best["survival"] > best["survival"]:
            best = {"field": s.name.split("[")[0], "survival": field_best["survival"]}
    return {"causal_field": best["field"], "fields": fields_out}


def _jsonable(v):
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)


def _read_prefix(clip) -> tuple[bytes, list, int]:
    from litgpt.byte.data import NALUnit  # noqa
    data = clip.h264_path.read_bytes()
    nals = parse_annexb_nals(data)
    prefix = b"".join(data[nals[i].start:nals[i].end] for i in range(clip.prefix_end_nal))
    return prefix, nals, clip.prefix_end_nal


def _failed_coeff_token_span(prefix: bytes, generated: bytes):
    """Reparse and return (span, nc) for the FAILED coeff_token at the first desync in
    the generated region -- the span the parser records with {'nC':.., 'failed':True}
    right before the VLC miss propagates. Returns (None, None) if the crash was not a
    coeff_token (e.g. run_before / total_zeros)."""
    n_prefix = len(prefix)
    parsed = HS.parse_stream(prefix + generated, parse_slice_data=True)
    for nal in parsed.nals:
        if nal.status == HS.ParseStatus.OK:
            continue
        if nal.desync_byte is None or nal.desync_byte < n_prefix:
            continue
        span = None
        for s in nal.spans:
            if s.name.endswith(".coeff_token"):
                span = s  # last coeff_token span before/at the crash
        if span is None or not (isinstance(span.value, dict) and span.value.get("failed")):
            return None, None  # last coeff_token succeeded -> crash was a later residual field
        return span, span.value["nC"]
    return None, None


def _cw_mass(byte_probs, bits, bit_off, fixed_high) -> float:
    """Summed model mass over bytes consistent with fixed_high whose tail begins codeword bits."""
    tail_len = 8 - bit_off
    pref = tuple(bits[: min(len(bits), tail_len)])
    total = 0.0
    for b in range(256):
        if (b >> tail_len) != fixed_high:
            continue
        tail = [(b >> (tail_len - 1 - i)) & 1 for i in range(tail_len)]
        if tuple(tail[: len(pref)]) == pref:
            total += byte_probs[b]
    return total


def exp1_coeff_token(raw, prefix: bytes, generated: bytes, sr, device, args) -> dict:
    """Exp 1 for a coeff_token crash: P_legal over the nC-selected VLC table + force each
    legal codeword and continue. Directly tests the nC/parse-state hypothesis -- P_legal
    high (model wanted a legal codeword, parser still missed) points to a wrong nC state
    (B/C); P_legal low points to a genuinely bad residual conditional (A)."""
    span, nc = _failed_coeff_token_span(prefix, generated)
    out: dict[str, Any] = {"p_legal": None, "exp1_rescue": None}
    if span is None:
        return out
    stream = prefix + generated
    p_bit = span.byte_start * 8 + (span.bit_start & 7)
    byte_b, bit_off = p_bit >> 3, p_bit & 7
    fixed_high = HE.read_bits_int(stream, byte_b * 8, bit_off)
    emitted_byte = stream[byte_b]
    codewords = HE.legal_coeff_token(nc)

    byte_probs = _byte_dist_at_end(raw, stream[:byte_b], device)
    pl = HE.legal_prob_mass_codewords(byte_probs, bit_off, fixed_high, codewords, emitted_byte)
    out["p_legal"] = {
        "field": span.name.split("[")[0] + ".coeff_token" if "[" in span.name else span.name,
        "nC": nc, "p_legal": pl["p_legal"],
        "best_legal_value": list(pl["best_value"]) if pl["best_value"] is not None else None,
        "best_legal_rank": pl["best_legal_rank"], "best_legal_prob": pl["best_prob"],
        "illegal_prob": pl["illegal_prob"],
    }

    prompt = stream[:byte_b]
    base_gen = generated[: byte_b - len(prefix)]
    max_gen = min(int(len(generated) * args.max_gen_multiple) + 512, model_max_gen(raw, prompt))
    io = _init_offset(prompt)
    ranked = sorted(codewords, key=lambda vc: _cw_mass(byte_probs, vc[1], bit_off, fixed_high), reverse=True)
    candidates = []
    orig_survival = sr.survival
    best_gain = -1
    for value, bits in ranked[:8]:
        forced = HE.int_to_bits(fixed_high, bit_off) + bits
        p_ids, r_ids, o_ids = _prompt_from_stream(prompt, device)
        gen2, _ = free_run_rollout(raw, p_ids, r_ids, o_ids, device, args.cont_frames, max_gen,
                                   args.temperature, forced_bits=forced, init_offset=io)
        sr2 = _survival(prefix, base_gen + gen2, args.cont_frames)
        candidates.append({"value": list(value), "survival": sr2.survival,
                           "next_crash_region": sr2.desync_region})
        best_gain = max(best_gain, sr2.survival - orig_survival)
    verdict = "A" if best_gain >= args.rescue_gain_bytes else "BC"
    out["exp1_rescue"] = {
        "verdict": verdict,
        "best_rescue_survival": max((c["survival"] for c in candidates), default=orig_survival),
        "candidates": candidates,
    }
    return out


def _mb_type_value_at(stream: bytes, byte_b: int):
    """Reparse and return the mb_type value whose span begins at on-disk byte byte_b."""
    parsed = HS.parse_stream(stream, parse_slice_data=True)
    for nal in parsed.nals:
        for s in nal.spans:
            if s.name == "mb_type" and s.byte_start == byte_b:
                return s.value
    return None


@torch.inference_mode()
def run_forcing_self_test(raw, clips, device, args) -> bool:
    """Validate the sub-byte forcing path BEFORE trusting any rescue survival. On the
    first clip that free-runs to an mb_type desync, force two legal mb_type values into
    the crash boundary byte, continue, reparse, and assert the forced field decodes to
    exactly that value (and that different forced values yield different tails, i.e. the
    model fills the remaining bits itself). Prints PASS/FAIL. Returns True on pass."""
    torch.manual_seed(args.seed)
    for idx, clip in enumerate(clips):
        prefix, nals, _ = _read_prefix(clip)
        p_ids, r_ids, o_ids = _prompt_from_stream(prefix, device)
        max_gen = model_max_gen(raw, prefix)
        gen, _ = free_run_rollout(raw, p_ids, r_ids, o_ids, device, args.cont_frames, max_gen, args.temperature)
        sr = _survival(prefix, gen, args.cont_frames)
        span, slice_type = _failed_mb_type_span(prefix, gen)
        if span is None or sr.desync_region != "mb_type":
            continue
        stream = prefix + gen
        p_bit = span.byte_start * 8 + (span.bit_start & 7)
        byte_b, bit_off = p_bit >> 3, p_bit & 7
        fixed_high = HE.read_bits_int(stream, byte_b * 8, bit_off)
        prompt = stream[:byte_b]
        io = _init_offset(prompt)
        legal = HE.legal_mb_type(slice_type if slice_type is not None else HE.SLICE_TYPE_P)
        tails, ok_all = [], True
        for v in legal[:2]:
            forced = HE.int_to_bits(fixed_high, bit_off) + HE.encode_ue(v)
            pp, rr, oo = _prompt_from_stream(prompt, device)
            gen2, _ = free_run_rollout(raw, pp, rr, oo, device, args.cont_frames, max_gen,
                                       args.temperature, forced_bits=forced, init_offset=io)
            got = _mb_type_value_at(prompt + gen2, byte_b)
            ok = got == v
            ok_all = ok_all and ok
            tails.append(bytes(gen2))
            print(f"  self-test clip {idx}: force mb_type={v} -> reparsed={got}  {'OK' if ok else 'FAIL'}", flush=True)
        if len(tails) == 2 and tails[0] == tails[1]:
            print("  self-test WARN: identical continuations for different forced values "
                  "(model may not be filling the free bits)", flush=True)
        print(f"  self-test: {'PASS' if ok_all else 'FAIL'} -- forcing decodes to the intended field", flush=True)
        return ok_all
    print("  self-test: no mb_type-desync clip among the selected clips; raise --num-clips", flush=True)
    return False


@torch.inference_mode()
def evaluate_checkpoint(raw, clips, device, args, name) -> tuple[dict, list]:
    clip_records = []
    survivals, regions = [], Counter()
    p_legals, case_A = [], 0
    exp2_triggered, causal_fields = 0, Counter()
    ct_p_legals, ct_case_A, ct_n = [], 0, 0  # coeff_token probe aggregates
    for idx, clip in enumerate(clips):
        prefix, nals, prefix_end = _read_prefix(clip)
        p_ids, r_ids, o_ids = _prompt_from_stream(prefix, device)
        max_gen = model_max_gen(raw, prefix)
        gen, _ = free_run_rollout(raw, p_ids, r_ids, o_ids, device, args.cont_frames, max_gen, args.temperature)
        sr = _survival(prefix, gen, args.cont_frames)
        survivals.append(sr.survival)
        regions[sr.desync_region or "none"] += 1

        rec: dict[str, Any] = {
            "clip": {"h264_path": str(clip.h264_path), "prefix_frames": args.prefix_frames,
                     "cont_frames": args.cont_frames},
            "free_run": {"survival_bytes": sr.survival, "desync_byte": len(prefix) + sr.survival,
                         "desync_region": sr.desync_region, "desync_reason": sr.desync_reason},
            "p_legal": None, "exp1_rescue": None, "exp2_backward": None, "field": None,
        }
        e1 = exp1_rescue(raw, prefix, gen, sr, device, args)
        if e1["p_legal"] is not None:  # mb_type crash
            rec["field"] = "mb_type"
            rec["p_legal"] = e1["p_legal"]
            rec["exp1_rescue"] = e1["exp1_rescue"]
            p_legals.append(e1["p_legal"]["p_legal"])
            verdict = (e1["exp1_rescue"] or {}).get("verdict")
            if verdict == "A":
                case_A += 1
                line = f"Case A (mb_type={e1['p_legal']['best_legal_value']} rescues)"
            elif verdict == "BC":
                exp2_triggered += 1
                e2 = exp2_backward(raw, prefix, gen, sr, device, args)
                rec["exp2_backward"] = e2
                causal_fields[e2["causal_field"] or "none"] += 1
                line = f"Case BC -> causal field = {e2['causal_field']}"
            else:
                line = "mb_type (no rescue)"
            pl = e1["p_legal"]["p_legal"]
        else:  # not mb_type -> try the coeff_token / nC probe
            ec = exp1_coeff_token(raw, prefix, gen, sr, device, args)
            if ec["p_legal"] is not None:
                rec["field"] = "coeff_token"
                rec["p_legal"] = ec["p_legal"]
                rec["exp1_rescue"] = ec["exp1_rescue"]
                ct_n += 1
                ct_p_legals.append(ec["p_legal"]["p_legal"])
                v = (ec["exp1_rescue"] or {}).get("verdict")
                if v == "A":
                    ct_case_A += 1
                line = f"coeff_token nC={ec['p_legal']['nC']} -> Case {v}"
                pl = ec["p_legal"]["p_legal"]
            else:
                line = f"desync@{sr.desync_region} (not a rescuable field)"
                pl = None
        print(f"  clip {idx}: desync@{sr.desync_region} P_legal={pl} -> {line}", flush=True)
        clip_records.append(rec)

    n = len(clips) or 1
    summary = {
        "meta": {"checkpoint": name, "n_clips": len(clips), "temperature": args.temperature},
        "free_run": {
            "survival_mean": sum(survivals) / n,
            "survival_median": sorted(survivals)[len(survivals) // 2] if survivals else None,
            "desync_region_hist": dict(regions.most_common()),
        },
        "exp1_mb_type": {
            "n": len(p_legals),
            "case_A_rate": case_A / n,
            "case_BC_rate": exp2_triggered / n,
            "p_legal_mean": (sum(p_legals) / len(p_legals)) if p_legals else None,
        },
        "exp1_coeff_token": {
            "n": ct_n,
            "case_A_rate": (ct_case_A / ct_n) if ct_n else None,
            "p_legal_mean": (sum(ct_p_legals) / len(ct_p_legals)) if ct_p_legals else None,
        },
        "exp2": {
            "n_triggered": exp2_triggered,
            "causal_field_hist": dict(causal_fields.most_common()),
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

    if args.self_test:
        ckpt = args.checkpoint_dirs[0]
        print(f"Loading checkpoint for self-test: {ckpt}", flush=True)
        model = load_model(ckpt, device)
        raw = model.module if hasattr(model, "module") else model
        ok = run_forcing_self_test(raw, clips, device, args)
        del model
        print(f"Forcing self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)

    clips_path = args.out_dir / "rescue_clips.jsonl"
    summaries = []
    with clips_path.open("w", encoding="utf-8") as fh:
        for ckpt in args.checkpoint_dirs:
            print(f"Loading checkpoint: {ckpt}", flush=True)
            model = load_model(ckpt, device)
            raw = model.module if hasattr(model, "module") else model
            summary, records = evaluate_checkpoint(raw, clips, device, args, ckpt.name)
            for r in records:
                r["checkpoint"] = ckpt.name
                fh.write(json.dumps(r) + "\n")
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
            del model
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Rescue test complete: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
