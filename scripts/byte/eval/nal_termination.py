#!/usr/bin/env python3
"""Why did each generated NAL end? Offline replay analyzer.

Auxiliary to eval_stream_continuation.py (same role as mb_length_check.py). A free-run
failure reports ``desync_reason = BitReaderError``, which only says *the parser wanted
more syntax bits than the NAL contained* -- not why the NAL ended there. The candidates:

  1. premature boundary timing -- the next start code lands at the expected structural
     position (one per macroblock) while the current MB's DECLARED syntax still needs bits;
  2. a wrong coeff_token / earlier structural value making the parser expect the wrong
     number of following fields;
  3. max_gen truncating an unfinished NAL.

(3) is named directly by ``stop_reason`` (recorded by free_run_rollout, since its six
exits share one return). (1) vs (2) is settled by the decisive question this script
answers:

    At each emitted boundary, was the automaton already in ``done``?

``stage != done`` at a boundary => the model learned HOW MANY macroblock NALs to emit but
not WHEN one is syntactically complete.

Runs on persisted streams -- no torch, no GPU, no model. Replays bytes through the same
h264_mask/h264_automaton state machine used during constrained decoding, so an UNMASKED
run gets syntax tracking it never had live (observe-only: nothing here can affect bytes
that were already generated).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

_BYTE_DIR = Path(__file__).resolve().parents[3] / "litgpt" / "byte"


def _load_byte_modules():
    """Import the stdlib-only byte modules without pulling in litgpt/__init__ (torch)."""
    if "litgpt" not in sys.modules:
        sys.modules["litgpt"] = types.ModuleType("litgpt")
        sys.modules["litgpt.byte"] = types.ModuleType("litgpt.byte")

    def load(short: str):
        name = f"litgpt.byte.{short}"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, _BYTE_DIR / f"{short}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(sys.modules["litgpt.byte"], short, module)
        spec.loader.exec_module(module)
        return module

    load("h264_cavlc_tables")
    return load("h264_syntax"), load("h264_automaton"), load("h264_mask")


HS, HA, HM = _load_byte_modules()

# Boundary classifications.
VALID = "valid_start_code"  # automaton reached done -> the NAL was syntactically complete
PREMATURE = "premature_start_code"  # boundary emitted mid-syntax
UNTRACKED_UNKNOWN = "untracked_unknown"  # automaton hit INVALID earlier in THIS NAL
UNTRACKED_NO_AUTO = "untracked_no_automaton"  # slice header never resolved (no SPS/PPS yet)
NON_VCL = "non_vcl"  # SPS/PPS/SEI: no macroblock syntax to complete
OPEN_AT_EOF = "open_at_eof"  # never closed; stop_reason says why


def _snapshot(state) -> dict[str, Any]:
    """The automaton state at a NAL boundary.

    Sound because ``advance`` resets ``automaton`` to None on close, so this must be
    taken BEFORE the closing byte is committed. The two 00 bytes preceding the 01 are
    fed to the automaton as RBSP -- unless it is already done, since _sync_automaton's
    loop guard is ``while auto.pos < nbits and auto.stage != "done"``. And a trailing
    RBSP byte can never be 0x00 (rbsp_stop_one_bit is a 1), so 00 00 can never
    legitimately end a NAL. Hence stage == "done" <=> the macroblock really completed.
    """
    a = state.automaton
    return {
        "cur_is_vcl": state.cur_is_vcl,
        "automaton_unknown": state.automaton_unknown,
        "stage": a.stage if a else None,
        "ae_tag": a.ae_tag if a else None,
        "mbs_done": a.mbs_done if a else None,
        "max_mbs": a.max_mbs if a else None,
        "res_phase": a.res_phase if a else None,
        "res_blk": a.res_blk if a else None,
        "rb_total_coeff": a.rb_total_coeff if a else None,
        "rb_i": a.rb_i if a else None,
        "rb_zeros_left": a.rb_zeros_left if a else None,
        "rbsp_bit_pos": a.pos if a else None,
    }


def _replay(stream: bytes, slice_max_mbs: int) -> dict[int, dict[str, Any]]:
    """Feed the stream through MaskState/advance; snapshot at every NAL close.

    Returns {nal_open_offset: snapshot}. ``nal_open_offset`` is the offset of the NAL's
    header byte, matching NALInfo.payload_start, so snapshots join to parsed NALs.
    """
    state = HM.MaskState(slice_max_mbs=slice_max_mbs)
    snaps: dict[int, dict[str, Any]] = {}
    open_off: int | None = None
    for off, b in enumerate(stream):
        if not state.cur_nal_bytes:
            open_off = off  # this byte becomes cur_nal_bytes[0] = the header byte
        tail = state.cur_nal_bytes
        closes = len(tail) >= 2 and tail[-2] == 0 and tail[-1] == 0 and b == 1
        if closes and open_off is not None:
            snaps[open_off] = _snapshot(state)
        HM.advance(state, b)
    if state.cur_nal_bytes and open_off is not None:
        snap = _snapshot(state)
        snap["open_at_eof"] = True
        snaps[open_off] = snap
    return snaps


def _classify(snap: dict[str, Any] | None) -> str:
    if snap is None:
        return UNTRACKED_NO_AUTO
    if not snap.get("cur_is_vcl"):
        return NON_VCL
    if snap.get("open_at_eof"):
        return OPEN_AT_EOF
    if snap.get("automaton_unknown"):
        return UNTRACKED_UNKNOWN
    if snap.get("stage") is None:
        return UNTRACKED_NO_AUTO
    return VALID if snap["stage"] == "done" else PREMATURE


def _total_coeffs(nal_parse) -> list:
    """TotalCoeff declared by each coeff_token in this NAL, in stream order."""
    return [
        s.value.get("total_coeff")
        for s in nal_parse.spans
        if s.name.endswith(".coeff_token") and isinstance(s.value, dict)
    ]


def _gen_region_nals(parsed, n_prefix: int) -> list:
    return [p for p in parsed.nals if p.nal.start_code_start >= n_prefix]


def analyze_stream(
    gen_stream: bytes,
    gt_stream: bytes,
    n_prefix: int,
    *,
    slice_max_mbs: int = 1,
    first_desync: int | None = None,
) -> dict[str, Any]:
    """Classify every generated-region NAL boundary in ``gen_stream``.

    ``gt_stream`` must share the same prefix so NALs pair by index (both are
    one-MB-per-NAL). Pairing is validated with slice_first_mb and flagged, not assumed.
    """
    snaps = _replay(gen_stream, slice_max_mbs)
    parsed_gen = HS.parse_stream(gen_stream, parse_slice_data=True)
    parsed_gt = HS.parse_stream(gt_stream, parse_slice_data=True)
    gen_nals = _gen_region_nals(parsed_gen, n_prefix)
    gt_nals = _gen_region_nals(parsed_gt, n_prefix)

    records: list[dict[str, Any]] = []
    hist: Counter = Counter()
    desync_nal_index: int | None = None

    for j, p in enumerate(gen_nals):
        nal = p.nal
        snap = snaps.get(nal.payload_start)
        cls = _classify(snap)
        hist[cls] += 1
        gen_len = nal.payload_end - nal.payload_start
        gt_p = gt_nals[j] if j < len(gt_nals) else None

        rec: dict[str, Any] = {
            "gen_nal_index": j,
            "byte_start": nal.payload_start,
            "byte_end": nal.payload_end,
            "nal_type": nal.nal_type,
            "classification": cls,
            "gen_len": gen_len,
            "parse_status": p.status.value,
            "parse_reason_kind": p.reason_kind,
        }
        if snap:
            rec.update(
                {k: snap[k] for k in ("stage", "ae_tag", "mbs_done", "res_phase",
                                      "res_blk", "rb_total_coeff", "rb_i",
                                      "rb_zeros_left", "automaton_unknown")}
            )
        if gt_p is not None:
            gt_len = gt_p.nal.payload_end - gt_p.nal.payload_start
            rec["gt_len"] = gt_len
            rec["delta_bytes"] = gen_len - gt_len
            # Validate the index pairing rather than trusting it.
            g_fmb = HS.slice_first_mb(gen_stream[nal.payload_start:nal.payload_end])
            t_fmb = HS.slice_first_mb(
                gt_stream[gt_p.nal.payload_start:gt_p.nal.payload_end]
            )
            rec["gen_first_mb"], rec["gt_first_mb"] = g_fmb, t_fmb
            rec["aligned"] = g_fmb is not None and g_fmb == t_fmb
            gen_tc, gt_tc = _total_coeffs(p), _total_coeffs(gt_p)
            rec["total_coeff_gen"] = gen_tc
            rec["total_coeff_gt"] = gt_tc
            rec["total_coeff_match"] = gen_tc == gt_tc
        if first_desync is not None and nal.payload_start <= first_desync < nal.payload_end:
            rec["is_first_desync_nal"] = True
            desync_nal_index = j
        records.append(rec)

    # The automaton and the recursive parser are independent implementations that agree
    # on ground truth. A boundary the automaton calls "done" whose NAL the parser still
    # desyncs on is a contradiction between OUR OWN two parsers -- evidence for a parser
    # bug rather than a model failure.
    contradictions = [
        r["gen_nal_index"]
        for r in records
        if r["classification"] == VALID and r["parse_status"] != "ok"
    ]
    return {
        "n_gen_nals": len(gen_nals),
        "classification_hist": dict(hist),
        "first_desync_nal_index": desync_nal_index,
        "parser_automaton_contradictions": contradictions,
        "nals": records,
    }


def _rows_for(details_path: Path) -> list[dict[str, Any]]:
    """Last row per clip that carries persisted streams.

    clip_details.jsonl is append-only, so a directory reused across runs holds several
    runs' rows; keep the most recent per clip_index.
    """
    by_clip: dict[int, dict[str, Any]] = {}
    for line in details_path.open():
        r = json.loads(line)
        if r.get("frames_target") == 2 or "stream_gen_path" in r:
            if r.get("stream_gen_path"):
                by_clip[r["clip_index"]] = r
    return [by_clip[k] for k in sorted(by_clip)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="dir with clip_details.jsonl + streams/")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--slice-max-mbs", type=int, default=1)
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows_for(args.results_dir / "clip_details.jsonl")
    if not rows:
        sys.exit(
            "No rows with stream_gen_path. These results predate stream persistence; "
            "re-run eval_stream_continuation.py to produce streams/."
        )

    clip_records: list[dict[str, Any]] = []
    totals: Counter = Counter()
    stop_reasons: Counter = Counter()
    deltas: list[int] = []
    tc_mismatch = 0
    tc_compared = 0
    for r in rows:
        gen = Path(r["stream_gen_path"]).read_bytes()
        gt = Path(r["stream_gt_path"]).read_bytes()
        res = analyze_stream(
            gen, gt, r["n_prefix_bytes"],
            slice_max_mbs=args.slice_max_mbs,
            first_desync=r.get("first_desync"),
        )
        stop_reasons[r.get("stop_reason") or "unknown"] += 1
        totals.update(res["classification_hist"])
        for n in res["nals"]:
            if "delta_bytes" in n:
                deltas.append(n["delta_bytes"])
            if "total_coeff_match" in n:
                tc_compared += 1
                tc_mismatch += int(not n["total_coeff_match"])
        clip_records.append({
            "clip_index": r["clip_index"],
            "stop_reason": r.get("stop_reason"),
            "desync_region": r.get("desync_region"),
            "desync_reason": r.get("desync_reason"),
            "first_desync": r.get("first_desync"),
            **res,
        })

    with (out_dir / "nal_termination_clips.jsonl").open("w") as fh:
        for rec in clip_records:
            fh.write(json.dumps(rec) + "\n")
    summary = {
        "n_clips": len(clip_records),
        "classification_hist": dict(totals),
        "stop_reason_hist": dict(stop_reasons),
        "premature_boundaries": totals.get(PREMATURE, 0),
        "delta_bytes_nonzero": sum(1 for d in deltas if d != 0),
        "delta_bytes_n": len(deltas),
        "total_coeff_mismatch": tc_mismatch,
        "total_coeff_compared": tc_compared,
        "parser_automaton_contradictions": sum(
            len(c["parser_automaton_contradictions"]) for c in clip_records
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"clips: {summary['n_clips']}")
    print(f"stop_reason      : {summary['stop_reason_hist']}")
    print(f"classification   : {summary['classification_hist']}")
    print(f"premature bounds : {summary['premature_boundaries']}")
    print(f"delta_bytes != 0 : {summary['delta_bytes_nonzero']}/{summary['delta_bytes_n']}")
    print(f"TotalCoeff wrong : {summary['total_coeff_mismatch']}/{summary['total_coeff_compared']}")
    print(f"parser/automaton contradictions: {summary['parser_automaton_contradictions']}")
    print(f"-> {out_dir}/nal_termination_clips.jsonl")


if __name__ == "__main__":
    main()
