#!/usr/bin/env python3
"""Randomized CAVLC coeff_token/dependent-field round-trip sanity check.

This deliberately does NOT splice variable-length fields into a real generated NAL:
doing so shifts every following bit and would conflate coefficient consistency with NAL
rewriting. Instead, for each coeff_token BitReaderError reported by analyze_nal_termination.py,
generate complete random residual blocks under the same nC-selected table and max_coeff,
then decode them with the recursive parser's _residual_block.

The generated subset uses TotalCoeff in 0..3 with TrailingOnes == TotalCoeff. Therefore
all nonzero coefficients are valid trailing +/-1 values; total_zeros and run_before are
sampled consistently. This directly tests the hypothesis that a correctly completed
coeff_token trajectory does not itself run off the RBSP. Failures in mvd/ref_idx/
mb_skip_run are printed as NOT_APPLICABLE rather than mislabeled coefficient cases.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

_BYTE_DIR = Path(__file__).resolve().parents[3] / "litgpt" / "byte"


def _load_byte_modules():
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

    tables = load("h264_cavlc_tables")
    syntax = load("h264_syntax")
    return tables, syntax


T, HS = _load_byte_modules()


def _code_for(label: str, value) -> list[int]:
    matches = [code for code, decoded in T.code_map(label).items() if decoded == value]
    if len(matches) != 1:
        raise AssertionError(f"{label}: expected one code for {value!r}, got {matches}")
    return [int(bit) for bit in matches[0]]


def _pack(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def random_block_bits(rng: random.Random, nc: int, max_coeff: int) -> tuple[list[int], dict]:
    """Generate a complete valid residual block with no non-trailing levels.

    Restricting TotalCoeff <= 3 and setting TrailingOnes == TotalCoeff removes level
    encoding from this first-principles sanity check while retaining the state changes
    named in the hypothesis: (TotalCoeff, TrailingOnes), total_zeros and run_before.
    """
    total_coeff = rng.randint(0, min(3, max_coeff))
    trailing_ones = total_coeff
    label = T.coeff_token_label(nc)
    bits = _code_for(label, (total_coeff, trailing_ones))
    meta = {
        "nC": nc,
        "max_coeff": max_coeff,
        "total_coeff": total_coeff,
        "trailing_ones": trailing_ones,
    }
    if total_coeff == 0:
        return bits, meta

    # One random sign per trailing +/-1 coefficient.
    bits.extend(rng.randrange(2) for _ in range(trailing_ones))

    total_zeros = 0
    runs: list[int] = []
    if total_coeff < max_coeff:
        tz_label = (
            f"total_zeros_cdc_{total_coeff}"
            if max_coeff == 4
            else f"total_zeros_4x4_{total_coeff}"
        )
        tz_values = sorted(
            v for v in set(T.code_map(tz_label).values())
            if v <= max_coeff - total_coeff
        )
        total_zeros = rng.choice(tz_values)
        bits.extend(_code_for(tz_label, total_zeros))

        zeros_left = total_zeros
        for _ in range(total_coeff - 1):
            if zeros_left <= 0:
                break
            rb_label = f"run_before_{min(zeros_left, 7)}"
            legal = [v for v in set(T.code_map(rb_label).values()) if v <= zeros_left]
            run = rng.choice(sorted(legal))
            bits.extend(_code_for(rb_label, run))
            runs.append(run)
            zeros_left -= run

    meta.update(total_zeros=total_zeros, runs=runs)
    return bits, meta


def round_trip_once(rng: random.Random, nc: int, max_coeff: int) -> dict:
    bits, meta = random_block_bits(rng, nc, max_coeff)
    rbsp = _pack(bits)
    reader = HS.BitReader(rbsp)
    record = HS._Recorder(list(range(len(rbsp))), len(rbsp))
    decoded_tc = HS._residual_block(
        reader, record, nc, max_coeff, 0, HS.Category.RESIDUAL_LUMA, "sanity"
    )
    if decoded_tc != meta["total_coeff"]:
        raise AssertionError(f"TotalCoeff {decoded_tc} != {meta['total_coeff']}: {meta}")
    if reader.pos != len(bits):
        raise AssertionError(f"consumed {reader.pos} bits, expected {len(bits)}: {meta}")
    return meta


def run_trials(nc: int, max_coeff: int, trials: int, seed: int) -> dict:
    rng = random.Random(seed)
    errors: list[dict] = []
    hist: dict[str, int] = {}
    for i in range(trials):
        try:
            meta = round_trip_once(rng, nc, max_coeff)
            key = f"TC={meta['total_coeff']},T1={meta['trailing_ones']}"
            hist[key] = hist.get(key, 0) + 1
        except Exception as exc:  # diagnostic: retain the exact randomized case
            errors.append({"trial": i, "kind": type(exc).__name__, "reason": str(exc)})
    return {"nC": nc, "max_coeff": max_coeff, "trials": trials, "errors": errors, "hist": hist}


def _failure_reports(results_dir: Path) -> list[dict]:
    summary = json.loads((results_dir / "summary.json").read_text())
    return summary.get("first_desync_reports", [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="analyze_nal_termination.py results directory")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args()

    reports = [r for r in _failure_reports(args.results_dir) if r.get("desync_reason") == "BitReaderError"]
    cases: dict[tuple[int, int], list[int]] = {}
    print("failure applicability:")
    for r in reports:
        element = r.get("failure_element") or "unknown"
        context = r.get("failure_context") or {}
        if element.endswith(".coeff_token") and "nC" in context and "max_coeff" in context:
            key = (int(context["nC"]), int(context["max_coeff"]))
            cases.setdefault(key, []).append(int(r["clip_index"]))
            print(f"  clip={r['clip_index']} {element}: TEST nC={key[0]} max_coeff={key[1]}")
        else:
            print(f"  clip={r['clip_index']} {element}: NOT_APPLICABLE (failure is before/after coeff_token)")

    results = []
    for case_i, ((nc, max_coeff), clips) in enumerate(sorted(cases.items())):
        result = run_trials(nc, max_coeff, args.trials, args.seed + case_i)
        result["clips"] = clips
        results.append(result)
        print(
            f"nC={nc} max_coeff={max_coeff} clips={clips}: "
            f"{args.trials - len(result['errors'])}/{args.trials} passed, "
            f"BitReaderError={sum(e['kind'] == 'BitReaderError' for e in result['errors'])}"
        )

    out = args.results_dir / "cavlc_coeff_sanity.json"
    out.write_text(json.dumps({"reports": reports, "results": results}, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
