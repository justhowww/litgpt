"""Summarize whole-clip BSCV eval into the density x GOP-position matrix.

Reads every ``summary.csv`` under an eval root (one per ``corrprob_<n>_<mode>/``
subdir written by ``bscv_eval_h100.sbatch``) and prints, per checkpoint:

1. The concealment floor: ``deleted_default`` PSNR by corr_prob x GOP bucket.
   Where this collapses (far bucket, high corr_prob, early mode) is where
   propagation actually hurts.
2. The model lift: ``model_default - deleted_default`` PSNR on the same grid.
   Positive where the model stops the cascade; ~0 means concealment suffices.

Usage:
    python scripts/byte/eval/summarize_bscv.py RUN_DIR/offline_bscv_eval
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

BUCKETS = ("overall", "idr", "near", "mid", "far")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Eval root containing corrprob_*/summary.csv")
    parser.add_argument("--floor-method", default="deleted_default", help="Baseline method (the concealment floor).")
    parser.add_argument("--model-method", default="model_default", help="Model method compared against the floor.")
    return parser.parse_args()


def to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def psnr_key(method: str, bucket: str) -> str:
    return f"{method}_psnr_mean" if bucket == "overall" else f"{method}_gop_{bucket}_psnr_mean"


def frames_key(method: str, bucket: str) -> str:
    return None if bucket == "overall" else f"{method}_gop_{bucket}_frames"


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(root.glob("**/summary.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(csv_path.parent.name)
                rows.append(row)
    if not rows:
        raise SystemExit(f"No summary.csv found under {root}")
    return rows


def fmt(value: float) -> str:
    return "   -  " if math.isnan(value) else f"{value:6.2f}"


def setting_key(row: dict[str, Any]) -> tuple[int, str]:
    corr_prob = int(to_float(row.get("corr_prob"))) if not math.isnan(to_float(row.get("corr_prob"))) else -1
    mode = row.get("gop_position_mode", "?")
    return corr_prob, mode


def print_table(title: str, checkpoint: str, settings: list[tuple[int, str]], cells: dict[tuple[int, str], list[float]]) -> None:
    print(f"\n  {title}")
    header = "    corr_prob/mode   " + "".join(f"{b:>8}" for b in BUCKETS)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for corr_prob, mode in settings:
        label = f"{corr_prob:>3} {mode:<8}"
        values = "".join(fmt(v) + "  " for v in cells[(corr_prob, mode)])
        print(f"    {label}    {values}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root)

    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[row.get("checkpoint", "?")].append(row)

    print(f"BSCV whole-clip eval: {args.root}")
    print(f"floor = {args.floor_method}   model = {args.model_method}   metric = PSNR (dB)")

    for checkpoint in sorted(by_checkpoint):
        crows = by_checkpoint[checkpoint]
        settings = sorted({setting_key(r) for r in crows})
        row_by_setting = {setting_key(r): r for r in crows}

        floor_cells: dict[tuple[int, str], list[float]] = {}
        lift_cells: dict[tuple[int, str], list[float]] = {}
        sparse: list[str] = []
        for key in settings:
            row = row_by_setting[key]
            floor_vals, lift_vals = [], []
            for bucket in BUCKETS:
                floor = to_float(row.get(psnr_key(args.floor_method, bucket)))
                model = to_float(row.get(psnr_key(args.model_method, bucket)))
                floor_vals.append(floor)
                lift_vals.append(model - floor)
                fk = frames_key(args.model_method, bucket)
                if fk is not None:
                    n = to_float(row.get(fk))
                    if not math.isnan(n) and 0 < n < 10:
                        sparse.append(f"{key[0]}/{key[1]} {bucket}={int(n)}")
            floor_cells[key] = floor_vals
            lift_cells[key] = lift_vals

        match_rate = to_float(row_by_setting[settings[0]].get(f"{args.model_method}_frame_count_match_rate"))
        clips = row_by_setting[settings[0]].get("num_clips", "?")
        print(f"\n=== {checkpoint}  (clips={clips}, model frame-count-match={fmt(match_rate).strip()})")
        print_table(f"Concealment floor — {args.floor_method} PSNR", checkpoint, settings, floor_cells)
        print_table(f"Model lift — ({args.model_method} - {args.floor_method}) PSNR", checkpoint, settings, lift_cells)
        if sparse:
            print(f"    (sparse buckets, <10 frames: {', '.join(sparse)})")


if __name__ == "__main__":
    main()
