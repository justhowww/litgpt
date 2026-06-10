"""Report target-slice lengths and realized FIM gap sizes from a NAL index.

Read-only. Run where the SQLite NAL index lives (e.g. Zaratan):

    python scripts/inspect_slice_lengths.py /path/to/nal_index.sqlite

It answers "how big is the mask in training?" by mirroring the gap sampling in
ByteSliceDataset._sample_fim_parts: the gap is sampled inside a single target
slice, so it is bounded by the slice payload minus the header guard. fim_max_gap
is only a ceiling. The script reports the P-slice length distribution and the
expected realized gap under one or more fim_max_gap caps, so you can see whether
raising the cap (e.g. 1400 -> 2800) actually changes the training distribution.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from pathlib import Path


def percentiles(values: list[int], points=(0, 5, 25, 50, 75, 90, 95, 99, 100)) -> dict[int, int]:
    if not values:
        return {p: 0 for p in points}
    ordered = sorted(values)
    out: dict[int, int] = {}
    for p in points:
        idx = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
        out[p] = ordered[idx]
    return out


def expected_gap(nal_len: int, start_code_len: int, guard: int, min_gap: int, max_gap_cap: int) -> float:
    """Mean realized gap for one slice under a fim_max_gap cap (mirrors sampling)."""
    protected = min(nal_len - 2, start_code_len + 1 + guard)
    max_gap = min(max_gap_cap, nal_len - protected - 1)
    if max_gap < 1:
        return 1.0
    lo = min(min_gap, max_gap)
    return (lo + max_gap) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_path", type=Path, help="Path to nal_index.sqlite")
    parser.add_argument("--nal-type", type=int, default=1, help="Target VCL type (1=P, 5=IDR/I)")
    parser.add_argument("--guard", type=int, default=64, help="slice_header_guard_bytes")
    parser.add_argument("--min-gap", type=int, default=64, help="fim_min_gap")
    parser.add_argument("--caps", type=int, nargs="+", default=[1400, 2800], help="fim_max_gap values to compare")
    args = parser.parse_args()

    if not args.index_path.is_file():
        raise SystemExit(f"NAL index not found: {args.index_path}")

    conn = sqlite3.connect(str(args.index_path))
    rows = conn.execute(
        "SELECT start_code_len, (end - start) FROM nals WHERE nal_type = ?",
        (args.nal_type,),
    ).fetchall()
    conn.close()

    if not rows:
        raise SystemExit(f"No NALs of type {args.nal_type} found in {args.index_path}")

    nal_lens = [length for _, length in rows]
    # Bytes actually available to mask in each slice (gap ceiling, pre-cap).
    available = [
        max(0, length - min(length - 2, scl + 1 + args.guard) - 1)
        for scl, length in rows
    ]

    n = len(nal_lens)
    print(f"Target NAL type {args.nal_type}: {n:,} slices")
    print(f"Slice length (end-start) bytes: mean={statistics.mean(nal_lens):.0f}  "
          f"median={statistics.median(nal_lens):.0f}  max={max(nal_lens):,}")
    pct = percentiles(nal_lens)
    print("  percentiles: " + "  ".join(f"p{p}={pct[p]:,}" for p in pct))

    print(f"\nMaskable bytes per slice (gap ceiling before cap): "
          f"median={statistics.median(available):.0f}  max={max(available):,}")

    print("\nReach of each fim_max_gap cap:")
    for cap in args.caps:
        reach = sum(a >= cap for a in available)
        binds = sum(a >= cap for a in available)  # cap binds exactly when available >= cap
        mean_gap = statistics.mean(
            expected_gap(length, scl, args.guard, args.min_gap, cap) for scl, length in rows
        )
        print(
            f"  cap={cap:>5}:  {reach:>8,} / {n:,} slices ({100*reach/n:5.1f}%) can reach it; "
            f"mean realized gap = {mean_gap:6.0f} B"
        )

    if len(args.caps) == 2:
        lo_cap, hi_cap = sorted(args.caps)
        affected = sum(a > lo_cap for a in available)
        print(
            f"\nRaising cap {lo_cap} -> {hi_cap}: {affected:,} slices ({100*affected/n:.1f}%) "
            f"have >{lo_cap} maskable bytes, i.e. their gap distribution changes."
        )


if __name__ == "__main__":
    main()
