"""Fail-fast checks for a JPEG-LM pretraining corpus and its NAL index.

The window-FIM sampler can silently concentrate holes in unusually large frames
when ordinary P frames are too small for the configured guard plus minimum gap.
This report checks the persistent index before an expensive GPU job is submitted.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


VCL_NAL_TYPES = (1, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", required=True, type=Path)
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--frame-guard-bytes", type=int, default=64)
    parser.add_argument("--min-p-frame-eligibility", type=float, default=0.5)
    parser.add_argument("--min-manifest-rows", type=int, default=1)
    parser.add_argument("--allow-low-fim-eligibility", action="store_true")
    return parser.parse_args()


def count_manifest_rows(path: Path) -> tuple[int, int, int]:
    total = 0
    usable = 0
    usable_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("h264_path"):
                usable += 1
                usable_paths.add(str(row["h264_path"]))
    return total, usable, len(usable_paths)


def main() -> None:
    args = parse_args()
    if args.fim_min_gap < 1:
        raise SystemExit("--fim-min-gap must be positive")
    if args.frame_guard_bytes < 0:
        raise SystemExit("--frame-guard-bytes must be non-negative")
    if not 0 <= args.min_p_frame_eligibility <= 1:
        raise SystemExit("--min-p-frame-eligibility must be in [0, 1]")
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if not args.nal_index_path.is_file():
        raise SystemExit(f"NAL index not found: {args.nal_index_path}")

    total_rows, usable_rows, unique_usable_paths = count_manifest_rows(args.manifest)
    if usable_rows < args.min_manifest_rows:
        raise SystemExit(
            f"Manifest has {usable_rows:,} usable rows; require at least "
            f"{args.min_manifest_rows:,} for this launch"
        )

    manifest_stat = args.manifest.stat()
    expected_metadata = {
        "manifest_path": str(args.manifest.resolve()),
        "manifest_size": str(manifest_stat.st_size),
        "manifest_mtime_ns": str(manifest_stat.st_mtime_ns),
    }
    with sqlite3.connect(args.nal_index_path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        stale = [
            key for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        ]
        if stale:
            raise SystemExit(
                f"Stale NAL index {args.nal_index_path}; metadata differs for "
                f"{', '.join(stale)}. Rebuild the index."
            )

        indexed_files = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        stats: dict[int, tuple[int, int, float, int, int]] = {}
        for nal_type in VCL_NAL_TYPES:
            rows = connection.execute(
                "SELECT start_code_len, (end - start) FROM nals WHERE nal_type = ?",
                (nal_type,),
            ).fetchall()
            lengths = [length for _, length in rows]
            eligible = [
                length
                for _, length in rows
                if length - args.frame_guard_bytes - 1 >= args.fim_min_gap
            ]
            stats[nal_type] = (
                len(lengths),
                len(eligible),
                len(eligible) / max(len(lengths), 1),
                min(lengths, default=0),
                max(lengths, default=0),
            )

    print("JPEG-LM pretraining corpus preflight")
    print(f"  manifest rows: {total_rows:,} total, {usable_rows:,} usable")
    print(f"  unique usable paths: {unique_usable_paths:,}")
    print(f"  indexed files: {indexed_files:,}")
    if indexed_files < unique_usable_paths:
        raise SystemExit(
            f"NAL index has {indexed_files:,} files but the manifest has "
            f"{unique_usable_paths:,} unique usable paths. Resume indexing."
        )
    print(
        "  FIM requirement: frame bytes >= "
        f"{args.frame_guard_bytes + args.fim_min_gap + 1} "
        f"(guard={args.frame_guard_bytes}, min_gap={args.fim_min_gap})"
    )
    labels = {1: "non-IDR", 5: "IDR"}
    for nal_type in VCL_NAL_TYPES:
        total, eligible, fraction, minimum, maximum = stats[nal_type]
        print(
            f"  {labels[nal_type]} VCL proxy: {eligible:,}/{total:,} eligible "
            f"({fraction:.1%}); NAL bytes min={minimum:,}, max={maximum:,}"
        )

    p_fraction = stats[1][2]
    if p_fraction < args.min_p_frame_eligibility:
        message = (
            f"Only {p_fraction:.1%} of non-IDR VCL NALs can host a FIM hole; "
            f"minimum accepted is {args.min_p_frame_eligibility:.1%}. This would "
            "bias FIM training toward large/IDR frames. Measure the finished corpus "
            "and choose the hole/guard settings deliberately."
        )
        if not args.allow_low_fim_eligibility:
            raise SystemExit(message)
        print(f"  WARNING: {message}")

    print("preflight passed")


if __name__ == "__main__":
    main()
