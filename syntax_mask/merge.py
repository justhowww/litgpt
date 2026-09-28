"""Merge per-shard ``--out-jsonl`` files from a sharded scan into one summary.

    python -m syntax_mask.merge RUN_DIR/shard_*.jsonl --json RUN_DIR/summary.json

Exits non-zero if any file failed, any reference check mismatched, or any probe
hit a dead end.  ``first_failures`` lists (path, offset, field, error) to debug.
"""

from __future__ import annotations

import argparse
import json
import sys

from .scan import _report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--json", help="write merged summary (per-file rows omitted)")
    ap.add_argument("--layout", default="frame")
    args = ap.parse_args(argv)
    results = {}
    for name in args.jsonl:
        with open(name, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results[r["path"]] = r  # last write wins after a resume
    rows = list(results.values())
    failures = [r for r in rows if not r["ok"]]
    status = _report(rows, args.layout, 0.0, None)
    if args.json:
        from .scan import summarize
        summary = summarize(rows)
        summary["failed_files"] = [{"path": r["path"], **r["failure"]} for r in failures]
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
    return status


if __name__ == "__main__":
    sys.exit(main())
