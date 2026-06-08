"""Atomically merge H.264 manifest shards.

Later rows replace earlier rows with the same source id. This lets each source
directory keep an independent manifest while exposing one corpus-level
manifest.jsonl to training.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
    return rows


def row_key(row: dict[str, Any]) -> str:
    key = row.get("id") or row.get("h264_path")
    if not key:
        raise RuntimeError(f"manifest row has neither id nor h264_path: {row}")
    return str(key)


def merge_manifests(output_path: Path, shard_paths: list[Path]) -> None:
    rows_by_key: dict[str, dict[str, Any]] = {}

    # Preserve legacy rows that have not yet been converted to per-directory
    # shards. New shard rows replace matching legacy rows.
    for row in load_rows(output_path):
        rows_by_key[row_key(row)] = row
    for shard_path in sorted(shard_paths):
        for row in load_rows(shard_path):
            rows_by_key[row_key(row)] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        tmp_path = Path(file.name)
        for key in sorted(rows_by_key):
            file.write(json.dumps(rows_by_key[key]) + "\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(tmp_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("shards", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_manifests(args.output, args.shards)


if __name__ == "__main__":
    main()
