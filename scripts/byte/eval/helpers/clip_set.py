"""Load and match reproducible evaluation clip sets across encoded corpora."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


def _path_from_record(record: Any) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    if record.get("mode") not in (None, "continuation"):
        return None
    value = record.get("h264_path") or record.get("path")
    return str(value) if value else None


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def load_clip_identifiers(path: Path) -> list[str]:
    """Read H.264 paths from text, JSON, or prior ``clip_details.jsonl`` output."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"clip list does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        values: list[str] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            value = _path_from_record(record)
            if value is not None:
                values.append(value)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("videos", payload.get("clips", []))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a list or a videos/clips list")
        values = [
            value
            for record in payload
            if (value := _path_from_record(record)) is not None
        ]
    else:
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    identifiers = _deduplicate(values)
    if not identifiers:
        raise ValueError(f"clip list contains no continuation H.264 paths: {path}")
    return identifiers


def h264_relative_key(value: str | Path) -> str | None:
    """Return the stable path below the corpus's ``h264/`` directory."""
    normalized = str(value).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    h264_positions = [index for index, part in enumerate(parts) if part == "h264"]
    if h264_positions:
        suffix = parts[h264_positions[-1] + 1 :]
        return "/".join(suffix) if suffix else None
    if not PurePosixPath(normalized).is_absolute() and "/" in normalized:
        return normalized.lstrip("./")
    return None


def match_manifest_rows(
    rows: list[dict[str, Any]], identifiers: list[str]
) -> list[dict[str, Any]]:
    """Map an ordered clip set to current-manifest rows.

    Exact paths are preferred. Otherwise paths are matched by their stable suffix
    below ``h264/``, allowing the same source clip to move from ``data-avclm`` to
    ``data``. A bare filename is accepted only when it is unique in the manifest.
    """
    exact: dict[str, dict[str, Any]] = {}
    relative: dict[str, dict[str, Any]] = {}
    basenames: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        current = str(Path(row["h264_path"]))
        exact[current] = row
        key = h264_relative_key(current)
        if key is not None:
            if key in relative and relative[key]["h264_path"] != row["h264_path"]:
                raise ValueError(f"manifest has duplicate h264-relative path: {key}")
            relative[key] = row
        basenames.setdefault(Path(current).name, []).append(row)

    matched: list[dict[str, Any]] = []
    matched_paths: set[str] = set()
    missing: list[str] = []
    ambiguous: list[str] = []
    for identifier in identifiers:
        current_identifier = str(Path(identifier))
        row = exact.get(current_identifier)
        if row is None:
            key = h264_relative_key(identifier)
            row = relative.get(key) if key is not None else None
        if row is None:
            candidates = basenames.get(Path(identifier).name, [])
            if len(candidates) == 1:
                row = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append(identifier)
                continue
        if row is None:
            missing.append(identifier)
            continue
        current_path = str(Path(row["h264_path"]))
        if current_path not in matched_paths:
            matched_paths.add(current_path)
            matched.append(row)

    if missing or ambiguous:
        parts: list[str] = []
        if missing:
            parts.append(
                "not found after manifest/train-split filtering: "
                + ", ".join(missing[:5])
            )
        if ambiguous:
            parts.append(
                "ambiguous bare filenames: " + ", ".join(ambiguous[:5])
            )
        raise ValueError("; ".join(parts))
    return matched
