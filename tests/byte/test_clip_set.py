from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.byte.eval.helpers.clip_set import (
    load_clip_identifiers,
    match_manifest_rows,
)


def test_loads_only_ordered_continuation_paths_from_prior_details(tmp_path: Path):
    path = tmp_path / "clip_details.jsonl"
    records = [
        {"mode": "continuation", "h264_path": "/old/h264/part1/a.h264"},
        {"mode": "teacher_forced", "h264_path": "/old/h264/part1/ignored.h264"},
        {"mode": "continuation", "h264_path": "/old/h264/part2/b.h264"},
        {"mode": "continuation", "h264_path": "/old/h264/part1/a.h264"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    assert load_clip_identifiers(path) == [
        "/old/h264/part1/a.h264",
        "/old/h264/part2/b.h264",
    ]


def test_matches_same_clips_across_corpus_roots_in_requested_order():
    rows = [
        {"h264_path": "/new/data/h264/part1/a.h264"},
        {"h264_path": "/new/data/h264/part2/b.h264"},
    ]
    matched = match_manifest_rows(
        rows,
        [
            "/old/data-avclm/h264/part2/b.h264",
            "/old/data-avclm/h264/part1/a.h264",
        ],
    )
    assert [row["h264_path"] for row in matched] == [
        "/new/data/h264/part2/b.h264",
        "/new/data/h264/part1/a.h264",
    ]


def test_missing_clip_fails_instead_of_silently_substituting_another_video():
    with pytest.raises(ValueError, match="not found"):
        match_manifest_rows(
            [{"h264_path": "/new/data/h264/part1/a.h264"}],
            ["/old/data-avclm/h264/part1/missing.h264"],
        )
