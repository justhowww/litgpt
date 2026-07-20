"""Build a deduplicated legal-next-byte mask table for a GT H.264 byte corpus.

Offline preprocessing: for each GT H.264 (Annex-B) file, walk it once with the
same constrained-decoding automaton already used for inference-time masked
generation (litgpt.byte.h264_mask -- MaskState / get_valid_byte_mask / advance,
the exact machinery that takes free-run full_continuation_rate from 0.750
unconstrained to 1.000 masked, see 260716 - Eval with syntax validation mask.md),
computing the legal-next-byte mask at every byte position BEFORE that byte is
consumed. Masks are deduplicated into a global mask_id -> 256-bit legal-set table
(most positions repeat a handful of masks -- permissive header/non-residual
positions are all identical [True]*256, and residual-coding positions repeat
structurally across the corpus), and each file's byte stream is stored as a
compact array of mask_ids.

Output is a single SQLite file (schema mirrors build_nal_index.py's style):
  metadata(key TEXT PRIMARY KEY, value TEXT)
  mask_table(mask_id INTEGER PRIMARY KEY, mask_bits BLOB UNIQUE)   -- 32-byte packed bool[256]
  files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, size INTEGER, mtime_ns INTEGER, num_bytes INTEGER)
  file_masks(file_id INTEGER PRIMARY KEY, mask_ids BLOB)           -- uint32[num_bytes], little-endian
                                                                    -- (enforced explicitly, see write_file_result)

Alignment for consumers: mask_ids[i] is the legal-byte set for data[i] GIVEN THE
TRUE PREFIX data[:i] (i.e. "before that byte is consumed", per iter_legal_masks
below) -- it already lines up index-for-index with THIS PROJECT's actual label
convention (litgpt/byte/data.py's ByteStreamWindowDataset: labels[t] == window[t],
input_ids[t] == window[t-1] via a separate BOS-prepended tensor, NOT an
overlapping slice of one array). For a training window starting at file offset
`window_start`, the correct slice is
`mask_ids[window_start : window_start + window_len]` matched directly against
that window's `labels` tensor, index for index -- NOT shifted by one relative to
`window_start`. Do not re-derive this against a generic "targets = data[start+1:]"
convention; it does not apply to this codebase's actual tensor layout.

Known limitations (not resolved by this script -- read before wiring into a loss):
  - AR order only. FIM training rearranges the presented byte order (prefix,
    hole marker, suffix, ..., middle), so a mask computed against TRUE file order
    is only valid for AR-ordered positions. Do not apply these masks to FIM
    sequence positions without separately re-deriving what's legal in the
    FIM-rearranged frame of reference.
  - Windowing/context visibility. This script computes masks using automaton
    state accumulated from byte 0 of each FILE. A later TRAINING window may
    start mid-file; the automaton state the mask depends on is only inferable
    from the model's own visible input if everything that state depends on
    (active SPS/PPS, any carried CAVLC neighbor context) is itself visible
    within the window. This project's per-MB corpus (slice_max_mbs=1) is
    described elsewhere (phase2_fim/train.sh) as "a desync firewall every ~10
    bytes", which suggests little state should actually carry past a NAL/slice
    boundary -- but that has NOT been verified here. Before trusting this table
    as a training target, verify it: recompute masks starting fresh from a
    window's own bytes at several real window-start points and diff against
    this whole-file table.
  - Exactness is bounded by litgpt.byte.h264_mask/h264_automaton's own
    correctness, which this script does not re-verify -- treat the output as
    "automaton-admissible next-byte mask", upgraded in confidence only by a
    full corpus run completing without tripping the assertion in
    iter_legal_masks below (a completed run is evidence the automaton agrees
    with itself on real content, not just synthetic test fixtures).

Usage:
    python scripts/byte/build_legal_mask_table.py manifest.jsonl
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

from litgpt.byte.data import load_manifest_rows
from litgpt.byte.h264_mask import MaskState, advance, get_valid_byte_mask

DEFAULT_MASK_TABLE_NAME = "legal_mask_table.sqlite"

if array("I").itemsize != 4:
    raise RuntimeError(
        "This format requires 32-bit unsigned array('I') elements on this platform"
    )


def default_mask_table_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(DEFAULT_MASK_TABLE_NAME)


def iter_legal_masks(data: bytes) -> Iterator[list[bool]]:
    """Walk a raw Annex-B byte stream, yielding the legal-next-byte mask before
    each byte is consumed. len(list(iter_legal_masks(data))) == len(data).

    Reuses MaskState/get_valid_byte_mask/advance exactly as free_run_eval.py does
    at inference time, just walking the real GT bytes instead of sampled ones --
    no new CAVLC parsing logic, only a new (offline, GT-driven) use of it.

    Asserts the fundamental invariant x_i in A(x_<i) for every i: the GT byte
    must always be in its own predicted-legal set. A violation means the
    automaton's trie-derived mask disagrees with its own bit-level transitions
    -- a real automaton bug -- and must fail loudly rather than silently write a
    corrupted mask table.
    """
    state = MaskState()
    for offset, byte in enumerate(data):
        mask = get_valid_byte_mask(state)
        if len(mask) != 256:
            raise RuntimeError(
                f"Expected 256-entry mask at byte {offset}, got {len(mask)}"
            )
        if not mask[byte]:
            raise RuntimeError(
                f"GT byte 0x{byte:02x} is marked illegal at offset {offset} "
                "-- automaton/mask disagreement, not a data problem"
            )
        yield mask
        advance(state, byte)


def pack_mask(mask: list[bool]) -> bytes:
    """Pack a 256-entry bool mask into 32 bytes (big-endian bit order), so it's a
    hashable, compact dict key / SQLite BLOB for deduplication."""
    value = 0
    for bit in mask:
        value = (value << 1) | (1 if bit else 0)
    return value.to_bytes(32, "big")


def unpack_mask(packed: bytes) -> list[bool]:
    value = int.from_bytes(packed, "big")
    return [(value >> (255 - i)) & 1 == 1 for i in range(256)]


def scan_file(path_string: str) -> tuple[str, int, int, array, list[bytes]]:
    """Compute per-byte legal masks for one file, deduplicated LOCALLY (per file).

    Global (cross-file) dedup happens in the writer, which is single-threaded
    since SQLite writes must be serialized anyway -- this keeps worker processes
    independent and avoids shipping a shared dedup table across process
    boundaries. Returns (path, size, mtime_ns, local_mask_ids, local_masks) where
    local_mask_ids[i] indexes into local_masks (list of 32-byte packed masks).
    """
    path = Path(path_string)
    data = path.read_bytes()
    stat = path.stat()

    local_table: dict[bytes, int] = {}
    local_masks: list[bytes] = []
    local_mask_ids = array("I", [0]) * len(data)
    for i, mask in enumerate(iter_legal_masks(data)):
        packed = pack_mask(mask)
        local_id = local_table.get(packed)
        if local_id is None:
            local_id = len(local_masks)
            local_table[packed] = local_id
            local_masks.append(packed)
        local_mask_ids[i] = local_id

    return path_string, stat.st_size, stat.st_mtime_ns, local_mask_ids, local_masks


def initialize_database(connection: sqlite3.Connection, rebuild: bool) -> None:
    if rebuild:
        connection.executescript(
            "DROP TABLE IF EXISTS file_masks; DROP TABLE IF EXISTS files; "
            "DROP TABLE IF EXISTS mask_table; DROP TABLE IF EXISTS metadata;"
        )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mask_table (
            mask_id INTEGER PRIMARY KEY,
            mask_bits BLOB NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            num_bytes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_masks (
            file_id INTEGER PRIMARY KEY,
            mask_ids BLOB NOT NULL,
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        """
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.commit()


def get_or_create_global_mask_id(
    connection: sqlite3.Connection, cache: dict[bytes, int], mask_bits: bytes
) -> int:
    cached = cache.get(mask_bits)
    if cached is not None:
        return cached
    row = connection.execute(
        "SELECT mask_id FROM mask_table WHERE mask_bits = ?", (mask_bits,)
    ).fetchone()
    if row is not None:
        cache[mask_bits] = row[0]
        return row[0]
    cursor = connection.execute(
        "INSERT INTO mask_table(mask_bits) VALUES (?)", (mask_bits,)
    )
    mask_id = cursor.lastrowid
    assert mask_id is not None
    cache[mask_bits] = mask_id
    return mask_id


def write_file_result(
    connection: sqlite3.Connection,
    global_cache: dict[bytes, int],
    result: tuple[str, int, int, array, list[bytes]],
) -> int:
    path, size, mtime_ns, local_mask_ids, local_masks = result

    previous = connection.execute(
        "SELECT id, size, mtime_ns FROM files WHERE path = ?", (path,)
    ).fetchone()
    if previous is not None and previous[1:] == (size, mtime_ns):
        return size

    # Remap this file's LOCAL mask ids to GLOBAL mask ids (deduplicated across
    # the whole corpus, not just within this file).
    local_to_global = [
        get_or_create_global_mask_id(connection, global_cache, m) for m in local_masks
    ]
    global_mask_ids = array("I", (local_to_global[local_id] for local_id in local_mask_ids))

    if previous is None:
        cursor = connection.execute(
            "INSERT INTO files(path, size, mtime_ns, num_bytes) VALUES (?, ?, ?, ?)",
            (path, size, mtime_ns, len(global_mask_ids)),
        )
        file_id = cursor.lastrowid
    else:
        file_id = previous[0]
        connection.execute(
            "UPDATE files SET size = ?, mtime_ns = ?, num_bytes = ? WHERE id = ?",
            (size, mtime_ns, len(global_mask_ids), file_id),
        )

    # array('I') is native-endian, not guaranteed little-endian -- enforce the
    # on-disk format explicitly so the schema's documented "little-endian" claim
    # is actually true, not just true on the platforms this happens to run on
    # today (inert in practice here since Vulcan/x86_64 and this dev machine/
    # arm64 are both little-endian, but readers must not have to assume that).
    if sys.byteorder != "little":
        global_mask_ids.byteswap()
    connection.execute(
        "INSERT OR REPLACE INTO file_masks(file_id, mask_ids) VALUES (?, ?)",
        (file_id, global_mask_ids.tobytes()),
    )
    return size


def read_mask_ids(blob: bytes) -> array:
    """Inverse of write_file_result's serialization: always returns native-endian
    uint32s regardless of the writer's or reader's platform endianness."""
    ids = array("I")
    ids.frombytes(blob)
    if sys.byteorder != "little":
        ids.byteswap()
    return ids


def build_table(
    manifest_path: Path, output_path: Path, workers: int, rebuild: bool
) -> None:
    rows = load_manifest_rows(manifest_path, report_progress=True)
    paths = list(dict.fromkeys(str(Path(row["h264_path"])) for row in rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(output_path) as connection:
        initialize_database(connection, rebuild)
        existing_files = {
            path: (size, mtime_ns)
            for path, size, mtime_ns in connection.execute(
                "SELECT path, size, mtime_ns FROM files"
            )
        }
        pending_paths: list[str] = []
        unchanged_files = 0
        for path_string in paths:
            cached_stat = existing_files.get(path_string)
            if cached_stat is None:
                pending_paths.append(path_string)
                continue
            stat = Path(path_string).stat()
            if cached_stat == (stat.st_size, stat.st_mtime_ns):
                unchanged_files += 1
            else:
                pending_paths.append(path_string)

        # Drop files no longer in the manifest -- mirrors build_nal_index.py's gap
        # (which never does this either), fixed here since this table is meant to
        # represent exactly the current manifest. file_masks rows cascade via FK;
        # mask_table rows for now-unused masks are left in place (harmless, may be
        # reused by future files -- only inflates the table after repeated churn).
        current_paths = set(paths)
        stale_paths = [p for p in existing_files if p not in current_paths]
        for stale_path in stale_paths:
            connection.execute("DELETE FROM files WHERE path = ?", (stale_path,))
        if stale_paths:
            connection.commit()
            print(f"Removed {len(stale_paths):,} files no longer in manifest", flush=True)

        global_cache: dict[bytes, int] = {
            mask_bits: mask_id
            for mask_id, mask_bits in connection.execute(
                "SELECT mask_id, mask_bits FROM mask_table"
            )
        }

        started_at = time.perf_counter()
        processed_bytes = 0
        processed_files = 0
        print(
            f"Building legal-mask table: {unchanged_files:,} cached, "
            f"{len(pending_paths):,} pending -> {output_path}",
            flush=True,
        )

        if workers == 1:
            results = map(scan_file, pending_paths)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(scan_file, pending_paths, chunksize=4)

        try:
            for result in results:
                processed_bytes += write_file_result(connection, global_cache, result)
                processed_files += 1
                if processed_files % 50 == 0:
                    connection.commit()
                    elapsed = time.perf_counter() - started_at
                    print(
                        f"Processed {processed_files:,}/{len(pending_paths):,} new files, "
                        f"{processed_bytes / 1e9:.2f} GB, "
                        f"{processed_bytes / max(elapsed, 1e-9) / 1e6:.1f} MB/s, "
                        f"{len(global_cache):,} unique masks so far",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown()
        connection.commit()

        manifest_stat = manifest_path.stat()
        metadata = {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_size": str(manifest_stat.st_size),
            "manifest_mtime_ns": str(manifest_stat.st_mtime_ns),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.commit()

        n_files, n_masks = connection.execute(
            "SELECT (SELECT COUNT(*) FROM files), (SELECT COUNT(*) FROM mask_table)"
        ).fetchone()

    print(
        f"Legal-mask table complete: {n_files:,} files, {n_masks:,} unique masks -> {output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    output = args.output or default_mask_table_path(args.manifest)
    build_table(args.manifest, output, args.workers, args.rebuild)


if __name__ == "__main__":
    main()
