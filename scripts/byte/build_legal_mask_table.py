"""Build a table of legal-next-byte masks for a ground-truth H.264 byte corpus.

WHAT THIS COMPUTES

For each ground-truth H.264 file, this script reads through the file once, byte
by byte, and at every position asks: "given everything that came before this
point, which of the 256 possible byte values could come next without breaking
the H.264 bitstream syntax?" The answer at each position is a list of 256
true/false flags, one per possible byte value. This reuses the exact same
checker that is already used at generation time to keep sampled output valid
(litgpt.byte.h264_mask), so nothing new is being taught to the checker here --
this script only runs it once, offline, walking the real data instead of
generated data, and saves the results.

Most positions in a file share the same answer as many other positions -- for
example, every position inside a file header allows all 256 byte values,
because the checker does not attempt to constrain headers (see "What this mask
does and does not guarantee" below). Because of this repetition, each distinct
256-flag answer is stored once in a shared table, and each file is stored as a
short list of pointers into that shared table, one pointer per byte in the
file. This keeps the output small.

WHAT THIS MASK DOES AND DOES NOT GUARANTEE

Read this before using the output as a training signal, not after.

A byte marked false is guaranteed to break the bitstream if used at that
position. That guarantee is meant to be exact: the checker replays the actual
H.264 bit-level decoding rules for that byte, so a false there should mean a
genuinely broken bitstream, not a guess.

A byte marked true is NOT guaranteed to be safe. It only means the checker did
not find a reason to reject it. There are two different reasons a true can
still be hiding a byte that is actually wrong:

  1. On purpose, for large parts of the file. The checker currently only
     applies its strict rules to the residual/coefficient-coding portion of
     each macroblock -- the part of the syntax that is hardest to get right
     and, not coincidentally, the part earlier evaluation on this project
     (phase 1's train/val results) already identified as where the model's
     accuracy gap actually lives. Everywhere else -- macroblock headers,
     motion vectors, quantization steps, and any slice type the checker does
     not support -- it currently answers "everything is allowed" without
     really checking. This is a known, deliberate simplification of the
     checker itself, not a bug in this script.

  2. By accident, if the checker has a bug, even inside the part it is
     supposed to check strictly. This script tests for one specific kind of
     bug: for every single byte actually present in the real file, it checks
     that the checker marks that real byte as allowed at its own position. If
     the checker ever says the real, true byte was not allowed, something is
     wrong with the checker, and this script stops immediately with an error
     rather than silently saving a table that contradicts the very data it was
     built from. This is a useful, cheap check, but it only ever tests the one
     byte that actually appears in the real file at each position -- it says
     nothing about whether the other 255 possible byte values at that same
     position are being judged correctly. A checker could pass this test at
     every single position in the entire corpus and still be wrong about some
     of those other byte values.

So: a false is trustworthy. A true is a "no objection," not a certificate of
correctness. If this table is later used to shape a training loss (for
example, penalizing the model for putting probability on bytes marked false),
that loss can be trusted to push the model away from bytes that are definitely
wrong. It cannot be trusted to guarantee that whatever probability is left
over is sitting only on bytes that are definitely right -- some of it may be
sitting on bytes nobody has actually verified.

TWO FURTHER LIMITS, SPECIFIC TO HOW THIS TABLE WOULD BE USED IN TRAINING

  - This table is only meaningful in the file's original byte order. Some
    training setups (fill-in-the-middle, where the model sees a prefix and a
    suffix and has to guess what belongs in between) present bytes to the
    model in a different order than they appear in the file. This table
    should not be applied to that rearranged order without separately working
    out what "legal" even means once the byte order has changed.

  - This table is built by reading each file from its very first byte. A
    training example, however, is usually a short window cut out of the
    middle of a longer recording, and the model only ever sees the bytes
    inside that window -- it has no memory of anything before the window
    started. If knowing whether a byte is legal at some position genuinely
    requires information from before the window (for instance, which set of
    encoding parameters is active), then the model is being asked to predict
    something it cannot actually know from what it was shown, and using this
    table as a training target there could do more harm than good. There is
    a reason to think this may not be a large problem for this particular
    corpus -- elsewhere in this project, the way this footage was encoded (one
    macroblock per independently-decodable slice) is described as resetting
    cleanly every few bytes, which would mean very little needs to be
    remembered from outside the window -- but that has not actually been
    checked yet, and it should be checked directly (for example, by rebuilding
    the mask starting fresh from a window's own bytes and comparing it against
    this whole-file table) before this table is trusted as a training target.

OUTPUT FORMAT

A single SQLite file (same style as build_nal_index.py):
  metadata(key, value)              -- bookkeeping about which manifest this came from
  mask_table(mask_id, mask_bits)    -- one row per distinct 256-flag answer, packed into 32 bytes
  files(id, path, size, mtime_ns, num_bytes)  -- one row per source file
  file_masks(file_id, mask_ids)     -- one row per file: a list of pointers into mask_table,
                                        one pointer per byte in the file, stored as 32-bit
                                        numbers in a fixed (little-endian) byte order

How to line this table up with a training example: pointer number i in a
file's list describes the byte at position i in that file, using only the
information available up to (not including) position i -- this matches, byte
for byte, the "label" tensor this project's training code already builds for
that same window (see ByteStreamWindowDataset in litgpt/byte/data.py), with no
extra shift needed. For a training window that starts at position
`window_start` in the file, take pointers `window_start` through
`window_start + window_len` and line them up one-to-one with that window's
labels -- do not add or subtract one from these positions; this project's
training code does not need that adjustment, even though a generic
next-byte-prediction setup elsewhere might.

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
from itertools import repeat
from pathlib import Path
from typing import Iterator

from litgpt.byte.data import load_manifest_rows
from litgpt.byte.h264_mask import (
    SLICE_LAYOUT_MACROBLOCK,
    SLICE_LAYOUTS,
    MaskState,
    advance,
    get_valid_byte_mask,
    slice_max_mbs_for_layout,
)

DEFAULT_MASK_TABLE_NAME = "legal_mask_table.sqlite"

if array("I").itemsize != 4:
    raise RuntimeError(
        "This format requires 32-bit unsigned array('I') elements on this platform"
    )


def default_mask_table_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(DEFAULT_MASK_TABLE_NAME)


def iter_legal_masks(
    data: bytes, slice_layout: str = SLICE_LAYOUT_MACROBLOCK
) -> Iterator[list[bool]]:
    """Walk one raw H.264 file byte by byte and, for each position, report which
    of the 256 possible byte values could legally appear there, given only the
    real bytes that came before it. Returns exactly one 256-entry answer per
    byte in the file, in file order.

    This does not add any new checking logic. It calls the same
    MaskState/get_valid_byte_mask/advance functions already used to keep
    generated output valid at inference time (see free_run_eval.py), just
    pointed at real recorded bytes instead of bytes the model is generating.

    Before moving on to the next byte, this function checks one thing about the
    answer it just got: that the real byte actually present in the file is one
    of the ones marked legal. If that is ever false, it means the checker
    disagrees with itself about a byte it just watched happen -- a bug in the
    checker, not a problem with this particular file -- so this function stops
    immediately with an error instead of quietly saving a wrong answer.

    This check is useful but narrow: it only ever looks at the one byte that
    really occurs at each position. It says nothing about whether the other
    255 byte values at that same position have been correctly marked legal or
    illegal. See the "What this mask does and does not guarantee" section at
    the top of this file for what that limitation means in practice.
    """
    state = MaskState(slice_max_mbs=slice_max_mbs_for_layout(slice_layout))
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


def scan_file(
    path_string: str, slice_layout: str = SLICE_LAYOUT_MACROBLOCK
) -> tuple[str, int, int, array, list[bytes]]:
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
    for i, mask in enumerate(iter_legal_masks(data, slice_layout)):
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
    manifest_path: Path,
    output_path: Path,
    workers: int,
    rebuild: bool,
    slice_layout: str = SLICE_LAYOUT_MACROBLOCK,
) -> None:
    rows = load_manifest_rows(manifest_path, report_progress=True)
    paths = list(dict.fromkeys(str(Path(row["h264_path"])) for row in rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(output_path) as connection:
        initialize_database(connection, rebuild)
        stored_layout = connection.execute(
            "SELECT value FROM metadata WHERE key = 'slice_layout'"
        ).fetchone()
        existing_count = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        if (
            existing_count
            and stored_layout is not None
            and stored_layout[0] != slice_layout
        ):
            raise RuntimeError(
                f"Legal-mask table was built with slice_layout={stored_layout[0]!r}; "
                f"requested {slice_layout!r}. Use --rebuild or a different --output."
            )
        if (
            existing_count
            and stored_layout is None
            and slice_layout != SLICE_LAYOUT_MACROBLOCK
        ):
            raise RuntimeError(
                "Existing legal-mask table predates slice-layout metadata. "
                "Use --rebuild before building frame-layout masks."
            )
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
            results = map(scan_file, pending_paths, repeat(slice_layout))
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(
                scan_file, pending_paths, repeat(slice_layout), chunksize=4
            )

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
            "slice_layout": slice_layout,
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
    parser.add_argument(
        "--slice-layout",
        choices=SLICE_LAYOUTS,
        default=SLICE_LAYOUT_MACROBLOCK,
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    output = args.output or default_mask_table_path(args.manifest)
    build_table(
        args.manifest,
        output,
        args.workers,
        args.rebuild,
        slice_layout=args.slice_layout,
    )


if __name__ == "__main__":
    main()
