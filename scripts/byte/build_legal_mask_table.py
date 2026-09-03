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

Exact DP pilot (run separately for ``macroblock`` and ``frame`` corpora):
    python scripts/byte/build_legal_mask_table.py manifest.jsonl \
        --output /tmp/legal-mask-pilot.sqlite --rebuild --max-files 100 \
        --slice-layout frame --mask-compiler compare --workers 1 \
        --statistics-json /tmp/legal-mask-pilot.json \
        --require-committed-grid-crossing

``compare`` is intentionally a validation mode, not a speed benchmark: it runs
both implementations.  After exact equality is established, use ``memoized`` and
repeat with the intended worker count to measure throughput.  The default
``--dp-cache-scope root`` bounds memory to one byte's search tree.  Use
``--dp-cache-scope worker`` only on a small pilot to measure cross-byte reuse.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
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
from litgpt.byte import h264_automaton as HA
from litgpt.byte.h264_mask import (
    SLICE_LAYOUT_MACROBLOCK,
    SLICE_LAYOUTS,
    MaskState,
    advance,
    get_valid_byte_mask,
    slice_max_mbs_for_layout,
)

DEFAULT_MASK_TABLE_NAME = "legal_mask_table.sqlite"
MASK_COMPILERS = ("legacy", "memoized", "compare")
DP_CACHE_SCOPES = ("root", "worker")
_WORKER_COMPILERS = {}

if array("I").itemsize != 4:
    raise RuntimeError(
        "This format requires 32-bit unsigned array('I') elements on this platform"
    )


def default_mask_table_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(DEFAULT_MASK_TABLE_NAME)


class ByteMaskCompiler:
    """Build-script adapter for legacy, memoized, and exact-comparison modes."""

    def __init__(
        self,
        mode: str,
        max_cache_entries: int,
        collect_stats: bool,
        dp_cache_scope: str,
    ):
        if mode not in MASK_COMPILERS:
            raise ValueError(f"unknown mask compiler {mode!r}")
        self.mode = mode
        self.memoized = (
            HA.MemoizedByteMaskCompiler(
                max_cache_entries=max_cache_entries,
                collect_field_cardinality=collect_stats,
                cache_scope=dp_cache_scope,
            )
            if mode != "legacy"
            else None
        )
        self.corpus_bytes = 0
        self.comparisons = 0
        self.mismatches = 0
        self.legacy_transition_computations = 0

    def record_corpus_byte(self):
        self.corpus_bytes += 1
        if self.memoized is not None:
            self.memoized.record_corpus_byte()

    def compile_byte_mask(self, auto, *, residual_only=False):
        def count_legacy_transition():
            self.legacy_transition_computations += 1

        if self.mode == "legacy":
            return HA.compile_byte_mask(
                auto,
                residual_only=residual_only,
                transition_counter=count_legacy_transition,
            )
        memoized = self.memoized.compile_byte_mask(
            auto, residual_only=residual_only
        )
        if self.mode == "memoized":
            return memoized

        legacy = HA.compile_byte_mask(
            auto,
            residual_only=residual_only,
            transition_counter=count_legacy_transition,
        )
        self.comparisons += 1
        if legacy != memoized:
            self.mismatches += 1
            differing = [i for i, (a, b) in enumerate(zip(legacy, memoized)) if a != b]
            raise AssertionError(
                "legacy/memoized legal-byte masks differ at "
                f"stage={auto.stage} syntax={auto.ae_tag} bit={auto.pos}; "
                f"first differing bytes={differing[:16]}"
            )
        return memoized

    def statistics(self):
        result = {
            "mode": self.mode,
            "corpus_bytes": self.corpus_bytes,
            "legacy_comparisons": self.comparisons,
            "legacy_mismatches": self.mismatches,
            "legacy_transition_computations": self.legacy_transition_computations,
            "legacy_transition_computations_per_byte": (
                self.legacy_transition_computations / self.corpus_bytes
                if self.corpus_bytes
                else None
            ),
        }
        if self.memoized is not None:
            result.update(self.memoized.statistics())
        return result


def _worker_compiler(mode, max_cache_entries, collect_stats, dp_cache_scope):
    """One compiler/stat accumulator per worker; DP sharing follows its scope."""
    key = (mode, max_cache_entries, collect_stats, dp_cache_scope)
    compiler = _WORKER_COMPILERS.get(key)
    if compiler is None:
        compiler = ByteMaskCompiler(
            mode, max_cache_entries, collect_stats, dp_cache_scope
        )
        _WORKER_COMPILERS[key] = compiler
    return compiler


def iter_legal_masks(
    data: bytes,
    slice_layout: str = SLICE_LAYOUT_MACROBLOCK,
    *,
    byte_mask_compiler=None,
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
        if byte_mask_compiler is not None:
            byte_mask_compiler.record_corpus_byte()
        mask = get_valid_byte_mask(
            state, byte_mask_compiler=byte_mask_compiler
        )
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
    path_string: str,
    slice_layout: str = SLICE_LAYOUT_MACROBLOCK,
    compiler_mode: str = "legacy",
    max_cache_entries: int = 1_000_000,
    collect_stats: bool = False,
    dp_cache_scope: str = "root",
) -> tuple[str, int, int, array, list[bytes], int, dict, dict]:
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
    compiler = _worker_compiler(
        compiler_mode, max_cache_entries, collect_stats, dp_cache_scope
    )

    local_table: dict[bytes, int] = {}
    local_masks: list[bytes] = []
    local_mask_ids = array("I", [0]) * len(data)
    pack_seconds = 0.0
    scan_started = time.perf_counter()
    try:
        for i, mask in enumerate(
            iter_legal_masks(
                data,
                slice_layout,
                byte_mask_compiler=compiler,
            )
        ):
            pack_started = time.perf_counter()
            packed = pack_mask(mask)
            pack_seconds += time.perf_counter() - pack_started
            local_id = local_table.get(packed)
            if local_id is None:
                local_id = len(local_masks)
                local_table[packed] = local_id
                local_masks.append(packed)
            local_mask_ids[i] = local_id
    except HA.MaskCacheLimitError as exc:
        raise RuntimeError(
            f"{exc}; worker_statistics="
            f"{json.dumps(compiler.statistics(), sort_keys=True)}"
        ) from exc

    timings = {
        "scan_seconds": time.perf_counter() - scan_started,
        "pack_seconds": pack_seconds,
    }
    return (
        path_string,
        stat.st_size,
        stat.st_mtime_ns,
        local_mask_ids,
        local_masks,
        os.getpid(),
        compiler.statistics(),
        timings,
    )


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
    result: tuple[str, int, int, array, list[bytes], int, dict, dict],
) -> int:
    path, size, mtime_ns, local_mask_ids, local_masks = result[:5]

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


def aggregate_worker_statistics(worker_statistics: dict[int, dict]) -> dict:
    """Aggregate counters while retaining per-process cache/cardinality details.

    Cache entries and field cardinalities are process-local and therefore are not
    presented as a fictitious global union.  The decision metric--actual parser
    transition computations per source byte--is additive and is aggregated.
    """
    workers = list(worker_statistics.values())

    def total(name):
        return sum(worker.get(name, 0) or 0 for worker in workers)

    corpus_bytes = total("corpus_bytes")
    transition_requests = total("transition_requests")
    transition_computations = total("transition_computations")
    legacy_transition_computations = total("legacy_transition_computations")
    root_requests = total("root_requests")
    root_hits = total("root_hits")
    dp_by_k = {}
    for k in range(1, 9):
        entries = [worker.get("dp_by_k", {}).get(str(k), {}) for worker in workers]
        requests = sum(entry.get("requests", 0) for entry in entries)
        hits = sum(entry.get("hits", 0) for entry in entries)
        same_root_hits = sum(entry.get("same_root_hits", 0) for entry in entries)
        cross_root_hits = sum(entry.get("cross_root_hits", 0) for entry in entries)
        misses = sum(entry.get("misses", 0) for entry in entries)
        dp_by_k[str(k)] = {
            "requests": requests,
            "hits": hits,
            "same_root_hits": same_root_hits,
            "cross_root_hits": cross_root_hits,
            "misses": misses,
            "hit_rate": hits / requests if requests else None,
        }

    dp_same_root_hits = total("dp_same_root_hits")
    dp_cross_root_hits = total("dp_cross_root_hits")

    bucket_labels = Counter()
    context_sources = Counter()
    seconds = Counter()
    for worker in workers:
        bucket_labels.update(worker.get("bucket_crossings_by_coeff_token_label", {}))
        context_sources.update(worker.get("context_crossings_by_source", {}))
        seconds.update(worker.get("seconds", {}))

    return {
        "worker_count": len(workers),
        "corpus_bytes": corpus_bytes,
        "root_requests": root_requests,
        "root_hits": root_hits,
        "root_misses": total("root_misses"),
        "root_hit_rate": root_hits / root_requests if root_requests else None,
        "dp_by_k": dp_by_k,
        "dp_same_root_hits": dp_same_root_hits,
        "dp_cross_root_hits": dp_cross_root_hits,
        "dp_cross_root_fraction_of_hits": (
            dp_cross_root_hits / (dp_same_root_hits + dp_cross_root_hits)
            if dp_same_root_hits + dp_cross_root_hits
            else None
        ),
        "transition_requests": transition_requests,
        "transition_computations": transition_computations,
        "transition_cache_removed": True,
        "transition_computations_per_byte": (
            transition_computations / corpus_bytes if corpus_bytes else None
        ),
        "legacy_transition_computations": legacy_transition_computations,
        "legacy_transition_computations_per_byte": (
            legacy_transition_computations / corpus_bytes if corpus_bytes else None
        ),
        "transition_reduction_factor": (
            legacy_transition_computations / transition_computations
            if legacy_transition_computations and transition_computations
            else None
        ),
        "coeff_context_crossings": total("coeff_context_crossings"),
        "committed_grid_bucket_crossings": total(
            "committed_grid_bucket_crossings"
        ),
        "overlay_bucket_crossings": total("overlay_bucket_crossings"),
        "bucket_crossings_by_coeff_token_label": dict(bucket_labels),
        "context_crossings_by_source": dict(context_sources),
        "legacy_comparisons": total("legacy_comparisons"),
        "legacy_mismatches": total("legacy_mismatches"),
        "seconds": dict(seconds),
        "sum_process_local_cache_entries": total("total_cache_entries"),
        "max_process_cache_entries": max(
            (worker.get("total_cache_entries", 0) for worker in workers),
            default=0,
        ),
        "max_process_peak_rss_mb": max(
            (worker.get("peak_rss_mb", 0) for worker in workers), default=0
        ),
        "per_worker": {
            str(pid): statistics for pid, statistics in sorted(worker_statistics.items())
        },
    }


def build_table(
    manifest_path: Path,
    output_path: Path,
    workers: int,
    rebuild: bool,
    slice_layout: str = SLICE_LAYOUT_MACROBLOCK,
    *,
    compiler_mode: str = "legacy",
    max_cache_entries: int = 1_000_000,
    dp_cache_scope: str = "root",
    statistics_path: Path | None = None,
    require_committed_grid_crossing: bool = False,
    max_files: int | None = None,
) -> None:
    _WORKER_COMPILERS.clear()
    rows = load_manifest_rows(manifest_path, report_progress=True)
    paths = list(dict.fromkeys(str(Path(row["h264_path"])) for row in rows))
    if max_files is not None:
        paths = paths[:max_files]
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
        stale_paths = (
            [p for p in existing_files if p not in current_paths]
            if max_files is None
            else []
        )
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
        worker_statistics = {}
        scan_seconds = 0.0
        pack_seconds = 0.0
        write_seconds = 0.0
        print(
            f"Building legal-mask table: {unchanged_files:,} cached, "
            f"{len(pending_paths):,} pending, compiler={compiler_mode} "
            f"-> {output_path}",
            flush=True,
        )

        if workers == 1:
            results = map(
                scan_file,
                pending_paths,
                repeat(slice_layout),
                repeat(compiler_mode),
                repeat(max_cache_entries),
                repeat(statistics_path is not None or compiler_mode == "compare"),
                repeat(dp_cache_scope),
            )
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(
                scan_file,
                pending_paths,
                repeat(slice_layout),
                repeat(compiler_mode),
                repeat(max_cache_entries),
                repeat(statistics_path is not None or compiler_mode == "compare"),
                repeat(dp_cache_scope),
                chunksize=4,
            )

        try:
            for result in results:
                worker_pid, compiler_statistics, timings = result[5:]
                worker_statistics[worker_pid] = compiler_statistics
                scan_seconds += timings["scan_seconds"]
                pack_seconds += timings["pack_seconds"]
                write_started = time.perf_counter()
                processed_bytes += write_file_result(connection, global_cache, result)
                write_seconds += time.perf_counter() - write_started
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
            "mask_compiler": compiler_mode,
            "dp_cache_scope": dp_cache_scope,
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.commit()

        n_files, n_masks = connection.execute(
            "SELECT (SELECT COUNT(*) FROM files), (SELECT COUNT(*) FROM mask_table)"
        ).fetchone()

    elapsed = time.perf_counter() - started_at
    aggregate = aggregate_worker_statistics(worker_statistics)
    aggregate.update(
        {
            "manifest": str(manifest_path.resolve()),
            "slice_layout": slice_layout,
            "compiler_mode": compiler_mode,
            "dp_cache_scope": dp_cache_scope,
            "processed_files": processed_files,
            "processed_bytes": processed_bytes,
            "wall_seconds": elapsed,
            "throughput_bytes_per_second": (
                processed_bytes / elapsed if elapsed else None
            ),
            "summed_worker_scan_seconds": scan_seconds,
            "summed_worker_pack_seconds": pack_seconds,
            "writer_seconds": write_seconds,
        }
    )
    if statistics_path is not None:
        statistics_path.parent.mkdir(parents=True, exist_ok=True)
        statistics_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
        print(f"Compiler statistics -> {statistics_path}", flush=True)

    transitions_per_byte = aggregate.get("transition_computations_per_byte")
    if transitions_per_byte is not None:
        reduction = aggregate.get("transition_reduction_factor")
        reduction_text = f", reduction={reduction:.2f}x" if reduction else ""
        cross_root_fraction = aggregate.get("dp_cross_root_fraction_of_hits")
        cross_root_text = (
            f", cross-byte share of DP hits={cross_root_fraction:.2%}"
            if cross_root_fraction is not None
            else ""
        )
        print(
            "Memoized compiler: "
            f"{transitions_per_byte:.3f} transition computations/source byte, "
            f"cache_scope={dp_cache_scope}, "
            f"committed-grid crossings={aggregate['committed_grid_bucket_crossings']:,}"
            f"{reduction_text}{cross_root_text}",
            flush=True,
        )
    if (
        require_committed_grid_crossing
        and aggregate.get("committed_grid_bucket_crossings", 0) == 0
    ):
        raise RuntimeError(
            "No look-ahead transition crossed into a coeff_token context that read "
            "the committed coefficient grid; the riskiest cache-key path was not tested"
        )

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
    parser.add_argument(
        "--mask-compiler",
        choices=MASK_COMPILERS,
        default="legacy",
        help=(
            "legacy keeps the existing exhaustive tree; memoized uses the DP; "
            "compare runs both and aborts on the first unequal mask"
        ),
    )
    parser.add_argument(
        "--max-cache-entries",
        type=int,
        default=1_000_000,
        help="hard DP-cache limit; entries are never evicted",
    )
    parser.add_argument(
        "--dp-cache-scope",
        choices=DP_CACHE_SCOPES,
        default="root",
        help=(
            "root clears DP state before each corpus byte (bounded default); "
            "worker retains it to measure cross-byte reuse"
        ),
    )
    parser.add_argument(
        "--statistics-json",
        type=Path,
        default=None,
        help="write compiler/cache/timing diagnostics as JSON",
    )
    parser.add_argument(
        "--require-committed-grid-crossing",
        action="store_true",
        help="fail if the pilot never exercises an nC lookup from the committed grid",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="only process the first N unique manifest files (pilot runs)",
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_cache_entries < 1:
        raise ValueError("--max-cache-entries must be at least 1")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be at least 1")
    if args.max_files is not None and args.output is None:
        raise ValueError(
            "--max-files requires an explicit --output so a pilot cannot overwrite "
            "the corpus-wide default table"
        )
    output = args.output or default_mask_table_path(args.manifest)
    build_table(
        args.manifest,
        output,
        args.workers,
        args.rebuild,
        slice_layout=args.slice_layout,
        compiler_mode=args.mask_compiler,
        max_cache_entries=args.max_cache_entries,
        dp_cache_scope=args.dp_cache_scope,
        statistics_path=args.statistics_json,
        require_committed_grid_crossing=args.require_committed_grid_crossing,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
