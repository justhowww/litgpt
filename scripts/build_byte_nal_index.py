"""Build a resumable SQLite NAL index for an H.264 byte corpus."""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from litgpt.data.byte_data import (
    NAL_INDEX_VERSION,
    default_nal_index_path,
    load_manifest_rows,
    parse_annexb_nals,
)


def scan_file(path_string: str) -> tuple[str, int, int, list[tuple[int, int, int, int]]]:
    path = Path(path_string)
    data = path.read_bytes()
    stat = path.stat()
    nals = [
        (nal.start, nal.end, nal.start_code_len, nal.nal_type)
        for nal in parse_annexb_nals(data)
    ]
    return path_string, stat.st_size, stat.st_mtime_ns, nals


def initialize_database(
    connection: sqlite3.Connection, manifest_path: Path, rebuild: bool
) -> None:
    if rebuild:
        connection.executescript(
            "DROP TABLE IF EXISTS nals; DROP TABLE IF EXISTS files; DROP TABLE IF EXISTS metadata;"
        )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS nals (
            file_id INTEGER NOT NULL,
            nal_index INTEGER NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            start_code_len INTEGER NOT NULL,
            nal_type INTEGER NOT NULL,
            PRIMARY KEY (file_id, nal_index),
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        """
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.commit()


def write_file_index(
    connection: sqlite3.Connection,
    result: tuple[str, int, int, list[tuple[int, int, int, int]]],
) -> int:
    path, size, mtime_ns, nals = result
    previous = connection.execute(
        "SELECT id, size, mtime_ns FROM files WHERE path = ?", (path,)
    ).fetchone()
    if previous is not None and previous[1:] == (size, mtime_ns):
        return size

    if previous is None:
        cursor = connection.execute(
            "INSERT INTO files(path, size, mtime_ns) VALUES (?, ?, ?)",
            (path, size, mtime_ns),
        )
        file_id = cursor.lastrowid
    else:
        file_id = previous[0]
        connection.execute(
            "UPDATE files SET size = ?, mtime_ns = ? WHERE id = ?",
            (size, mtime_ns, file_id),
        )
        connection.execute("DELETE FROM nals WHERE file_id = ?", (file_id,))

    connection.executemany(
        """
        INSERT INTO nals(file_id, nal_index, start, end, start_code_len, nal_type)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (file_id, nal_index, start, end, start_code_len, nal_type)
            for nal_index, (start, end, start_code_len, nal_type) in enumerate(nals)
        ],
    )
    return size


def build_index(
    manifest_path: Path, index_path: Path, workers: int, rebuild: bool
) -> None:
    rows = load_manifest_rows(manifest_path, report_progress=True)
    paths = list(dict.fromkeys(str(Path(row["h264_path"])) for row in rows))
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(index_path) as connection:
        initialize_database(connection, manifest_path, rebuild)
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

        started_at = time.perf_counter()
        processed_bytes = 0
        processed_files = 0
        print(
            f"Building NAL index: {unchanged_files:,} cached, "
            f"{len(pending_paths):,} pending -> {index_path}",
            flush=True,
        )

        if workers == 1:
            results = map(scan_file, pending_paths)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            results = executor.map(scan_file, pending_paths, chunksize=8)

        try:
            for result in results:
                processed_bytes += write_file_index(connection, result)
                processed_files += 1
                if processed_files % 100 == 0:
                    connection.commit()
                    elapsed = time.perf_counter() - started_at
                    print(
                        f"Indexed {processed_files:,}/{len(pending_paths):,} new files, "
                        f"{processed_bytes / 1e9:.2f} GB, "
                        f"{processed_bytes / max(elapsed, 1e-9) / 1e6:.1f} MB/s",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown()
        connection.commit()

        manifest_stat = manifest_path.stat()
        metadata = {
            "version": str(NAL_INDEX_VERSION),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_size": str(manifest_stat.st_size),
            "manifest_mtime_ns": str(manifest_stat.st_mtime_ns),
        }
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.commit()

    print(f"NAL index complete: {len(paths):,} files -> {index_path}", flush=True)


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
    output = args.output or default_nal_index_path(args.manifest)
    build_index(args.manifest, output, args.workers, args.rebuild)


if __name__ == "__main__":
    main()
