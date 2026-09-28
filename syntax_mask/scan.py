"""Scan ground-truth H.264 files with the syntax mask and report correctness/speed.

    python -m syntax_mask.scan DATA_DIR [more files/dirs] --workers 32 \
        --layout frame --reference-rate 0.01 --probe-rate 0.002 --json out.json

Per file it checks:

* **GT acceptance** -- the real byte is legal at every position (a failure is a
  grammar bug or an out-of-scope stream; the scan of that file stops there);
* **reference parity** (sampled) -- the field-level Step 2 mask equals the
  bit-by-bit reference mask;
* **no dead ends** (sampled probes) -- from a GT prefix, follow random legal
  bytes (preferring bytes different from GT) for ``--probe-len`` steps; every
  mask along the way must be non-empty.

``--dump-dir`` writes ``<name>.masks``: 32 bytes per input byte, little-endian
bit order (bit b of the 256-bit int = byte value b), i.e. the training table.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .grammar import Illegal, Profile
from .stream import MaskStream, mask_to_bytes


def _popcount(m: int) -> int:
    return bin(m).count("1")


def _field_group(where: str) -> str:
    return where.split(":")[-1]


def _probe(ms: MaskStream, rng: random.Random, length: int, avoid: int):
    """Random legal walk from a copy of ``ms``; returns an error string or None."""
    probe = copy.deepcopy(ms)
    for step in range(length):
        m = probe.mask()
        if not m:
            return f"empty mask after {step} probe bytes at {probe.where()}"
        choices = [b for b in range(256) if (m >> b) & 1]
        if step == 0 and len(choices) > 1:
            choices = [b for b in choices if b != avoid] or choices
        b = rng.choice(choices)
        try:
            probe.advance(b)
        except Illegal as exc:
            return f"legal byte 0x{b:02x} rejected on advance: {exc}"
    return None


def scan_file(path: str, opts: dict) -> dict:
    data = Path(path).read_bytes()
    profile = Profile(slice_layout=opts["layout"])
    rng = random.Random(hash((path, opts["seed"])) & 0xFFFFFFFF)
    ms = MaskStream(profile)
    fields = defaultdict(lambda: [0, 0])  # where -> [count, sum popcount]
    ref_checked = ref_mismatch = probes = probe_fail = 0
    errors = []
    dump = bytearray() if opts["dump_dir"] else None
    t0 = time.perf_counter()
    mask_seconds = 0.0
    failure = None
    n = 0
    for i, byte in enumerate(data):
        where = ms.where()
        t = time.perf_counter()
        m = ms.mask()
        mask_seconds += time.perf_counter() - t
        f = fields[_field_group(where)]
        f[0] += 1
        f[1] += _popcount(m)
        if dump is not None:
            dump += mask_to_bytes(m)
        if not (m >> byte) & 1:
            failure = {"offset": i, "byte": byte, "where": where,
                       "error": "GT byte masked out", "nal": ms.nal_count}
            break
        if opts["reference_rate"] and rng.random() < opts["reference_rate"]:
            ref_checked += 1
            r = ms.mask_reference()
            if r != m:
                ref_mismatch += 1
                if len(errors) < 5:
                    errors.append({"offset": i, "where": where, "kind": "reference",
                                   "only_fast": _popcount(m & ~r),
                                   "only_ref": _popcount(r & ~m)})
        if opts["probe_rate"] and rng.random() < opts["probe_rate"]:
            probes += 1
            err = _probe(ms, rng, opts["probe_len"], byte)
            if err:
                probe_fail += 1
                if len(errors) < 5:
                    errors.append({"offset": i, "where": where, "kind": "probe",
                                   "error": err})
        try:
            ms.advance(byte)
        except Illegal as exc:
            failure = {"offset": i, "byte": byte, "where": where,
                       "error": f"advance: {exc}", "nal": ms.nal_count}
            break
        n += 1
    seconds = time.perf_counter() - t0
    if dump is not None and failure is None:
        out = Path(opts["dump_dir"]) / (Path(path).name + ".masks")
        out.write_bytes(bytes(dump))
    return {
        "path": path,
        "bytes": len(data),
        "bytes_scanned": n,
        "ok": failure is None,
        "failure": failure,
        "seconds": seconds,
        "mask_seconds": mask_seconds,
        "nals": ms.nal_count,
        "fields": {k: v for k, v in fields.items()},
        "reference_checked": ref_checked,
        "reference_mismatch": ref_mismatch,
        "probes": probes,
        "probe_failures": probe_fail,
        "errors": errors,
    }


def _collect(inputs, list_file, limit, seed):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(str(x) for x in p.rglob("*") if x.suffix in (".h264", ".264"))
        else:
            paths.append(str(p))
    if list_file:
        paths += [ln.strip() for ln in Path(list_file).read_text().splitlines() if ln.strip()]
    if limit and len(paths) > limit:
        paths = sorted(random.Random(seed).sample(paths, limit))
    return paths


def summarize(results: list[dict]) -> dict:
    total_bytes = sum(r["bytes_scanned"] for r in results)
    fields = defaultdict(lambda: [0, 0])
    for r in results:
        for k, (c, s) in r["fields"].items():
            fields[k][0] += c
            fields[k][1] += s
    mask_seconds = sum(r["mask_seconds"] for r in results)
    failures = [r for r in results if not r["ok"]]
    fail_where = Counter(r["failure"]["where"] for r in failures)
    return {
        "files": len(results),
        "files_ok": len(results) - len(failures),
        "bytes": total_bytes,
        # undeduplicated table: one 256-bit mask (32 bytes) per stream byte
        "mask_storage_gb": total_bytes * 32 / 1e9,
        "cpu_seconds": sum(r["seconds"] for r in results),
        "mask_us_per_byte": mask_seconds / total_bytes * 1e6 if total_bytes else None,
        "mean_legal_bytes": (
            sum(s for _, s in fields.values()) / total_bytes if total_bytes else None
        ),
        "reference_checked": sum(r["reference_checked"] for r in results),
        "reference_mismatch": sum(r["reference_mismatch"] for r in results),
        "probes": sum(r["probes"] for r in results),
        "probe_failures": sum(r["probe_failures"] for r in results),
        "failures_by_field": dict(fail_where.most_common()),
        "first_failures": [
            {"path": r["path"], **r["failure"]} for r in failures[:20]
        ],
        "errors": [dict(e, path=r["path"]) for r in results for e in r["errors"]][:20],
        "fields": {
            k: {"positions": c, "mean_legal": s / c}
            for k, (c, s) in sorted(fields.items(), key=lambda kv: -kv[1][0])
        },
    }


def _manifest_paths(manifest: str, max_rows: int = 0) -> list[str]:
    """``status == ok`` rows of a corpus manifest.jsonl (same resolution rule as
    litgpt/byte/data.py: relative paths live next to the manifest, under h264/)."""
    root = Path(manifest).parent
    out = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok" or not row.get("h264_path"):
                continue
            path = Path(row["h264_path"])
            if not path.is_absolute():
                if path.parts[:1] != ("h264",) and "h264" in path.parts:
                    k = len(path.parts) - 1 - path.parts[::-1].index("h264")
                    path = Path(*path.parts[k:])
                elif path.parts[:1] != ("h264",) and not (root / path).exists():
                    path = Path("h264") / path
                path = root / path
            out.append(str(path))
            if max_rows and len(out) >= max_rows:  # = load_manifest_rows(max_rows)
                break
    return out


def _safe_scan(path: str, opts: dict) -> dict:
    """Never let one unreadable/odd file kill a shard: report it as a failure."""
    try:
        return scan_file(path, opts)
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "bytes": 0, "bytes_scanned": 0, "ok": False,
                "failure": {"offset": -1, "byte": -1, "where": "exception",
                            "error": f"{type(exc).__name__}: {exc}", "nal": -1},
                "seconds": 0.0, "mask_seconds": 0.0, "nals": 0, "fields": {},
                "reference_checked": 0, "reference_mismatch": 0, "probes": 0,
                "probe_failures": 0, "errors": []}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="files or directories (*.h264)")
    ap.add_argument("--list", help="text file with one path per line")
    ap.add_argument("--manifest", help="corpus manifest.jsonl (status==ok rows)")
    ap.add_argument("--max-manifest-rows", type=int, default=0,
                    help="first N status==ok rows in manifest order "
                         "(same subset as training data.max_rows)")
    ap.add_argument("--layout", choices=("frame", "mb"), default="frame",
                    help="frame: one slice per picture (JPEG-LM/default); mb: one MB per slice (AVC-LM)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="random subset of N files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0,
                    help="scan every num_shards-th file starting here (Slurm array index)")
    ap.add_argument("--reference-rate", type=float, default=0.0)
    ap.add_argument("--probe-rate", type=float, default=0.0)
    ap.add_argument("--probe-len", type=int, default=24)
    ap.add_argument("--dump-dir", help="write per-file .masks tables here")
    ap.add_argument("--json", help="write summary + per-file results here")
    ap.add_argument("--out-jsonl", help="append one line per scanned file (resumable)")
    ap.add_argument("--resume", action="store_true",
                    help="skip files already present in --out-jsonl")
    args = ap.parse_args(argv)

    paths = _collect(args.inputs, args.list, 0, args.seed)
    if args.manifest:
        paths += _manifest_paths(args.manifest, args.max_manifest_rows)
    paths = sorted(set(paths))
    if args.num_shards > 1:
        paths = paths[args.shard_index :: args.num_shards]
    if args.limit and len(paths) > args.limit:
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    if not paths:
        ap.error("no input files")
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
        names = Counter(Path(p).name for p in paths)
        dup = [n for n, c in names.items() if c > 1]
        if dup:
            ap.error(f"--dump-dir needs unique file names; duplicates: {dup[:5]}")
    opts = {k: getattr(args, k) for k in
            ("layout", "seed", "reference_rate", "probe_rate", "probe_len", "dump_dir")}

    results = []
    sink = None
    if args.out_jsonl:
        out = Path(args.out_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and out.exists():
            done = {}
            for line in out.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # torn final line from a killed job
                    continue
                done[r["path"]] = r
            results = [done[p] for p in paths if p in done]
            paths = [p for p in paths if p not in done]
            print(f"resume: {len(results)} files already scanned, {len(paths)} left",
                  file=sys.stderr, flush=True)
        sink = out.open("a" if args.resume else "w", encoding="utf-8")

    def record(r):
        results.append(r)
        if sink is not None:
            sink.write(json.dumps(r) + "\n")
            sink.flush()

    started = time.perf_counter()
    total = len(results) + len(paths)
    if args.workers <= 1:
        for p in paths:
            record(_safe_scan(p, opts))
            _progress(results, len(results), total, started)
    elif paths:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(_safe_scan, p, opts) for p in paths]
            for fut in as_completed(futs):
                record(fut.result())
                _progress(results, len(results), total, started)
    if sink is not None:
        sink.close()
    if args.dump_dir:
        _write_dump_index(Path(args.dump_dir), results)
    return _report(results, args.layout, time.perf_counter() - started, args.json)


def _write_dump_index(dump_dir: Path, results) -> None:
    """index.jsonl: h264 path -> .masks file (only files that passed)."""
    with (dump_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda r: r["path"]):
            if r["ok"]:
                f.write(json.dumps({
                    "h264_path": r["path"],
                    "masks": Path(r["path"]).name + ".masks",
                    "bytes": r["bytes"],
                    "mask_bytes": 32 * r["bytes"],
                }) + "\n")


def _report(results, layout, wall_seconds, json_path) -> int:
    summary = summarize(results)
    summary["wall_seconds"] = wall_seconds
    summary["layout"] = layout
    print(json.dumps({k: v for k, v in summary.items() if k != "fields"}, indent=2))
    print("per-field (positions, mean legal bytes):")
    for k, v in list(summary["fields"].items())[:25]:
        print(f"  {k:45s} {v['positions']:10d} {v['mean_legal']:8.2f}")
    if json_path:
        Path(json_path).write_text(json.dumps({"summary": summary, "files": results}, indent=1))
    bad = (summary["files_ok"] != summary["files"] or summary["reference_mismatch"]
           or summary["probe_failures"])
    return 1 if bad else 0


def _progress(results, done, total, started):
    if done % 200 and done != total:
        return
    ok = sum(r["ok"] for r in results)
    nbytes = sum(r["bytes_scanned"] for r in results)
    el = time.perf_counter() - started
    print(f"[{done}/{total}] ok={ok} bytes={nbytes} wall={el:.0f}s "
          f"throughput={nbytes / max(el, 1e-9) / 1e3:.1f} kB/s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
