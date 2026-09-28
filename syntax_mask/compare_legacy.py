"""Differential comparison with the legacy bit-by-bit mask (litgpt/byte/h264_mask).

Run from the litgpt repo root:

    python -m syntax_mask.compare_legacy FILE.h264 [--layout frame|mb] [--max-bytes N]

Per syntax field of the new oracle it reports positions where the new mask is
*stricter* (legacy allows bytes the new grammar rejects -- expected, the legacy
automaton leaves SPS/PPS/SEI and several header fields permissive) and where it
is *looser* (legacy rejects bytes the new grammar accepts -- each one is either
a legacy over-restriction or a missing constraint here, and deserves a look).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from collections import Counter
from pathlib import Path

from .grammar import Profile
from .stream import MaskStream


def _load_legacy(repo_root: Path):
    """Load litgpt/byte/{h264_cavlc_tables,h264_syntax,h264_automaton,h264_mask}
    without importing litgpt/__init__ (and its training dependencies)."""
    byte_dir = repo_root / "litgpt" / "byte"
    sys.modules.setdefault("litgpt", types.ModuleType("litgpt"))
    sys.modules.setdefault("litgpt.byte", types.ModuleType("litgpt.byte"))
    mod = None
    for short in ("h264_cavlc_tables", "h264_syntax", "h264_automaton", "h264_mask"):
        name = f"litgpt.byte.{short}"
        if name in sys.modules and getattr(sys.modules[name], "__file__", None):
            mod = sys.modules[name]
            continue
        spec = importlib.util.spec_from_file_location(name, byte_dir / f"{short}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        setattr(sys.modules["litgpt.byte"], short, mod)
        spec.loader.exec_module(mod)
    return mod


def compare(data: bytes, layout: str, legacy) -> dict:
    ms = MaskStream(Profile(slice_layout=layout))
    st = legacy.MaskState(
        slice_max_mbs=None if layout == "frame" else 1,
        nal_header_policy=legacy.NAL_HEADER_POLICY_SYNTAX_ONLY,
    )
    stricter, looser, positions = Counter(), Counter(), Counter()
    identical = 0
    removed = 0  # legal bytes legacy allows that the new mask removes
    looser_examples = []
    t_new = t_old = 0.0
    for i, byte in enumerate(data):
        where = ms.where()
        t = time.perf_counter()
        m = ms.mask()
        t_new += time.perf_counter() - t
        t = time.perf_counter()
        old = legacy.get_valid_byte_mask(st)
        t_old += time.perf_counter() - t
        o = sum(1 << b for b in range(256) if old[b])
        positions[where] += 1
        identical += m == o
        removed += bin(o & ~m).count("1")
        if o & ~m:
            stricter[where] += 1
        if m & ~o:
            looser[where] += 1
            if len(looser_examples) < 10:
                looser_examples.append((i, where, [b for b in range(256) if (m & ~o) >> b & 1][:8]))
        ms.advance(byte)
        legacy.advance(st, byte)
    return {
        "bytes": len(data),
        "new_us_per_byte": t_new / len(data) * 1e6,
        "legacy_us_per_byte": t_old / len(data) * 1e6,
        "positions": positions,
        "identical": identical,
        "removed": removed,
        "stricter": stricter,
        "looser": looser,
        "looser_examples": looser_examples,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--layout", choices=("frame", "mb"), default="frame")
    ap.add_argument("--max-bytes", type=int, default=0)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)
    legacy = _load_legacy(Path(args.repo_root))
    for f in args.files:
        data = Path(f).read_bytes()
        if args.max_bytes:
            data = data[: args.max_bytes]
        r = compare(data, args.layout, legacy)
        print(f"== {f}: {r['bytes']} bytes  new {r['new_us_per_byte']:.0f} us/B  "
              f"legacy {r['legacy_us_per_byte']:.0f} us/B")
        n = r["bytes"]
        n_str = sum(r["stricter"].values())
        n_loose = sum(r["looser"].values())
        print(f"   identical masks {r['identical']}/{n} ({100 * r['identical'] / n:.1f}%), "
              f"new stricter at {n_str}, new looser at {n_loose}, "
              f"legacy-allowed bytes removed: {r['removed']} "
              f"({r['removed'] / n:.2f} per position)")
        print(f"   {'field':42s} {'positions':>9s} {'stricter':>9s} {'looser':>7s}")
        for where, n in r["positions"].most_common():
            s, l = r["stricter"][where], r["looser"][where]
            if s or l:
                print(f"   {where:42s} {n:9d} {s:9d} {l:7d}")
        for ex in r["looser_examples"]:
            print("   looser example (offset, field, bytes only new allows):", ex)


if __name__ == "__main__":
    main()
