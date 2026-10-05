#!/usr/bin/env python3
"""How many bits could the macroblock layer take under simple models of its syntax values?

Parses CAVLC H.264 streams with ``litgpt/byte/h264_syntax`` and re-codes every
macroblock-layer syntax element (mb_skip_run, mb_type, intra modes, mvd, cbp,
coeff_token, levels, total_zeros, run_before, ...) with count-based models of
increasing context, fit on one set of videos and scored on a disjoint set:

  M0  the bits CAVLC actually spends (reference; = the bitstream)
  M1  per-element value frequencies with (roughly) CAVLC's own table context:
      nC for coeff_token, TotalCoeff for total_zeros, zeros_left for run_before,
      level index for levels (CAVLC's adaptive suffix length enters at M2)
  M2  + spatial / intra-macroblock context (left/top macroblock class, cbp and
      mvd buckets, block kind, level index and previous level, macroblock class)
  M3  + temporal context: the co-located macroblock / block in the previous
      frame (skip, class, cbp, first mvd, TotalCoeff)

Probabilities back off M3 -> M2 -> M1 -> per-element unigram (additive
smoothing). Values outside a clipped alphabet are coded as an escape symbol
plus their CAVLC bits; sign bits (trailing ones, level signs) are charged 1 bit
in every model. Code lengths are summed per syntax category (as labelled by
h264_syntax) and frame type, and reported as "bits per byte" = 8 x model bits /
CAVLC bits, the same unit as a byte model's per-field CE (nats x 1.4427).

A count model only bounds the achievable code length from above: a gap below
CAVLC shows learnable structure; no gap does not prove a floor.

    python scripts/byte/eval/value_entropy.py --inputs DATA/part1 \\
        --fit-videos 2000 --test-videos 200 --workers 8 --out OUT/value_entropy.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # full environment
    from litgpt.byte import h264_syntax as HS
except ImportError:  # no torch: load the parser without litgpt/__init__
    from syntax_mask.compare_legacy import _load_legacy

    _load_legacy(_ROOT)
    HS = sys.modules["litgpt.byte.h264_syntax"]

MODELS = ("M0", "M1", "M2", "M3")
MB_CATEGORIES = ("mb_header", "mb_pred", "cbp", "mb_qp_delta", "residual_luma", "residual_chroma")
ALPHA = 2.0  # backoff weight at every context level
MVD_CLIP, LEVEL_CLIP, RUN_CLIP = 32, 32, 64


def bucket(v: int, edges=(0, 1, 2, 3, 5, 9, 17)) -> int:
    """Small-integer bucket: index of the largest edge <= v."""
    b = 0
    for i, e in enumerate(edges):
        if v >= e:
            b = i
    return b


def sbucket(v: int) -> int:
    return bucket(abs(v)) * (1 if v >= 0 else -1)


# ---------------------------------------------------------------------------
# per-stream tokenisation
# ---------------------------------------------------------------------------
_BLOCK = re.compile(r"^(luma_dc|luma_ac|luma|chroma_dc|chroma_ac)((?:\[\d+\])*)\.(.+)$")


class Frame:
    """Per-macroblock summary of one decoded picture, for spatial/temporal context."""

    def __init__(self) -> None:
        self.cls: dict[int, str] = {}  # mb_addr -> class ("skip", "I4", "I16", "P16", ...)
        self.cbp: dict[int, int] = {}
        self.mvd: dict[int, tuple[int, int]] = {}  # first mvd of the macroblock
        self.chroma_mode: dict[int, int] = {}
        self.tc: dict[tuple, int] = {}  # (mb_addr, block name) -> TotalCoeff


def mb_class(slice_p: bool, mb_type: int) -> str:
    if slice_p:
        if mb_type <= 4:
            return ("P16", "P16x8", "P8x16", "P8x8", "P8x8r0")[mb_type]
        mb_type -= 5
    if mb_type == 0:
        return "I4"
    if mb_type == 25:
        return "IPCM"
    return "I16"


def tokens_for_stream(data: bytes):
    """Yield (kind, symbol, ctx1, ctx2, ctx3, cavlc_bits, extra_bits, category, ftype)."""
    sp = HS.parse_stream(data, parse_slice_data=True)
    prev: Frame | None = None
    cur: Frame | None = None
    width = None
    for nal in sp.nals:
        if nal.nal.nal_type not in (1, 5):
            continue
        spans = nal.spans
        hdr = {s.name: s.value for s in spans if s.category.value == "slice_header"}
        if nal.status.value != "ok" or "slice_type" not in hdr:
            prev = cur = None  # desync: drop temporal context
            continue
        if hdr.get("first_mb_in_slice", 0) == 0:
            prev, cur = cur, Frame()
        if width is None:
            sps = next(iter(sp.sps.values()))
            width = sps.pic_width_in_mbs
        slice_p = hdr["slice_type"] % 5 == 0
        ftype = "p" if slice_p else ("idr" if nal.nal.nal_type == 5 else "i")
        yield from _tokens_for_slice(spans, slice_p, ftype, cur, prev, width)


def _tokens_for_slice(spans, slice_p, ftype, cur: Frame, prev: Frame | None, width: int):
    last_run = -1
    mb_state: dict[int, dict] = defaultdict(dict)

    def left_top(addr, table, default):
        left = table.get(addr - 1, default) if addr % width else "edge"
        top = table.get(addr - width, default) if addr >= width else "edge"
        return left, top

    for s in spans:
        cat = s.category.value
        if cat not in MB_CATEGORIES:
            continue
        bits = s.bit_end - s.bit_start
        addr = s.mb_addr
        name = s.name
        v = s.value
        co = prev  # temporal context source
        if name == "mb_skip_run":
            if co is not None:
                k = 0
                while k < 64 and co.cls.get(addr + k) == "skip":
                    k += 1
                tctx = bucket(k)
            else:
                tctx = "na"
            for a in range(addr, addr + v):
                cur.cls[a] = "skip"
                cur.cbp[a] = 0
                cur.mvd[a] = (0, 0)
            sym = v if v <= RUN_CLIP else "ESC"
            c1 = ()
            c2 = (bucket(last_run) if last_run >= 0 else "first",)
            c3 = c2 + (tctx,)
            yield ("mb_skip_run", sym, c1, c2, c3, bits, bits if sym == "ESC" else 0, cat, ftype)
            last_run = v
            continue
        if name == "mb_type":
            cls = mb_class(slice_p, v)
            cur.cls[addr] = cls
            mb_state[addr]["cls"] = cls
            lt = left_top(addr, cur.cls, "none")
            c1 = (slice_p,)
            c2 = c1 + lt
            c3 = c2 + ((co.cls.get(addr, "none") if co else "na"),)
            yield ("mb_type", v, c1, c2, c3, bits, 0, cat, ftype)
            continue
        cls = mb_state[addr].get("cls", "none")
        if name.startswith("sub_mb_type"):
            yield ("sub_mb_type", v, (), (), (), bits, 0, cat, ftype)
            continue
        if name.startswith("prev_intra4x4_pred_mode_flag"):
            prevflag = mb_state[addr].get("pflag", "first")
            mb_state[addr]["pflag"] = v
            yield ("prev_intra_flag", v, (), (prevflag,), (prevflag,), bits, 0, cat, ftype)
            continue
        if name.startswith("rem_intra4x4_pred_mode"):
            yield ("rem_intra_mode", v, (), (), (), bits, 0, cat, ftype)
            continue
        if name == "intra_chroma_pred_mode":
            cur.chroma_mode[addr] = v
            lt = left_top(addr, cur.chroma_mode, "none")
            c3 = lt + ((co.chroma_mode.get(addr, "none") if co else "na"),)
            yield ("intra_chroma_mode", v, (), lt, c3, bits, 0, cat, ftype)
            continue
        if name.startswith("ref_idx"):
            yield ("ref_idx", v, (), (), (), bits, 0, cat, ftype)
            continue
        if name.startswith("mvd_l0"):
            comp = name[-1]  # "x" or "y"
            part = int(name[name.index("[") + 1 : name.index("]")])
            st = mb_state[addr]
            if part == 0:
                st.setdefault("mvd0", [0, 0])["xy".index(comp)] = v
                cur.mvd[addr] = tuple(st["mvd0"])
            sym = v if abs(v) <= MVD_CLIP else "ESC"
            i = "xy".index(comp)
            left = cur.mvd.get(addr - 1) if addr % width else None
            prev_part = st.get(f"last_{comp}")
            st[f"last_{comp}"] = v
            c1 = (comp,)
            c2 = c1 + (
                part > 0,
                sbucket(prev_part) if prev_part is not None else "none",
                sbucket(left[i]) if left else "none",
                cls,
            )
            com = co.mvd.get(addr) if co else None
            c3 = c2 + ((sbucket(com[i]) if com else "none"),)
            yield ("mvd", sym, c1, c2, c3, bits, bits if sym == "ESC" else 0, cat, ftype)
            continue
        if name == "coded_block_pattern":
            cur.cbp[addr] = v
            intra = cls.startswith("I")
            lt = left_top(addr, cur.cbp, -1)
            c1 = (intra,)
            c2 = c1 + lt
            c3 = c2 + ((co.cbp.get(addr, -1) if co else "na"),)
            yield ("cbp", v, c1, c2, c3, bits, 0, cat, ftype)
            continue
        if name == "mb_qp_delta":
            sym = v if abs(v) <= 26 else "ESC"
            yield ("mb_qp_delta", sym, (), (), (), bits, 0, cat, ftype)
            continue
        m = _BLOCK.match(name)
        if m is None:  # pcm samples and anything else: charged as-is
            yield ("raw", 0, (), (), (), bits, bits, cat, ftype)
            continue
        kind, idx, elem = m.group(1), m.group(2), m.group(3)
        blk = kind + idx
        intra = cls.startswith("I")
        st = mb_state[addr]
        co_tc = co.tc.get((addr, blk)) if co else None
        tctx = bucket(co_tc) if co_tc is not None else ("na" if co is None else "none")
        if elem == "coeff_token":
            if isinstance(v, dict) and v.get("failed"):
                continue
            tc, t1, nc = v["total_coeff"], v["trailing_ones"], v["nC"]
            cur.tc[(addr, blk)] = tc
            st["blk"] = {"tc": tc, "t1": t1, "prevlev": None}
            c1 = (kind, bucket(nc, (0, 2, 4, 8)) if nc >= 0 else nc)
            c2 = c1 + (intra, idx)
            c3 = c2 + (tctx,)
            yield ("coeff_token", (tc, t1), c1, c2, c3, bits, 0, cat, ftype)
        elif elem == "trailing_ones_sign_flag":
            yield ("signs", 0, (), (), (), bits, bits, cat, ftype)
        elif elem.startswith("level["):
            b = st.get("blk", {"tc": 0, "t1": 0, "prevlev": None})
            i = int(elem[6:-1])
            mag = abs(v)
            sym = mag if mag <= LEVEL_CLIP else "ESC"
            pl = b["prevlev"]
            c1 = (kind, min(i, 3), i == 0 and b["t1"] < 3, b["tc"] > 10 and b["t1"] < 3)
            c2 = c1 + (bucket(pl) if pl is not None else "first", bucket(b["tc"]), intra)
            c3 = c2 + (tctx,)
            b["prevlev"] = mag
            # sign: 1 bit in every model; escape: CAVLC bits
            yield ("level", sym, c1, c2, c3, bits, 1 + (bits if sym == "ESC" else 0), cat, ftype)
        elif elem == "total_zeros":
            b = st.get("blk", {"tc": 0})
            b["zl"] = v
            c1 = (kind in ("chroma_dc",), b["tc"])
            c2 = c1 + (intra, kind)
            c3 = c2 + (tctx,)
            yield ("total_zeros", v, c1, c2, c3, bits, 0, cat, ftype)
        elif elem.startswith("run_before["):
            b = st.get("blk", {"tc": 0})
            zl = b.get("zl", 0)
            i = int(elem[11:-1])
            c1 = (min(zl, 7),)  # CAVLC's own context: zeros_left
            c2 = c1 + (min(i, 4), bucket(b["tc"]), kind)
            c3 = c2 + (tctx,)
            b["zl"] = zl - v
            yield ("run_before", v, c1, c2, c3, bits, 0, cat, ftype)
        else:
            yield ("raw", 0, (), (), (), bits, bits, cat, ftype)


# ---------------------------------------------------------------------------
# counting and scoring
# ---------------------------------------------------------------------------
def fit_file(path: str):
    counts = [defaultdict(Counter) for _ in range(4)]  # level 0..3: (kind, ctx) -> Counter(sym)
    try:
        data = Path(path).read_bytes()
        for kind, sym, c1, c2, c3, _bits, _extra, _cat, _ft in tokens_for_stream(data):
            if kind in ("raw", "signs"):
                continue
            counts[0][(kind,)][sym] += 1
            counts[1][(kind, c1)][sym] += 1
            counts[2][(kind, c2)][sym] += 1
            counts[3][(kind, c3)][sym] += 1
    except Exception as exc:  # noqa: BLE001 - skip unparsable files
        return None, f"{path}: {exc!r}"[:300]
    return [{k: dict(v) for k, v in c.items()} for c in counts], None


_TABLES = None


def _init_score(tables):
    global _TABLES
    _TABLES = tables


def _prob(kind, sym, ctxs):
    counts, totals, alphabet = _TABLES
    v = alphabet.get(kind, 1) + 1
    c0 = counts[0].get((kind,), {})
    p = (c0.get(sym, 0) + 1) / (totals[0].get((kind,), 0) + v)
    out = []
    for level, ctx in enumerate(ctxs, start=1):
        key = (kind, ctx)
        c = counts[level].get(key)
        if c is not None:
            p = (c.get(sym, 0) + ALPHA * p) / (totals[level][key] + ALPHA)
        out.append(p)
    return out  # p under M1, M2, M3


def score_file(path: str):
    sums = defaultdict(float)  # (model, category, ftype) -> bits
    counts, _totals, alphabet = _TABLES
    try:
        data = Path(path).read_bytes()
        for kind, sym, c1, c2, c3, bits, extra, cat, ft in tokens_for_stream(data):
            sums[("M0", cat, ft)] += bits
            if kind in ("raw", "signs"):
                for m in MODELS[1:]:
                    sums[(m, cat, ft)] += bits
                continue
            if sym not in counts[0].get((kind,), {}) and sym != "ESC":
                sym, extra = "ESC", extra + bits  # unseen value: escape + raw bits
            p1, p2, p3 = _prob(kind, sym, (c1, c2, c3))
            for m, p in zip(MODELS[1:], (p1, p2, p3)):
                sums[(m, cat, ft)] += -math.log2(p) + extra
    except Exception as exc:  # noqa: BLE001
        return None, f"{path}: {exc!r}"[:300]
    return dict(sums), None


def merge_counts(parts):
    counts = [defaultdict(Counter) for _ in range(4)]
    for part in parts:
        for level in range(4):
            for key, c in part[level].items():
                counts[level][key].update(c)
    totals = [{k: sum(c.values()) for k, c in lvl.items()} for lvl in counts]
    alphabet = {key[0]: len(c) for key, c in counts[0].items()}
    for key in counts[0]:
        counts[0][key]["ESC"] += 0  # make ESC a known symbol of every kind
    plain = [{k: dict(c) for k, c in lvl.items()} for lvl in counts]
    return plain, totals, alphabet


def video_id(path: str) -> str:
    """Clips of one source video share a prefix; keep them on one side of the split."""
    return re.sub(r"_\d+$", "", Path(path).stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="dirs or .h264 files")
    ap.add_argument("--fit-videos", type=int, default=2000)
    ap.add_argument("--test-videos", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transformer-fields", type=Path, nargs="*", default=[],
                    help="megabyte_global_ablation *-fields.json files to compare against")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        files += [str(x) for x in (sorted(p.rglob("*.h264")) if p.is_dir() else [p])]
    groups = defaultdict(list)
    for f in files:
        groups[video_id(f)].append(f)
    vids = sorted(groups)
    random.Random(args.seed).shuffle(vids)
    test, fit = [], []
    for v in vids:  # fill the test set first, by clip count
        (test if len(test) < args.test_videos else fit).extend(groups[v])
    fit = fit[: args.fit_videos]
    print(f"{len(files)} files from {len(vids)} source videos; fit {len(fit)} clips, test {len(test)} clips",
          flush=True)

    started = time.perf_counter()
    with Pool(args.workers) as pool:
        parts, errors = [], []
        for i, (res, err) in enumerate(pool.imap_unordered(fit_file, fit, chunksize=4)):
            (parts.append(res) if res is not None else errors.append(err))
            if (i + 1) % 200 == 0:
                print(f"  fit {i + 1}/{len(fit)}  {time.perf_counter() - started:.0f}s", flush=True)
    tables = merge_counts(parts)
    print(f"fit done: {len(parts)} ok, {len(errors)} failed, {time.perf_counter() - started:.0f}s", flush=True)

    totals = defaultdict(float)
    with Pool(args.workers, initializer=_init_score, initargs=(tables,)) as pool:
        for i, (res, err) in enumerate(pool.imap_unordered(score_file, test, chunksize=2)):
            if res is None:
                errors.append(err)
                continue
            for k, v in res.items():
                totals[k] += v
    print(f"score done: {time.perf_counter() - started:.0f}s", flush=True)

    def agg(model, cats, ftypes):
        return sum(v for (m, c, f), v in totals.items() if m == model and c in cats and f in ftypes)

    report = {"fit_clips": len(parts), "test_clips": len(test), "errors": errors[:20], "rows": []}
    groups_out = [(c, (c,)) for c in MB_CATEGORIES] + [("mb_layer_total", MB_CATEGORIES)]
    for ftname, fts in (("all", ("idr", "i", "p")), ("idr", ("idr", "i")), ("p", ("p",))):
        for label, cats in groups_out:
            base = agg("M0", cats, fts)
            if base <= 0:
                continue
            row = {"category": label, "frame_type": ftname, "cavlc_bits": base}
            for m in MODELS:
                row[f"{m}_bits_per_byte"] = 8 * agg(m, cats, fts) / base
            report["rows"].append(row)

    # Transformer per-field CE (nats/byte -> bits/byte) for the same categories.
    for f in args.transformer_fields:
        d = json.loads(f.read_text(encoding="utf-8"))
        ce = d["ce_by_field"]["baseline"]
        nb = d.get("bytes_by_field", {})
        layer = [k for k in (*MB_CATEGORIES, "mixed") if k in ce]
        total_bytes = sum(nb.get(k, 0) for k in layer)
        report.setdefault("transformer", {})[str(f)] = {
            **{k: ce[k] / math.log(2) for k in layer},
            "mb_layer_total_incl_mixed": (
                sum(ce[k] * nb.get(k, 0) for k in layer) / total_bytes / math.log(2) if total_bytes else None
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n{'category':18s}{'type':5s}{'CAVLC MB':>10s}" + "".join(f"{m:>8s}" for m in MODELS) + "   (bits per CAVLC byte)")
    for r in report["rows"]:
        print(f"{r['category']:18s}{r['frame_type']:5s}{r['cavlc_bits'] / 8e6:9.2f}M"
              + "".join(f"{r[f'{m}_bits_per_byte']:8.2f}" for m in MODELS))
    for f, t in report.get("transformer", {}).items():
        print(f"transformer {Path(f).parent.parent.name[:40]}: "
              + ", ".join(f"{k} {v:.2f}" for k, v in t.items() if v is not None))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
