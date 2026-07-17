#!/usr/bin/env python3
"""Render a paired GT-vs-generated H.264 byte comparison as self-contained HTML.

This is an offline analyzer for streams persisted by eval_stream_continuation.py:

    python scripts/byte/eval/compare_gt_gen_bytes.py RESULTS_DIR --clip 0

or explicit files:

    python scripts/byte/eval/compare_gt_gen_bytes.py --gt clip_gt.h264 --gen clip_gen.h264 --out cmp.html

The report shows:
  * byte-by-byte GT vs generated values, marked same/different/missing/extra;
  * primary syntax element covering each byte in the GT parse and generated parse;
  * per-NAL syntax span tables so byte divergence can be tied to parser state.

It intentionally parses GT and generated independently. After the first divergent
bit/byte, the two parsers may assign different syntax meanings to the same byte
offset; that disagreement is the signal this report is meant to expose.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BYTE_DIR = _REPO_ROOT / "litgpt" / "byte"


def _load_h264_syntax():
    """Import stdlib-only h264_syntax without importing litgpt/__init__."""
    if "litgpt" not in sys.modules:
        sys.modules["litgpt"] = types.ModuleType("litgpt")
        sys.modules["litgpt.byte"] = types.ModuleType("litgpt.byte")
    if str(_BYTE_DIR) not in sys.path:
        sys.path.insert(0, str(_BYTE_DIR))

    def load(short: str):
        name = f"litgpt.byte.{short}"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, _BYTE_DIR / f"{short}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(sys.modules["litgpt.byte"], short, module)
        spec.loader.exec_module(module)
        return module

    load("h264_cavlc_tables")
    return load("h264_syntax")


HS = _load_h264_syntax()


CAT_COLORS = {
    "start_code": "#444b5a",
    "nal_header": "#6b7280",
    "emulation_prevention": "#9ca3af",
    "sps": "#0e7490",
    "pps": "#0891b2",
    "sei": "#475569",
    "slice_header": "#2563eb",
    "mb_header": "#7c3aed",
    "mb_pred": "#db2777",
    "cbp": "#d97706",
    "mb_qp_delta": "#ca8a04",
    "residual_luma": "#16a34a",
    "residual_chroma": "#65a30d",
    "rbsp_trailing": "#334155",
    "slice_data": "#16a34a",
    "unknown": "#94a3b8",
}

NAL_NAMES = {1: "non-IDR slice", 5: "IDR slice", 6: "SEI", 7: "SPS", 8: "PPS", 9: "AUD"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _cat(span) -> str:
    return span.category.value if hasattr(span.category, "value") else str(span.category)


def _assign_syntax(spans: list, n_bytes: int) -> list[dict[str, Any]]:
    by_byte: list[list] = [[] for _ in range(n_bytes)]
    for span in spans:
        lo = max(0, span.byte_start)
        hi = min(n_bytes, span.byte_end)
        for off in range(lo, hi):
            by_byte[off].append(span)

    out: list[dict[str, Any]] = []
    for off, overlaps in enumerate(by_byte):
        if not overlaps:
            out.append({"name": "?", "category": "unknown", "value": None, "mb_addr": None, "all": []})
            continue
        primary = min(overlaps, key=lambda s: (s.byte_end - s.byte_start, s.byte_start))
        out.append(
            {
                "name": primary.name,
                "category": _cat(primary),
                "value": _jsonable(primary.value),
                "mb_addr": primary.mb_addr,
                "all": [_span_summary(s) for s in overlaps],
            }
        )
    return out


def _hx(b: int | None) -> str:
    return "--" if b is None else f"{b:02x}"


def _val(value: Any) -> str:
    if isinstance(value, dict):
        return html.escape(", ".join(f"{k}={value[k]}" for k in value))
    return html.escape(str(value))


def _syntax_cell(s: dict[str, Any]) -> str:
    color = CAT_COLORS.get(s["category"], CAT_COLORS["unknown"])
    mb = "" if s.get("mb_addr") is None else f" mb={s['mb_addr']}"
    overlaps = s.get("all") or []
    overlap_txt = "\n".join(overlaps[:12])
    tip = html.escape(f"primary: {s['name']} {s['value']}{mb}" + (f"\n\nall overlaps:\n{overlap_txt}" if overlap_txt else ""))
    return (
        f"<span class='syntax' style='background:{color}' title='{tip}'>"
        f"{html.escape(s['name'])}</span>"
    )


def _span_summary(span) -> str:
    mb = "" if span.mb_addr is None else f" mb={span.mb_addr}"
    return (
        f"{span.name} value={_jsonable(span.value)} bytes={span.byte_start}:{span.byte_end} "
        f"rbsp_bits={span.bit_start}:{span.bit_end}{mb}"
    )


def _bits(b: int | None) -> str:
    return "--------" if b is None else f"{b:08b}"


def _first_diff_bit(gt_b: int | None, gen_b: int | None) -> int | None:
    if gt_b is None or gen_b is None or gt_b == gen_b:
        return None
    for bit in range(8):
        mask = 1 << (7 - bit)
        if (gt_b & mask) != (gen_b & mask):
            return bit
    return None


def _bit_cells(gt_b: int | None, gen_b: int | None) -> str:
    gt_bits = _bits(gt_b)
    gen_bits = _bits(gen_b)
    first = _first_diff_bit(gt_b, gen_b)
    cells = []
    for i, (gb, eb) in enumerate(zip(gt_bits, gen_bits, strict=True)):
        cls = "bit samebit" if gb == eb else "bit diffbit"
        if first == i:
            cls += " firstbit"
        cells.append(f"<span class='{cls}' title='bit {i} msb-first: GT={gb} GEN={eb}'>{gb}/{eb}</span>")
    return "".join(cells)


def _overlap_list(title: str, syntax: list[dict[str, Any]], off: int) -> str:
    if off >= len(syntax):
        return f"<h3>{html.escape(title)}</h3><div class='muted'>byte outside stream</div>"
    overlaps = syntax[off].get("all") or []
    if not overlaps:
        return f"<h3>{html.escape(title)}</h3><div class='muted'>no syntax span covers this byte</div>"
    items = "".join(f"<li>{html.escape(str(x))}</li>" for x in overlaps)
    return f"<h3>{html.escape(title)}</h3><ul>{items}</ul>"


def _first_diff_panel(gt: bytes, gen: bytes, gt_syntax: list[dict[str, Any]], gen_syntax: list[dict[str, Any]], first_diff: int | None) -> str:
    if first_diff is None:
        return "<h2>First differing byte</h2><div class='summary'>No byte difference.</div>"
    gt_b = gt[first_diff] if first_diff < len(gt) else None
    gen_b = gen[first_diff] if first_diff < len(gen) else None
    first_bit = _first_diff_bit(gt_b, gen_b)
    bit_txt = "n/a" if first_bit is None else f"{first_bit} (MSB-first within byte)"
    return (
        "<h2>First differing byte</h2>"
        "<div class='summary'>"
        f"<b>offset {first_diff}</b> | GT=0x{_hx(gt_b)} ({_bits(gt_b)}) | "
        f"GEN=0x{_hx(gen_b)} ({_bits(gen_b)}) | first differing bit={bit_txt}<br>"
        f"<div class='bitrow'>{_bit_cells(gt_b, gen_b)}</div>"
        "<div class='twocol'>"
        f"<div>{_overlap_list('GT spans overlapping byte', gt_syntax, first_diff)}</div>"
        f"<div>{_overlap_list('GEN spans overlapping byte', gen_syntax, first_diff)}</div>"
        "</div>"
        "</div>"
    )


def _first_diff(gt: bytes, gen: bytes) -> int | None:
    n = min(len(gt), len(gen))
    for i in range(n):
        if gt[i] != gen[i]:
            return i
    if len(gt) != len(gen):
        return n
    return None


def _common_prefix_len(gt: bytes, gen: bytes) -> int:
    n = min(len(gt), len(gen))
    for i in range(n):
        if gt[i] != gen[i]:
            return i
    return n


def _find_stream_pair(results_dir: Path, clip: int, checkpoint: str | None) -> tuple[Path, Path]:
    stream_root = results_dir / "streams"
    if not stream_root.exists():
        raise FileNotFoundError(f"no streams/ directory under {results_dir}")
    if checkpoint is None:
        candidates = sorted(p for p in stream_root.iterdir() if p.is_dir())
        if len(candidates) != 1:
            names = ", ".join(p.name for p in candidates[:10])
            raise ValueError(f"pass --checkpoint; found {len(candidates)} stream dirs: {names}")
        stream_dir = candidates[0]
    else:
        stream_dir = stream_root / checkpoint
    stem = f"clip_{clip:04d}_continuation"
    gt = stream_dir / f"{stem}_gt.h264"
    gen = stream_dir / f"{stem}_gen.h264"
    if not gt.exists() or not gen.exists():
        raise FileNotFoundError(f"missing {gt} or {gen}")
    return gt, gen


def _parse(data: bytes):
    return HS.parse_stream(data, parse_slice_data=True)


def _nal_title(nal_parse, idx: int) -> str:
    n = nal_parse.nal
    name = NAL_NAMES.get(n.nal_type, f"type {n.nal_type}")
    status = nal_parse.status.value
    return (
        f"NAL {idx} {name} type={n.nal_type} bytes="
        f"{n.start_code_start}:{n.payload_end} status={status}"
    )


def _span_rows(nal_parse, side: str) -> str:
    rows = []
    for i, s in enumerate(nal_parse.spans):
        color = CAT_COLORS.get(_cat(s), CAT_COLORS["unknown"])
        rows.append(
            "<tr>"
            f"<td>{side}</td><td>{i}</td>"
            f"<td><span class='syntax' style='background:{color}'>{html.escape(s.name)}</span></td>"
            f"<td>{_val(_jsonable(s.value))}</td>"
            f"<td>{s.byte_start}:{s.byte_end}</td>"
            f"<td>{s.bit_start}:{s.bit_end}</td>"
            f"<td>{'' if s.mb_addr is None else s.mb_addr}</td>"
            "</tr>"
        )
    return "".join(rows)


def _nal_compare_html(gt_parsed, gen_parsed, max_nals: int) -> str:
    parts = ["<h2>NAL syntax spans</h2>"]
    n = max(len(gt_parsed.nals), len(gen_parsed.nals))
    for idx in range(min(n, max_nals)):
        gt_nal = gt_parsed.nals[idx] if idx < len(gt_parsed.nals) else None
        gen_nal = gen_parsed.nals[idx] if idx < len(gen_parsed.nals) else None
        gt_title = _nal_title(gt_nal, idx) if gt_nal else "GT missing"
        gen_title = _nal_title(gen_nal, idx) if gen_nal else "GEN missing"
        open_attr = " open" if (gt_nal and gt_nal.status != HS.ParseStatus.OK) or (gen_nal and gen_nal.status != HS.ParseStatus.OK) else ""
        parts.append(f"<details{open_attr}><summary>{html.escape(gt_title)} | {html.escape(gen_title)}</summary>")
        for label, nal in (("GT", gt_nal), ("GEN", gen_nal)):
            if nal is not None and nal.status != HS.ParseStatus.OK:
                parts.append(
                    f"<div class='desync'>{label} {html.escape(nal.status.value)} "
                    f"bit={nal.desync_bit} byte={nal.desync_byte} "
                    f"failure={html.escape(str(nal.failure_element))} "
                    f"reason={html.escape(str(nal.reason_kind))}</div>"
                )
        parts.append("<table><tr><th>side</th><th>#</th><th>element</th><th>value</th><th>bytes</th><th>bits</th><th>mb</th></tr>")
        if gt_nal:
            parts.append(_span_rows(gt_nal, "GT"))
        if gen_nal:
            parts.append(_span_rows(gen_nal, "GEN"))
        parts.append("</table></details>")
    if n > max_nals:
        parts.append(f"<div class='summary'>NAL list truncated: {max_nals}/{n}. Raise --max-nals.</div>")
    return "".join(parts)


def _byte_rows(gt: bytes, gen: bytes, gt_syntax: list[dict[str, Any]], gen_syntax: list[dict[str, Any]], prefix_len: int | None, max_bytes: int) -> str:
    n = max(len(gt), len(gen))
    shown = min(n, max_bytes)
    rows = []
    for off in range(shown):
        gtb = gt[off] if off < len(gt) else None
        geb = gen[off] if off < len(gen) else None
        if gtb is None:
            cls = "extra"
        elif geb is None:
            cls = "missing"
        elif gtb == geb:
            cls = "same"
        else:
            cls = "diff"
        region = "prefix" if prefix_len is not None and off < prefix_len else "generated"
        gs = gt_syntax[off] if off < len(gt_syntax) else {"name": "?", "category": "unknown", "value": None, "mb_addr": None, "all": []}
        es = gen_syntax[off] if off < len(gen_syntax) else {"name": "?", "category": "unknown", "value": None, "mb_addr": None, "all": []}
        bit_cls = " bitdiff" if cls == "diff" else ""
        rows.append(
            f"<tr class='{cls} {region}{bit_cls}'>"
            f"<td>{off}</td><td>{region}</td><td>{_hx(gtb)}</td><td>{_hx(geb)}</td>"
            f"<td>{'same' if cls == 'same' else cls}<br><span class='bits'>{_bit_cells(gtb, geb)}</span></td>"
            f"<td>{_syntax_cell(gs)}</td><td>{html.escape(str(gs['value']))}</td>"
            f"<td>{_syntax_cell(es)}</td><td>{html.escape(str(es['value']))}</td>"
            "</tr>"
        )
    if n > shown:
        rows.append(
            f"<tr><td colspan='9'>Byte table truncated: {shown}/{n}. Raise --max-bytes.</td></tr>"
        )
    return "".join(rows)


def render(gt: bytes, gen: bytes, title: str, prefix_len: int | None, max_bytes: int, max_nals: int) -> str:
    gt_parsed = _parse(gt)
    gen_parsed = _parse(gen)
    gt_syntax = _assign_syntax(gt_parsed.all_spans(), len(gt))
    gen_syntax = _assign_syntax(gen_parsed.all_spans(), len(gen))
    first_diff = _first_diff(gt, gen)
    inferred_prefix = _common_prefix_len(gt, gen)
    same = sum(1 for i in range(min(len(gt), len(gen))) if gt[i] == gen[i])
    overlap = min(len(gt), len(gen))
    prefix_text = "none" if prefix_len is None else str(prefix_len)
    first_diff_text = "none" if first_diff is None else str(first_diff)
    gen_desync = next((n for n in gen_parsed.nals if n.status != HS.ParseStatus.OK), None)
    gt_desync = next((n for n in gt_parsed.nals if n.status != HS.ParseStatus.OK), None)

    body = [_HEAD.replace("__TITLE__", html.escape(title))]
    body.append(
        "<div class='summary'>"
        f"<b>{html.escape(title)}</b><br>"
        f"GT bytes={len(gt)} | GEN bytes={len(gen)} | overlap={overlap} | "
        f"same in overlap={same} ({same / overlap if overlap else 0:.3f})<br>"
        f"first byte diff={first_diff_text} | inferred common prefix={inferred_prefix} | "
        f"declared prefix={prefix_text}<br>"
        f"GT NALs={len(gt_parsed.nals)} first_bad={html.escape(str(gt_desync.failure_element if gt_desync else 'none'))} | "
        f"GEN NALs={len(gen_parsed.nals)} first_bad={html.escape(str(gen_desync.failure_element if gen_desync else 'none'))}"
        "</div>"
    )
    body.append(_first_diff_panel(gt, gen, gt_syntax, gen_syntax, first_diff))
    body.append(
        "<h2>Byte comparison</h2>"
        "<table><tr><th>off</th><th>region</th><th>GT</th><th>GEN</th><th>match</th>"
        "<th>GT syntax</th><th>GT value</th><th>GEN syntax</th><th>GEN value</th></tr>"
    )
    body.append(_byte_rows(gt, gen, gt_syntax, gen_syntax, prefix_len, max_bytes))
    body.append("</table>")
    body.append(_nal_compare_html(gt_parsed, gen_parsed, max_nals))
    body.append("</body></html>")
    return "".join(body)


_HEAD = """<!doctype html><html><head><meta charset='utf-8'><title>__TITLE__</title>
<style>
 body{background:#15161c;color:#e5e7eb;font:13px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:0;padding:16px}
 h2{font-size:16px;margin:18px 0 8px}
 .summary{background:#222532;border:1px solid #34384a;border-radius:6px;padding:10px 12px;margin-bottom:12px}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th,td{border-bottom:1px solid #2b2f3d;padding:3px 6px;text-align:left;vertical-align:top}
 th{position:sticky;top:0;background:#20232e;color:#aeb6c7;z-index:1}
 tr.same{background:#12251b}
 tr.diff{background:#311b1b}
 tr.missing{background:#32230f}
 tr.extra{background:#20203a}
 tr.prefix{opacity:.72}
 .syntax{display:inline-block;color:white;border-radius:3px;padding:1px 5px;white-space:nowrap;max-width:280px;overflow:hidden;text-overflow:ellipsis}
 .bits{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px}
 .bitrow{margin-top:8px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
 .bit{display:inline-block;border-radius:3px;padding:2px 4px;margin-right:3px;border:1px solid #3b4154}
 .samebit{background:#12251b;color:#bbf7d0}
 .diffbit{background:#3b1d1d;color:#fecaca}
 .firstbit{outline:2px solid #facc15}
 .twocol{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin-top:10px}
 .twocol h3{font-size:13px;margin:6px 0;color:#cbd5e1}
 .twocol ul{margin:4px 0 0 18px;padding:0}
 .twocol li{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;margin:2px 0}
 .muted{color:#94a3b8}
 details{margin:8px 0;border:1px solid #2c3040;border-radius:6px;overflow:hidden}
 summary{cursor:pointer;background:#222532;padding:8px 10px}
 .desync{background:#3a1c1c;color:#fecaca;border-left:3px solid #ef4444;padding:7px 10px}
</style></head><body>"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", type=Path, help="eval output dir containing streams/<checkpoint>/")
    ap.add_argument("--clip", type=int, default=0, help="clip index for results_dir mode")
    ap.add_argument("--checkpoint", default=None, help="stream subdirectory under results_dir/streams")
    ap.add_argument("--gt", type=Path, default=None, help="explicit GT Annex-B stream")
    ap.add_argument("--gen", type=Path, default=None, help="explicit generated Annex-B stream")
    ap.add_argument("--out", type=Path, default=None, help="output HTML path")
    ap.add_argument("--prefix-bytes", type=int, default=None, help="known GT prefix length; defaults to inferred common prefix")
    ap.add_argument("--max-bytes", type=int, default=20000, help="maximum byte rows to render")
    ap.add_argument("--max-nals", type=int, default=400, help="maximum NAL span sections to render")
    ap.add_argument("--title", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.gt is not None or args.gen is not None:
        if args.gt is None or args.gen is None:
            raise SystemExit("--gt and --gen must be passed together")
        gt_path, gen_path = args.gt, args.gen
    else:
        if args.results_dir is None:
            raise SystemExit("pass RESULTS_DIR or explicit --gt/--gen")
        gt_path, gen_path = _find_stream_pair(args.results_dir, args.clip, args.checkpoint)

    gt = gt_path.read_bytes()
    gen = gen_path.read_bytes()
    prefix_len = args.prefix_bytes if args.prefix_bytes is not None else _common_prefix_len(gt, gen)
    title = args.title or f"{gt_path.name} vs {gen_path.name}"
    out = args.out
    if out is None:
        base = args.results_dir if args.results_dir is not None else gen_path.parent
        out = base / f"byte_compare_clip_{args.clip:04d}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(gt, gen, title, prefix_len, args.max_bytes, args.max_nals), encoding="utf-8")

    meta = {
        "out": str(out),
        "gt": str(gt_path),
        "gen": str(gen_path),
        "gt_bytes": len(gt),
        "gen_bytes": len(gen),
        "prefix_bytes": prefix_len,
        "first_diff": _first_diff(gt, gen),
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
