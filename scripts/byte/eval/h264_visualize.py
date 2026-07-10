"""Visualize an H.264 Annex-B byte stream (clean OR model-generated / corrupted) as a
self-contained HTML page: every NAL broken into its CAVLC syntax elements, each showing
its raw bits, its decoded value, and -- crucially for free-run debugging -- exactly where
and why the stream desyncs.

Use it to eyeball what the model actually generated: e.g. "start code + NAL header + slice
header decode fine, then mb_type = 47 (illegal, >30) at bit 112 -> DESYNC".

Usage:
    python scripts/byte/eval/h264_visualize.py stream.h264 --out stream.html
    # mark where a clean GT prefix ends and model generation begins:
    python scripts/byte/eval/h264_visualize.py clip.h264 --prefix-bytes 4096 --out clip.html
    # limit huge streams:
    python scripts/byte/eval/h264_visualize.py big.h264 --max-nals 200 --out big.html

The parser (litgpt.byte.h264_syntax) is the same one the eval uses, so the spans, values,
and desync location here are exactly what the desync-syntax metrics see.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte import h264_syntax as HS  # noqa: E402

_NAL_NAMES = {1: "non-IDR slice", 5: "IDR slice", 6: "SEI", 7: "SPS", 8: "PPS", 9: "AUD"}

# Category -> (background, label) for coloring the syntax elements.
_CAT_COLOR = {
    "start_code": "#3b3b52",
    "nal_header": "#4a3f6b",
    "sps": "#2f5d50",
    "pps": "#2f5d50",
    "slice_header": "#1f4e79",
    "mb_header": "#7a4d1f",   # mb_type / mb_skip_run / sub_mb_type
    "mb_pred": "#8a5a2b",
    "cbp": "#9a6a1a",
    "residual_luma": "#5a2f6b",
    "residual_chroma": "#6b2f5a",
    "mb_qp_delta": "#3f5a2f",
    "rbsp_trailing": "#333",
    "sei": "#444",
    "unknown": "#5a1f1f",
}


def _span_own_bits(data: bytes, span) -> str:
    """The field's own bits (MSB-first), read from the raw stream at
    byte_start*8 + (bit_start & 7). Exact when the field has no interior emulation-
    prevention byte (the common case); a rare interior 00-00-03 shifts it, flagged by
    the raw-vs-rbsp byte-count mismatch the caller can check."""
    n = span.bit_end - span.bit_start
    if n <= 0:
        return ""
    start = span.byte_start * 8 + (span.bit_start & 7)
    out = []
    for i in range(min(n, 64)):  # cap very long fields
        p = start + i
        b = data[p >> 3] if (p >> 3) < len(data) else 0
        out.append(str((b >> (7 - (p & 7))) & 1))
    s = "".join(out)
    return s + ("…" if n > 64 else "")


def _fmt_value(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}={v[k]}" for k in v)
    return html.escape(str(v))


def _desync_region(nal) -> str:
    """Name of the element right before the desync (the break is in the next one)."""
    if nal.desync_bit is None:
        return "?"
    last = None
    for s in nal.spans:
        if s.bit_start <= nal.desync_bit:
            last = s
    return last.name if last is not None else "(start of NAL)"


def _nal_html(idx: int, nal, data: bytes, is_generated: bool) -> str:
    info = nal.nal
    nt = info.nal_type
    name = _NAL_NAMES.get(nt, f"type {nt}")
    status = nal.status.value
    badge = {"ok": "#2e7d32", "desync": "#c62828", "unsupported": "#f9a825"}.get(status, "#555")
    gen_tag = " <span class='gen'>GEN</span>" if is_generated else ""
    parts = [
        f"<details {'open' if status != 'ok' else ''}>",
        f"<summary><span class='badge' style='background:{badge}'>{status.upper()}</span> "
        f"NAL #{idx} · <b>{name}</b> (type {nt}) · bytes [{info.start_code_start}:{info.payload_end}) "
        f"· {len(nal.spans)} elems{gen_tag}</summary>",
    ]
    if status != "ok":
        parts.append(
            f"<div class='desync'>⚠ {status.upper()} at bit {nal.desync_bit} "
            f"(byte {nal.desync_byte}) · region <b>{html.escape(_desync_region(nal))}</b> "
            f"· reason <b>{html.escape(str(nal.reason_kind))}</b>"
            + (f" · {html.escape(str(nal.reason))}" if nal.reason else "")
            + "</div>"
        )
    parts.append("<table><tr><th>#</th><th>element</th><th>value</th><th>bytes</th>"
                 "<th>bits (RBSP)</th><th>bits</th></tr>")
    for i, s in enumerate(nal.spans):
        color = _CAT_COLOR.get(s.category.value, "#444")
        is_last = (nal.desync_bit is not None and s.bit_start <= nal.desync_bit
                   and (i + 1 == len(nal.spans) or nal.spans[i + 1].bit_start > nal.desync_bit))
        row_cls = " class='crash'" if (status == "desync" and is_last) else ""
        parts.append(
            f"<tr{row_cls}><td>{i}</td>"
            f"<td><span class='cat' style='background:{color}'>{html.escape(s.name)}</span></td>"
            f"<td>{_fmt_value(s.value)}</td>"
            f"<td>{s.byte_start}:{s.byte_end}</td>"
            f"<td>{s.bit_start}:{s.bit_end} ({s.bit_end - s.bit_start}b)</td>"
            f"<td class='bits'>{_span_own_bits(data, s)}</td></tr>"
        )
    parts.append("</table></details>")
    return "".join(parts)


def render(data: bytes, prefix_len: int, max_nals: int, title: str) -> str:
    parsed = HS.parse_stream(data, parse_slice_data=True)
    nals = parsed.nals
    n_ok = sum(1 for n in nals if n.status == HS.ParseStatus.OK)
    n_desync = sum(1 for n in nals if n.status == HS.ParseStatus.DESYNC)
    n_unsup = sum(1 for n in nals if n.status == HS.ParseStatus.UNSUPPORTED)
    first_bad = next((n for n in nals if n.status != HS.ParseStatus.OK), None)
    first_bad_txt = (
        f"first desync: NAL @byte {first_bad.nal.start_code_start}, "
        f"region {html.escape(_desync_region(first_bad))}, {first_bad.status.value}"
        if first_bad else "no desync — fully valid"
    )

    body = [_HEAD.replace("__TITLE__", html.escape(title))]
    body.append(
        f"<div class='summary'><b>{html.escape(title)}</b> · {len(data)} bytes · {len(nals)} NALs "
        f"· <span style='color:#7fd18a'>{n_ok} ok</span> "
        f"· <span style='color:#ff8a8a'>{n_desync} desync</span> "
        f"· <span style='color:#ffd76a'>{n_unsup} unsupported</span><br>{first_bad_txt}"
        + (f" · prefix ends at byte {prefix_len} (GT); NALs after are model-generated"
           if prefix_len else "")
        + "</div>"
    )
    shown = 0
    for idx, nal in enumerate(nals):
        if max_nals and shown >= max_nals:
            body.append(f"<div class='summary'>… {len(nals) - shown} more NALs truncated "
                        f"(raise --max-nals)</div>")
            break
        is_gen = bool(prefix_len) and nal.nal.start_code_start >= prefix_len
        body.append(_nal_html(idx, nal, data, is_gen))
        shown += 1
    body.append("</body></html>")
    return "".join(body)


_HEAD = """<!doctype html><html><head><meta charset='utf-8'><title>__TITLE__</title>
<style>
 body{background:#16161c;color:#e0e0e6;font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:16px}
 .summary{background:#22222c;padding:10px 14px;border-radius:6px;margin:10px 0;font-size:13px}
 details{margin:6px 0;border:1px solid #2c2c38;border-radius:6px;overflow:hidden}
 summary{cursor:pointer;padding:8px 12px;background:#22222c;user-select:none}
 summary:hover{background:#2a2a36}
 .badge{color:#fff;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:700}
 .gen{background:#b8860b;color:#fff;padding:1px 6px;border-radius:4px;font-size:10px}
 .desync{background:#3a1c1c;color:#ffb3b3;padding:8px 12px;border-left:3px solid #c62828}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th,td{padding:3px 8px;text-align:left;border-bottom:1px solid #2a2a34;vertical-align:top}
 th{color:#9a9aa6;font-weight:600;position:sticky;top:0;background:#1c1c24}
 .cat{color:#fff;padding:1px 6px;border-radius:3px;font-size:11px;white-space:nowrap}
 .bits{font-family:ui-monospace,Menlo,monospace;color:#8fd3ff;word-break:break-all}
 tr.crash{background:#3a1c1c}
 tr.crash .cat{outline:2px solid #ff5252}
</style></head><body>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Annex-B .h264 byte stream")
    ap.add_argument("--out", type=Path, required=True, help="output .html")
    ap.add_argument("--prefix-bytes", type=int, default=0,
                    help="mark NALs starting at/after this byte as model-generated")
    ap.add_argument("--max-nals", type=int, default=500)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    data = args.input.read_bytes()
    title = args.title or args.input.name
    args.out.write_text(render(data, args.prefix_bytes, args.max_nals, title), encoding="utf-8")
    print(f"Wrote {args.out} ({len(data)} bytes, prefix_bytes={args.prefix_bytes})")


if __name__ == "__main__":
    main()
