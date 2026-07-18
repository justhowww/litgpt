"""Syntax-aware byte-level diagnoser for the Stage 0 (H0) AR byte-LM.

Single-clip deep dive: pick one stream window, then emit a self-contained
interactive HTML report that overlays the model's per-byte behaviour on the
H.264 syntax structure. Pixel metrics (PSNR/SSIM) are confounded by the decoder's
error concealment (0607.md:214); this looks at the bytes directly and labels each
one with the syntax element it belongs to (NAL header / SPS-PPS / slice header /
mb_type / MV / CBP / CAVLC residual coefficient), via litgpt/byte/h264_syntax.py.

Three views (all share one window):
  1. Teacher-forced per-byte -- feed GT bytes, record per position the argmax
     byte, p(GT), entropy and top-k, tagged with its (exactly known) syntax span.
     Aggregated to top-1 accuracy / bits-per-byte per syntax category.
  2. Free-run structural diff -- free-run generate, parse the result, and report
     the syntax-level desync point + NAL alignment vs GT.
  3. Hex/byte grid -- the per-byte grid doubles as a syntax-coloured hex view with
     a syntax<->correctness toggle and per-byte tooltips.

The syntax overlay and HTML rendering are pure (torch-free) so they can be unit
tested without the training environment; torch is imported lazily in ``main``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# h264_syntax is stdlib-only. Prefer the package import; fall back to a direct
# file load so the pure (torch-free) helpers import even without the training env
# (litgpt/__init__ pulls in lightning).
try:
    from litgpt.byte import h264_syntax as HS  # noqa: E402
except Exception:  # pragma: no cover - exercised only outside the training env
    import importlib.util

    _byte_dir = _REPO_ROOT / "litgpt" / "byte"
    if str(_byte_dir) not in sys.path:
        sys.path.insert(0, str(_byte_dir))
    _spec = importlib.util.spec_from_file_location("h264_syntax", _byte_dir / "h264_syntax.py")
    HS = importlib.util.module_from_spec(_spec)
    sys.modules["h264_syntax"] = HS
    _spec.loader.exec_module(HS)

# Category -> display colour (background in syntax mode).
CATEGORY_COLORS: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# Pure helpers (no torch) -- syntax overlay, record building, HTML render
# ---------------------------------------------------------------------------


def assign_syntax(spans: list[HS.SyntaxSpan], n_bytes: int) -> list[dict[str, Any]]:
    """Per-byte primary syntax element (most byte-local overlapping span) + all overlaps.

    A byte may carry bits from several sub-byte syntax elements; the "primary"
    label is the overlapping span with the smallest byte range (the most specific
    element occupying that byte).
    """
    by_byte: list[list[HS.SyntaxSpan]] = [[] for _ in range(n_bytes)]
    for span in spans:
        lo = max(0, span.byte_start)
        hi = min(n_bytes, span.byte_end)
        for off in range(lo, hi):
            by_byte[off].append(span)
    out: list[dict[str, Any]] = []
    for off in range(n_bytes):
        overlaps = by_byte[off]
        if not overlaps:
            out.append({"name": "?", "category": "unknown", "value": None, "mb_addr": None, "all": []})
            continue
        primary = min(overlaps, key=lambda s: (s.byte_end - s.byte_start, s.byte_start))
        out.append({
            "name": primary.name,
            "category": primary.category.value if hasattr(primary.category, "value") else str(primary.category),
            "value": _jsonable(primary.value),
            "mb_addr": primary.mb_addr,
            "all": [s.name for s in overlaps],
        })
    return out


def build_records(
    window_bytes: bytes,
    spans: list[HS.SyntaxSpan],
    preds: list[dict[str, Any]] | None = None,
    region_ids: list[int] | None = None,
    offset_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Merge GT bytes + syntax overlay + (optional) teacher-forced predictions."""
    n = len(window_bytes)
    syntax = assign_syntax(spans, n)
    records: list[dict[str, Any]] = []
    for off in range(n):
        rec: dict[str, Any] = {
            "off": off,
            "gt": window_bytes[off],
            "syntax": syntax[off],
        }
        if region_ids is not None:
            rec["region"] = region_ids[off]
        if offset_ids is not None:
            rec["offset"] = offset_ids[off]
        if preds is not None:
            rec.update(preds[off])
        records.append(rec)
    return records


def category_accuracy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-syntax-category top-1 accuracy + mean bits/byte (teacher-forced)."""
    agg: dict[str, dict[str, float]] = {}
    for rec in records:
        if "p_target" not in rec:
            continue
        cat = rec["syntax"]["category"]
        a = agg.setdefault(cat, {"n": 0, "correct": 0, "bits": 0.0})
        a["n"] += 1
        a["correct"] += 1 if rec.get("correct") else 0
        p = max(rec["p_target"], 1e-12)
        a["bits"] += -math.log2(p)
    rows = []
    for cat, a in agg.items():
        n = a["n"]
        rows.append({
            "category": cat,
            "bytes": int(n),
            "top1_acc": a["correct"] / n if n else 0.0,
            "bits_per_byte": a["bits"] / n if n else 0.0,
        })
    rows.sort(key=lambda r: -r["bytes"])
    return rows


def _name_stem(name: str) -> str:
    """Normalize a syntax-element name into a grouping key by dropping array
    indices and block numbers: ``mvd_l0[0].x`` -> ``mvd_l0.x``, ``sub_mb_type[1]``
    -> ``sub_mb_type``. Keeps ``mb_type`` distinct from ``mb_skip_run`` etc. so the
    per-element view isolates the structural decisions the category view bundles."""
    import re

    s = re.sub(r"\[\d+\]", "", name)
    s = re.sub(r"_\d+(?=\b|\.|$)", "", s)
    return s


def element_accuracy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-syntax-ELEMENT top-1 + mean bits/byte (teacher-forced), keyed by the
    normalized element name rather than the (coarser) category. This separates,
    e.g., ``mb_type`` from the rest of ``mb_header`` so a low category number can
    be attributed to the element that actually carries it."""
    agg: dict[str, dict[str, Any]] = {}
    for rec in records:
        if "p_target" not in rec:
            continue
        stem = _name_stem(rec["syntax"]["name"])
        cat = rec["syntax"]["category"]
        a = agg.setdefault(stem, {"n": 0, "correct": 0, "bits": 0.0, "category": cat})
        a["n"] += 1
        a["correct"] += 1 if rec.get("correct") else 0
        a["bits"] += -math.log2(max(rec["p_target"], 1e-12))
    rows = []
    for stem, a in agg.items():
        n = a["n"]
        rows.append({
            "element": stem,
            "category": a["category"],
            "bytes": int(n),
            "top1_acc": a["correct"] / n if n else 0.0,
            "bits_per_byte": a["bits"] / n if n else 0.0,
        })
    rows.sort(key=lambda r: -r["bytes"])
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _hexb(b: int) -> str:
    return f"{b:02x}"


def _tooltip(rec: dict[str, Any]) -> str:
    s = rec["syntax"]
    parts = [f"#{rec['off']}  GT=0x{_hexb(rec['gt'])}"]
    if s["mb_addr"] is not None:
        parts.append(f"mb={s['mb_addr']}")
    parts.append(f"{s['name']}")
    if s["value"] is not None:
        parts.append(f"val={s['value']}")
    if "region" in rec:
        parts.append(f"region={rec['region']} off={rec.get('offset')}")
    if "p_target" in rec:
        parts.append(
            f"pred=0x{_hexb(rec['argmax'])} p(gt)={rec['p_target']:.3f} "
            f"H={rec['entropy']:.2f} {'OK' if rec.get('correct') else 'X'}"
        )
        if rec.get("topk"):
            tk = " ".join(f"{_hexb(b)}:{p:.2f}" for b, p in rec["topk"])
            parts.append(f"top: {tk}")
    if len(s.get("all", [])) > 1:
        parts.append("[" + ", ".join(s["all"]) + "]")
    return html.escape("  |  ".join(parts))


def render_html(
    records: list[dict[str, Any]],
    *,
    title: str,
    summary: dict[str, Any],
    cat_rows: list[dict[str, Any]],
    freerun: dict[str, Any] | None,
    elem_rows: list[dict[str, Any]] | None = None,
    bytes_per_row: int = 32,
) -> str:
    elem_rows = elem_rows or []
    has_preds = any("p_target" in r for r in records)
    # Group bytes by NAL for collapsible sections (nal_addr inferred from spans
    # is not per-byte; instead break on start_code category boundaries).
    cells: list[str] = []
    for rec in records:
        cat = rec["syntax"]["category"]
        color = CATEGORY_COLORS.get(cat, "#94a3b8")
        if has_preds and "p_target" in rec:
            p = rec["p_target"]
            corr = "1" if rec.get("correct") else "0"
            cgreen = f"rgba(22,163,74,{0.15 + 0.7 * p:.3f})"
            cred = f"rgba(220,38,38,{0.25 + 0.6 * (1 - p):.3f})"
            correct_bg = cgreen if rec.get("correct") else cred
        else:
            corr = "1"
            correct_bg = "transparent"
        tip = _tooltip(rec)
        cells.append(
            f'<span class="b" data-correct="{corr}" '
            f'style="--cat:{color};--corr:{correct_bg}" title="{tip}">{_hexb(rec["gt"])}</span>'
        )
        if (rec["off"] + 1) % bytes_per_row == 0:
            cells.append('<br>')
    grid = "".join(cells)

    cat_table = "".join(
        f"<tr><td><span class='swatch' style='background:{CATEGORY_COLORS.get(r['category'], '#94a3b8')}'></span>"
        f"{html.escape(r['category'])}</td><td>{r['bytes']}</td>"
        f"<td>{r['top1_acc']*100:.1f}%</td><td>{r['bits_per_byte']:.2f}</td></tr>"
        for r in cat_rows
    )

    elem_table = "".join(
        f"<tr><td>{html.escape(r['element'])}</td>"
        f"<td><span class='swatch' style='background:{CATEGORY_COLORS.get(r['category'], '#94a3b8')}'></span>"
        f"{html.escape(r['category'])}</td><td>{r['bytes']}</td>"
        f"<td>{r['top1_acc']*100:.1f}%</td><td>{r['bits_per_byte']:.2f}</td></tr>"
        for r in elem_rows
    )

    summary_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
    )

    freerun_html = ""
    if freerun is not None:
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in freerun.items()
        )
        freerun_html = f"<h2>Free-run structural diff</h2><table class='kv'>{rows}</table>"

    legend = "".join(
        f"<span class='leg'><span class='swatch' style='background:{c}'></span>{html.escape(k)}</span>"
        for k, c in CATEGORY_COLORS.items()
    )

    toggle = ""
    if has_preds:
        toggle = ("<button id='toggle' onclick=\"document.getElementById('grid')."
                  "classList.toggle('mode-correct')\">toggle syntax / correctness</button>")

    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
 body {{ font-family: ui-monospace, Menlo, monospace; background:#0f172a; color:#e2e8f0; margin:16px; }}
 h1 {{ font-size:16px; }} h2 {{ font-size:14px; margin-top:20px; }}
 table {{ border-collapse:collapse; margin:8px 0; font-size:12px; }}
 td, th {{ border:1px solid #334155; padding:2px 8px; text-align:left; }}
 .swatch {{ display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; vertical-align:middle; }}
 .leg {{ display:inline-block; margin:2px 10px 2px 0; font-size:11px; }}
 #grid {{ line-height:1.5; font-size:13px; letter-spacing:1px; word-break:break-all; }}
 .b {{ padding:1px 1px; background:var(--cat); color:#f8fafc; cursor:default; }}
 #grid.mode-correct .b {{ background:var(--corr); color:#0f172a; }}
 #grid.mode-correct .b[data-correct='0'] {{ font-weight:bold; }}
 button {{ background:#1e293b; color:#e2e8f0; border:1px solid #475569; padding:4px 10px; cursor:pointer; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<table class='kv'>{summary_rows}</table>
{('<h2>Per-syntax-category accuracy (teacher-forced)</h2>'
  '<table><tr><th>category</th><th>bytes</th><th>top-1</th><th>bits/byte</th></tr>'
  + cat_table + '</table>') if has_preds else ''}
{('<h2>Per-syntax-element accuracy (teacher-forced)</h2>'
  '<table><tr><th>element</th><th>category</th><th>bytes</th><th>top-1</th><th>bits/byte</th></tr>'
  + elem_table + '</table>') if has_preds else ''}
{freerun_html}
<h2>Byte grid (GT bytes, coloured by syntax; hover for details)</h2>
{toggle}
<div style='margin:6px 0'>{legend}</div>
<div id='grid'>{grid}</div>
</body></html>"""


# ---------------------------------------------------------------------------
# torch-dependent orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path)
    p.add_argument("--nal-index-path", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-manifest-rows", type=int, default=0)
    p.add_argument("--h264-path", type=Path, default=None, help="Pick a specific video.")
    p.add_argument("--window-index", type=int, default=0, help="Which qualifying window.")
    p.add_argument("--prefix-frames", type=int, default=8)
    p.add_argument("--cont-frames", type=int, default=4)
    p.add_argument("--max-window-bytes", type=int, default=16384)
    p.add_argument("--min-frames", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-freerun", action="store_true", help="Skip the free-run view.")
    p.add_argument("--ffmpeg-binary", default="ffmpeg")
    p.add_argument("--timeout-sec", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch  # noqa: F401
    from litgpt.byte.data import (
        ByteStreamWindowDataset,
        default_nal_index_path,
        load_manifest_rows,
        load_nal_index,
    )
    from scripts.byte.eval.helpers.checkpoint_eval_helpers import jsonable, load_model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = load_manifest_rows(args.manifest, args.max_manifest_rows or None)
    if args.h264_path is not None:
        target = str(args.h264_path)
        rows = [r for r in rows if str(Path(r["h264_path"])) == target] or rows
    index_path = args.nal_index_path or default_nal_index_path(args.manifest)
    nal_index = load_nal_index(index_path, args.manifest, rows)

    dataset = ByteStreamWindowDataset(
        rows, args.max_window_bytes, min_frames=args.min_frames, nal_index=nal_index
    )
    needed = args.prefix_frames + args.cont_frames
    # Only windows that begin at the SPS are self-contained (parseable / decodable).
    # ~27% of windows start at the IDR because _windows_for_video's back-up stops at
    # an intervening SEI/AUD and drops the SPS/PPS; those can't be parsed, so skip
    # them here. (See the _windows_for_video back-up note.)
    qualifying = [
        i for i, s in enumerate(dataset.samples)
        if s.num_frames >= needed
        and nal_index[str(s.h264_path)][s.start_nal].nal_type == HS.NAL_SPS
    ]
    if not qualifying:
        raise SystemExit(
            f"No self-contained (SPS-anchored) window has >= {needed} frames; "
            "lower --prefix/--cont-frames."
        )
    item = dataset[qualifying[min(args.window_index, len(qualifying) - 1)]]

    window_bytes = bytes(item["labels"].tolist())
    region_ids = item["region_ids"].tolist()
    offset_ids = item["offset_ids"].tolist()

    # Parse the window bytes once -> byte->syntax map (window-offset space).
    parsed = HS.parse_stream(window_bytes, parse_slice_data=True)
    spans = parsed.all_spans()

    model = load_model(args.checkpoint_dir, device)
    preds = _teacher_forced(model, item, device, args.top_k)

    records = build_records(window_bytes, spans, preds, region_ids, offset_ids)
    cat_rows = category_accuracy(records)
    elem_rows = element_accuracy(records)
    overall_acc = sum(1 for r in records if r.get("correct")) / max(1, len(records))
    overall_bits = sum(-math.log2(max(r["p_target"], 1e-12)) for r in records) / max(1, len(records))

    desyncs = [p for p in parsed.nals if p.status != HS.ParseStatus.OK]
    summary = {
        "checkpoint": args.checkpoint_dir.name,
        "h264_path": item["sample_meta"]["h264_path"],
        "window_frames": item["sample_meta"]["num_frames"],
        "window_bytes": len(window_bytes),
        "overall_top1_acc": f"{overall_acc*100:.2f}%",
        "overall_bits_per_byte": f"{overall_bits:.3f}",
        "gt_parse_desyncs": len(desyncs),
    }

    freerun = None
    if not args.no_freerun:
        freerun = _freerun_diff(model, item, dataset, device, args)

    title = f"Byte diagnosis - {args.checkpoint_dir.name} - {Path(summary['h264_path']).name}"
    htmlout = render_html(
        records, title=title, summary=summary, cat_rows=cat_rows,
        elem_rows=elem_rows, freerun=freerun,
    )
    stem = f"diagnose_{args.checkpoint_dir.name}_w{args.window_index}"
    (args.out_dir / f"{stem}.html").write_text(htmlout, encoding="utf-8")
    (args.out_dir / f"{stem}_bytes.json").write_text(
        json.dumps({"summary": summary, "category_accuracy": cat_rows,
                    "element_accuracy": elem_rows,
                    "records": [_record_json(r) for r in records], "freerun": freerun},
                   indent=2), encoding="utf-8")
    (args.out_dir / "config.json").write_text(json.dumps(jsonable(vars(args)), indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir / (stem + '.html')}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def _record_json(rec: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in rec.items() if k != "topk"}
    if "topk" in rec:
        out["topk"] = [[int(b), float(p)] for b, p in rec["topk"]]
    return _jsonable(out)


def _teacher_forced(model, item, device, top_k: int) -> list[dict[str, Any]]:
    import torch

    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    input_ids = item["input_ids"].to(device).unsqueeze(0)
    region = item["region_ids"].to(device).unsqueeze(0)
    offset = item["offset_ids"].to(device).unsqueeze(0)
    labels = item["labels"]
    # Full teacher-forced forward: no input_pos (that path demands a KV cache);
    # input_pos=None gives the default causal mask + arange positions, which is
    # exactly teacher forcing.
    with torch.no_grad():
        logits = raw(input_ids, region_ids=region, offset_ids=offset)
    logits = logits[0, :, :256].float()
    probs = torch.softmax(logits, dim=-1)
    argmax = probs.argmax(dim=-1)
    entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=-1)
    tk_p, tk_i = probs.topk(min(top_k, 256), dim=-1)
    preds: list[dict[str, Any]] = []
    for i in range(labels.numel()):
        gt = int(labels[i])
        preds.append({
            "argmax": int(argmax[i]),
            "p_target": float(probs[i, gt]),
            "entropy": float(entropy[i]),
            "correct": bool(int(argmax[i]) == gt),
            "topk": [(int(tk_i[i, j]), float(tk_p[i, j])) for j in range(tk_i.size(1))],
        })
    return preds


def _freerun_diff(model, item, dataset, device, args) -> dict[str, Any]:
    """Free-run from the window's first prefix_frames; parse + structural diff."""
    from litgpt.byte.data import NALUnit, parse_annexb_nals

    window_bytes = bytes(item["labels"].tolist())
    nals = parse_annexb_nals(window_bytes)
    vcl_idx = [i for i, n in enumerate(nals) if n.nal_type in (1, 5)]
    if len(vcl_idx) <= args.prefix_frames:
        return {"status": "window too short for free-run"}
    prefix_end_nal = vcl_idx[args.prefix_frames]
    prefix_bytes = b"".join(window_bytes[nals[i].start:nals[i].end] for i in range(prefix_end_nal))

    gen_bytes = _generate(model, prefix_bytes, nals, prefix_end_nal, device, args)
    gen_parsed = HS.parse_stream(prefix_bytes + gen_bytes, parse_slice_data=True)
    gen_vcl = [p for p in gen_parsed.nals if p.nal.nal_type in (1, 5)]
    first_bad = next((p for p in gen_vcl if p.status != HS.ParseStatus.OK), None)
    return {
        "prefix_frames": args.prefix_frames,
        "generated_bytes": len(gen_bytes),
        "generated_nals": len(gen_parsed.nals),
        "generated_vcl_frames": len(gen_vcl),
        "first_syntax_desync": (
            f"nal {first_bad.nal.index} ({first_bad.status.value}: {first_bad.reason}) "
            f"@byte {first_bad.desync_byte}" if first_bad else "none (all generated frames parse clean)"
        ),
    }


def _generate(model, prefix_bytes, nals, prefix_end_nal, device, args) -> bytes:
    """Thin wrapper around the continuation generator used by the stream eval."""
    from scripts.byte.eval.eval_ar_continuation import generate_continuation, model_max_gen

    max_gen = model_max_gen(model, prefix_bytes)
    gen, _ = generate_continuation(model, prefix_bytes, nals, prefix_end_nal, device,
                                   args, args.cont_frames, max_gen)
    return gen


if __name__ == "__main__":
    main()
