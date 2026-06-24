"""Torch-free tests for the byte diagnoser's pure overlay + HTML render path.

The teacher-forced forward and free-run generation need torch + a checkpoint (run
in the training env); here we exercise everything else -- syntax overlay, record
building, per-category accuracy, and HTML rendering -- against a real fixture
window with synthetic per-byte predictions, so the visualization is verifiable
without torch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str, relpath: str):
    path = _REPO_ROOT / relpath
    byte_dir = _REPO_ROOT / "litgpt" / "byte"
    if str(byte_dir) not in sys.path:
        sys.path.insert(0, str(byte_dir))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DB = _load("diagnose_bytes", "scripts/byte/eval/diagnose_bytes.py")
HS = DB.HS


def _fixture_window() -> bytes:
    path = _FIXTURES / "baseline_qcif.h264"
    if not path.exists():
        pytest.skip("fixture missing")
    return path.read_bytes()


def test_syntax_overlay_covers_every_byte():
    window = _fixture_window()
    parsed = HS.parse_stream(window, parse_slice_data=True)
    syntax = DB.assign_syntax(parsed.all_spans(), len(window))
    assert len(syntax) == len(window)
    # Every byte got a concrete (non-unknown) syntax label.
    assert all(s["category"] != "unknown" for s in syntax)
    # Residual + slice-header + start_code categories all appear.
    cats = {s["category"] for s in syntax}
    assert {"start_code", "nal_header", "slice_header", "residual_luma"} <= cats


def test_build_records_and_category_accuracy():
    window = _fixture_window()
    parsed = HS.parse_stream(window, parse_slice_data=True)
    spans = parsed.all_spans()
    # Synthetic teacher-forced predictions: pretend the model nails low-entropy
    # structural bytes and guesses on residuals -- just to exercise the math.
    preds = []
    for off, b in enumerate(window):
        correct = (off % 3 != 0)
        preds.append({
            "argmax": b if correct else (b ^ 1),
            "p_target": 0.9 if correct else 0.1,
            "entropy": 0.5 if correct else 4.0,
            "correct": correct,
            "topk": [(b, 0.9 if correct else 0.1)],
        })
    records = DB.build_records(window, spans, preds, [1] * len(window), list(range(len(window))))
    assert len(records) == len(window)
    rows = DB.category_accuracy(records)
    assert rows and all(0.0 <= r["top1_acc"] <= 1.0 and r["bits_per_byte"] >= 0 for r in rows)
    assert sum(r["bytes"] for r in rows) == len(window)


def test_render_html_is_self_contained(tmp_path):
    window = _fixture_window()
    parsed = HS.parse_stream(window, parse_slice_data=True)
    records = DB.build_records(window, parsed.all_spans(), None)
    out = DB.render_html(
        records, title="t", summary={"window_bytes": len(window)},
        cat_rows=[], freerun={"first_syntax_desync": "none"},
    )
    assert out.startswith("<!doctype html>") and "</html>" in out
    assert "Free-run structural diff" in out
    # Renders without external resources (offline-usable).
    assert "http://" not in out and "https://" not in out
    (tmp_path / "out.html").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    window = _fixture_window()
    parsed = HS.parse_stream(window, parse_slice_data=True)
    spans = parsed.all_spans()
    preds = [{"argmax": b, "p_target": 0.8, "entropy": 1.0, "correct": True,
              "topk": [(b, 0.8)]} for b in window]
    records = DB.build_records(window, spans, preds, [1] * len(window), list(range(len(window))))
    rows = DB.category_accuracy(records)
    out = DB.render_html(records, title="demo", summary={"window_bytes": len(window)},
                         cat_rows=rows, freerun={"first_syntax_desync": "none (demo)"})
    demo_dir = _REPO_ROOT / "tmp"
    demo_dir.mkdir(exist_ok=True)
    demo = demo_dir / "diagnose_demo.html"
    demo.write_text(out, encoding="utf-8")
    print("per-category accuracy rows:")
    for r in rows:
        print(f"  {r['category']:20s} bytes={r['bytes']:6d} top1={r['top1_acc']*100:5.1f}% bits/byte={r['bits_per_byte']:.2f}")
    print(f"wrote demo HTML ({len(out)} bytes) -> {demo}")
