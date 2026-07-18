"""Consolidate an AVC-LM sampling sweep (eval_avclm.sh output) into one table:
teacher-forced bit-accuracy (sampling-independent) + free-run desync by regime.

Usage:
    python scripts/byte/eval/summarize_avclm_sweep.py OUT_ROOT [config ...]
    # default configs: greedy temp1 avclm_topk
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(root: Path, name: str):
    p = root / name / "metrics.jsonl"
    if not p.exists():
        return None, None
    tf = cont = None
    for line in p.open():
        r = json.loads(line)
        if r.get("mode") == "teacher_forced":
            tf = r
        elif r.get("mode") == "continuation":
            cont = r
    return tf, cont


def main() -> None:
    root = Path(sys.argv[1])
    names = sys.argv[2:] or ["greedy", "temp1", "avclm_topk"]

    tf0, _ = rows(root, names[0])
    if tf0:
        print("\n[teacher-forced value metrics | sampling-independent]")
        print("  argmax-decodes-to: CORRECT value (exact) | LEGAL value (would not desync)")
        cor = tf0.get("element_correct", {})
        leg = tf0.get("element_legal", {})
        for k in sorted(cor, key=lambda k: cor[k]["acc"]):
            legs = f"{leg[k]['acc']:.3f}" if k in leg else "  -  "
            print(f"  {k:24s} correct={cor[k]['acc']:.3f}  legal={legs}  n={cor[k]['n']}")

    print("\n[free-run desync by sampling regime]")
    print(f"  {'config':12s} {'surv_med':>8s} {'surv_mean':>9s} {'top desync region':>22s}")
    for name in names:
        _, c = rows(root, name)
        if not c:
            print(f"  {name:12s}  (missing)")
            continue
        med = c.get("survival_bytes_median")
        mean = c.get("survival_bytes_mean")
        hist = c.get("desync_region_hist", {}) or {}
        top = max(hist, key=hist.get) if hist else "-"
        mean_s = f"{mean:.1f}" if isinstance(mean, (int, float)) else "-"
        print(f"  {name:12s} {str(med):>8s} {mean_s:>9s} {top:>22s}")
        print(f"               desync_region_hist={hist}")


if __name__ == "__main__":
    main()
