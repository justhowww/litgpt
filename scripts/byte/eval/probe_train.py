#!/usr/bin/env python3
"""Train linear probes on extracted hidden states and report parser-state readout.

Usage:
  python scripts/byte/eval/probe_train.py --npz G4=g4.npz G4-L=g4l.npz random=g4_random.npz \
      --out probe_report.json

Tasks (labels from litgpt.byte.probe_labels, state after each byte):
  element     which syntax element the next bit belongs to (all positions)
  element_mb  same, restricted to macroblock-layer data
  bits_into   bits already consumed of that element (macroblock data)
  mb_x, mb_y  macroblock column / row (macroblock data)

A single linear layer per (model, layer, task), trained full-batch on 80% of source
videos and tested on the rest. Reported per region (context / suffix / prefix /
middle) against the majority-class baseline of the training labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litgpt.byte.probe_labels import MB_CLASSES  # noqa: E402

REGIONS = ("context", "suffix", "prefix", "middle")


def task_rows(d) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """task -> (row mask, labels)."""
    element = d["element"]
    in_mb = np.isin(element, list(MB_CLASSES))
    return {
        "element": (np.ones_like(element, dtype=bool), element),
        "element_mb": (in_mb, element),
        "bits_into": (in_mb & (d["bits_into"] >= 0), d["bits_into"]),
        "mb_x": (in_mb & (d["mb_x"] >= 0), d["mb_x"]),
        "mb_y": (in_mb & (d["mb_y"] >= 0), d["mb_y"]),
    }


def fit_probe(x_tr, y_tr, x_te, n_cls, device, epochs=300, lr=1e-2, wd=1e-4):
    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    x_te = torch.tensor(x_te, dtype=torch.float32, device=device)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-5
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    y = torch.tensor(y_tr, dtype=torch.long, device=device)
    probe = torch.nn.Linear(x_tr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(probe(x_tr), y).backward()
        opt.step()
    with torch.no_grad():
        return probe(x_te).argmax(1).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", nargs="+", required=True, help="name=path.npz ...")
    p.add_argument("--test-fold", type=int, default=0, help="sources with id %% 5 == fold are test")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report: dict = {}
    for spec in args.npz:
        name, path = spec.split("=", 1)
        d = np.load(path)
        test = (d["source"] % 5) == args.test_fold
        report[name] = {}
        for task, (rows, labels) in task_rows(d).items():
            tr, te = rows & ~test, rows & test
            if tr.sum() < 100 or te.sum() < 50:
                continue
            n_cls = int(labels[rows].max()) + 1
            majority = np.bincount(labels[tr], minlength=n_cls).argmax()
            entry = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "layers": {}}
            entry["baseline"] = {
                r: float((labels[te & (d["region"] == i)] == majority).mean())
                for i, r in enumerate(REGIONS) if (te & (d["region"] == i)).any()
            }
            for layer in d["layers"]:
                pred = fit_probe(d[f"X_{layer}"][tr], labels[tr], d[f"X_{layer}"][te], n_cls, device)
                correct = pred == labels[te]
                region_te = d["region"][te]
                entry["layers"][int(layer)] = {
                    "all": float(correct.mean()),
                    **{r: float(correct[region_te == i].mean()) for i, r in enumerate(REGIONS) if (region_te == i).any()},
                }
            report[name][task] = entry
            best = max(entry["layers"], key=lambda l: entry["layers"][l]["all"])
            row = " ".join(f"{r}={entry['layers'][best].get(r, float('nan')):.2f}(base {entry['baseline'].get(r, float('nan')):.2f})"
                           for r in REGIONS)
            print(f"{name:10s} {task:11s} best layer {best:2d}: {row}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
