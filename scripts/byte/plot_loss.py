#!/usr/bin/env python
"""Plot training/val loss curves from TensorBoard event files with a log x-axis.

Pretraining curves improve on a *log* step axis (~constant gain per doubling of
compute), so a linear x-axis makes a still-descending run look "plateaued". This
plots one or more runs' scalar (default ``val_loss_ar``) against step (or samples)
with a logarithmic x-axis, and can overlay runs to compare horizons/corpora
(e.g. 40k vs 300k, QP28 vs AVC-LM).

Reads scalars via tensorboard's EventAccumulator (present in the training env);
imports no torch, so it runs in a light environment.

Examples:
  # single run
  python scripts/byte/plot_loss.py $OUT_DIR --out loss.png
  # compare two runs, x-axis in samples, bits/byte
  python scripts/byte/plot_loss.py $XL_QP28 $XL_AVCLM \
      --labels qp28 avclm --x samples --global-batch 64 --bits --out cmp.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / HPC-safe
import matplotlib.pyplot as plt  # noqa: E402


def find_event_files(run_dir: Path) -> list[Path]:
    """All tfevents files under a run dir (Lightning may write several)."""
    return sorted(run_dir.rglob("*tfevents*"))


def read_scalar(run_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    """Merge a scalar series for ``tag`` across all event files under run_dir.

    Returns (steps, values) sorted by step with duplicate steps de-duplicated
    (last write wins, matching how a resumed run overwrites earlier points).
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        DEFAULT_SIZE_GUIDANCE,
        EventAccumulator,
    )

    guidance = dict(DEFAULT_SIZE_GUIDANCE)
    guidance["scalars"] = 0  # 0 = load every scalar point, no downsampling
    merged: dict[int, float] = {}
    seen_tags: set[str] = set()
    for event_file in find_event_files(run_dir):
        acc = EventAccumulator(str(event_file), size_guidance=guidance)
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        seen_tags.update(tags)
        if tag not in tags:
            continue
        for event in acc.Scalars(tag):
            merged[event.step] = event.value
    if not merged:
        raise KeyError(
            f"tag {tag!r} not found under {run_dir}. Available: {sorted(seen_tags)}"
        )
    steps = sorted(merged)
    return steps, [merged[s] for s in steps]


def ema(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average; alpha=0 disables (returns input)."""
    if alpha <= 0:
        return values
    out: list[float] = []
    prev = values[0]
    for v in values:
        prev = alpha * prev + (1 - alpha) * v
        out.append(prev)
    return out


def plot_runs(
    series: list[tuple[str, list[int], list[float]]],
    *,
    tag: str,
    x_kind: str,
    global_batch: int,
    bits: bool,
    xlog: bool,
    smooth: float,
    out_path: Path,
) -> None:
    """series: list of (label, steps, values). Renders one line per run."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ln2 = math.log(2.0)
    for label, steps, values in series:
        xs = [s * global_batch for s in steps] if x_kind == "samples" else list(steps)
        ys = [v / ln2 for v in values] if bits else list(values)
        # log x cannot show step 0; drop non-positive x.
        pts = [(x, y) for x, y in zip(xs, ys) if x > 0]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ys = ema(list(ys), smooth)
        ax.plot(xs, ys, marker=".", ms=3, lw=1.4, label=label)

    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel("samples" if x_kind == "samples" else "optimizer step")
    ax.set_ylabel(f"{tag} ({'bits/byte' if bits else 'nats'})")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    if len(series) > 1:
        ax.legend()
    ax.set_title(f"{tag} vs {'samples' if x_kind == 'samples' else 'step'}"
                 f"{' (log-x)' if xlog else ''}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", type=Path, nargs="+", help="Run dir(s) containing tfevents (searched recursively).")
    p.add_argument("--tag", default="val_loss_ar", help="Scalar tag to plot (fallback: val_loss).")
    p.add_argument("--labels", nargs="*", default=None, help="Legend labels, one per run dir.")
    p.add_argument("--x", dest="x_kind", choices=["step", "samples"], default="step")
    p.add_argument("--global-batch", type=int, default=64, help="For --x samples: step*global_batch.")
    p.add_argument("--bits", action="store_true", help="Convert nats -> bits/byte (divide by ln 2).")
    p.add_argument("--no-xlog", dest="xlog", action="store_false", help="Linear x-axis instead of log.")
    p.add_argument("--smooth", type=float, default=0.0, help="EMA factor in [0,1); 0 = off.")
    p.add_argument("--out", type=Path, default=Path("loss_curve.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.labels or [d.name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise SystemExit("--labels must match the number of run dirs")

    series: list[tuple[str, list[int], list[float]]] = []
    for run_dir, label in zip(args.run_dirs, labels):
        try:
            steps, values = read_scalar(run_dir, args.tag)
        except KeyError:
            steps, values = read_scalar(run_dir, "val_loss")  # fallback
            print(f"[{label}] tag {args.tag!r} absent -> using 'val_loss'")
        series.append((label, steps, values))
        print(f"[{label}] {len(steps)} points, final={values[-1]:.4f} nats @ step {steps[-1]}")

    plot_runs(
        series,
        tag=args.tag,
        x_kind=args.x_kind,
        global_batch=args.global_batch,
        bits=args.bits,
        xlog=args.xlog,
        smooth=args.smooth,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
