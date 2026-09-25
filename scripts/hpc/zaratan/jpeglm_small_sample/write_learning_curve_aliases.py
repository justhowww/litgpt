#!/usr/bin/env python3
"""Add clearly named TensorBoard curves for one completed small-sample run.

The original tags and event files remain intact. This script is invoked only by
the isolated small-sample launcher; no shared trainer or older run is changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


ALIASES = {
    "loss": "learning/train/optimized_objective",
    "training/full_ce": "learning/train/full_sequence_ce_nats",
    "training/fim_span_ce": "learning/train/missing_byte_ce_nats",
    "training/eos_aux_loss": "learning/train/eos_aux_loss",
    "val_loss": "learning/val/full_sequence_ce_nats",
    # Validation's REGION_BRIDGE includes EOS; training's span CE does not.
    "val_loss_fim": "learning/val/bridge_ce_including_eos_nats",
    "val_eos_probability_fim": "learning/val/eos_probability",
    "val_eos_rank_fim": "learning/val/eos_rank",
}


def write_aliases(run_dir: Path) -> None:
    event_dir = run_dir / "logs" / "tensorboard" / "optimizer_steps"
    if not event_dir.is_dir():
        raise FileNotFoundError(f"Training TensorBoard directory is missing: {event_dir}")
    events = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags()["scalars"])
    missing = set(ALIASES) - tags
    if missing:
        raise RuntimeError(f"Cannot label incomplete training curves; missing tags: {sorted(missing)}")
    if any(alias in tags for alias in ALIASES.values()):
        raise RuntimeError(f"Small-sample learning-curve aliases already exist in {event_dir}")

    # Keep these curves in the same TensorBoard run as the originals. The new
    # names are additive; no old event file or metric is rewritten.
    writer = EventFileWriter(str(event_dir))
    try:
        for source, alias in ALIASES.items():
            for point in events.Scalars(source):
                writer.add_event(
                    Event(
                        wall_time=point.wall_time,
                        step=point.step,
                        summary=Summary(value=[Summary.Value(tag=alias, simple_value=point.value)]),
                    )
                )
        writer.flush()
    finally:
        writer.close()
    print(f"Added {len(ALIASES)} clearly named curves to {event_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    write_aliases(args.run_dir)


if __name__ == "__main__":
    main()
