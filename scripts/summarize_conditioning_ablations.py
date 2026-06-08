"""Combine matched conditioning-ablation reports into one comparison matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRAINING_MODES = ("normal", "no_ref", "shuffled_ref")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {}
    for training_mode in TRAINING_MODES:
        path = args.ablation_root / training_mode / "conditioning_eval.json"
        reports[training_mode] = json.loads(path.read_text(encoding="utf-8"))

    validation_samples = reports["normal"]["validation_samples"]
    for training_mode, report in reports.items():
        if report["validation_samples"] != validation_samples:
            raise ValueError(f"{training_mode} used a different validation sample set")

    loss_matrix = {
        training_mode: {
            eval_mode: metrics["loss_nats_per_byte"]
            for eval_mode, metrics in report["conditions"].items()
        }
        for training_mode, report in reports.items()
    }
    native_eval_mode = {
        "normal": "normal",
        "no_ref": "no_ref",
        "shuffled_ref": "shuffled_ref",
    }
    summary = {
        "ablation_root": str(args.ablation_root),
        "num_validation_samples": len(validation_samples),
        "loss_matrix_nats_per_byte": loss_matrix,
        "native_condition_loss": {
            training_mode: loss_matrix[training_mode][eval_mode]
            for training_mode, eval_mode in native_eval_mode.items()
        },
        "normal_checkpoint_conditioning_deltas": reports["normal"]["loss_delta_vs_normal"],
        "unigram": reports["normal"]["unigram"],
    }
    output = args.output or args.ablation_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
