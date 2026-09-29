#!/usr/bin/env python3
"""Free-run FIM repair on the fixed small-sample holes (learned EOS, no mask).

Runs ``eval_fim_avclm.py`` once per stratum of the ``feasible_35_holes_v1``
protocol with exactly the hole-selection arguments ``eval_tf.build_samples``
uses, so the repaired holes are the same ones scored by ``eval_tf`` and
``eval_syntax_prior``. The sample set is verified against the shared pinned
copy before any generation. ``--split train`` places holes in training videos
(known content, unseen cut); ``--split val`` in held-out videos.

Output: RUN/eval_free_run/<checkpoint>/feasible_35_holes_v1/<split>/<type>_<len>B/
(summary.csv and per-sample results from eval_fim_avclm).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.byte.eval import eval_fim_avclm as FIM  # noqa: E402
from scripts.hpc.zaratan.jpeglm_small_sample import eval_tf  # noqa: E402
from scripts.hpc.zaratan.jpeglm_small_sample.submit import (  # noqa: E402
    load_config,
    load_evaluation_config,
)


def stratum_argv(
    values: dict[str, str],
    evaluation: dict,
    split: str,
    frame_type: str,
    length: int,
    checkpoint: Path,
    out_dir: Path,
) -> list[str]:
    """CLI mirror of eval_tf.build_samples' Namespace plus generation settings."""
    return [
        values["MANIFEST"],
        "--nal-index-path", values["NAL_INDEX"],
        "--checkpoint-dirs", str(checkpoint),
        "--out-dir", str(out_dir),
        "--model-dtype", "bf16",
        "--train-split-file", str(Path(values["OUT_DIR"]) / "train_split.json"),
        "--eval-split", split,
        "--max-manifest-rows", values["MAX_ROWS"],
        "--max-window-bytes", str(int(values["RAW_CONTEXT_BYTES"]) - 1),
        "--window-min-frames", values["WINDOW_MIN_FRAMES"],
        "--window-unit", values["WINDOW_UNIT"],
        "--val-fraction", values["VAL_FRACTION"],
        "--split-by-video",
        "--seed", str(evaluation["seed"]),
        "--fim-format", values["FIM_FORMAT"],
        "--fim-loss-scope", values["FIM_LOSS_SCOPE"],
        "--use-eos",
        "--fim-min-gap", values["FIM_MIN_GAP"],
        "--fim-max-gap", values["FIM_MAX_GAP"],
        "--slice-header-guard-bytes", values["SLICE_HEADER_GUARD_BYTES"],
        "--hole-placement", "corrupt_gen_frame",
        "--hole-set", "sampled",
        "--corr-pos", str(evaluation["corruption_position"]),
        "--corr-len-bytes-list", str(length),
        "--corr-samples-per-length", str(evaluation["samples_per_length"]),
        "--corr-eligibility-bytes", str(length),
        "--corr-header-guard-bytes", str(evaluation["corruption_header_guard_bytes"]),
        "--corr-frame-type", frame_type,
        "--num-clips", str(evaluation["samples_per_length"]),
        # Generation: the model's own learned EOS, greedy, unconstrained.
        "--stop-modes", "learned_eos",
        "--slice-layout", "frame",
        "--temperature", "0",
        "--num-visualizations", "0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Frozen small-sample YAML in the run directory")
    parser.add_argument("--checkpoint", default="final", help="final or step-XXXXXXXX")
    parser.add_argument("--split", choices=("val", "train"), default="train")
    args = parser.parse_args()

    values = load_config(args.config)
    evaluation = load_evaluation_config(args.config)
    run_dir = Path(values["OUT_DIR"])
    values["SMALL_SAMPLE_CONFIG"] = str(run_dir / "small_sample_config.yaml")
    record = json.loads((run_dir / "small_sample_config.json").read_text(encoding="utf-8"))
    if record != {"training_environment": values, "evaluation": evaluation}:
        raise RuntimeError("YAML differs from frozen training configuration")
    checkpoint = run_dir / args.checkpoint
    if not (checkpoint / "lit_model.pth").is_file():
        raise FileNotFoundError(checkpoint / "lit_model.pth")
    out_root = run_dir / "eval_free_run" / args.checkpoint / eval_tf.EVAL_PROTOCOL_ID / args.split
    if out_root.exists():
        raise RuntimeError(f"Output already exists; refusing overwrite: {out_root}")

    # Same holes as eval_tf / eval_syntax_prior, checked against the pinned set.
    samples = eval_tf.build_samples(values, evaluation, args.split)
    eval_tf.verify_shared_sample_set(samples, evaluation, args.split)

    strata, omitted = eval_tf.evaluation_strata(evaluation)
    for frame_type, length in strata:
        out_dir = out_root / f"{frame_type}_{length}B"
        argv = stratum_argv(values, evaluation, args.split, frame_type, length, checkpoint, out_dir)
        print(f"== {args.split} {frame_type} {length}B -> {out_dir}", flush=True)
        saved = sys.argv
        sys.argv = ["eval_fim_avclm.py", *argv]
        try:
            FIM.main()
        finally:
            sys.argv = saved
    (out_root / "protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "split": args.split,
                "protocol": eval_tf.EVAL_PROTOCOL_ID,
                "strata": [{"frame_type": t, "length": n} for t, n in strata],
                "omitted": omitted,
                "generation": "learned_eos, greedy, unconstrained (no syntax mask)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Done: {out_root}", flush=True)


if __name__ == "__main__":
    main()
