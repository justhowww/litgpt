#!/usr/bin/env python3
"""Audit current H.264 mask exclusions on fixed, teacher-forced FIM holes.

This does not alter the model distribution or train the model. A position is
scored only when the ground-truth middle and suffix are accepted by the same
incremental automaton used for constrained decoding. Mask exclusions include
dataset-profile rules and do not exhaust FFmpeg's validity checks. EOS and FIM
markers are not bytes; their probability/top-choice rates are reported separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from litgpt.byte import h264_mask as HM  # noqa: E402
from litgpt.byte.data import BYTE_VOCAB_SIZE, SEQ_EOS_ID  # noqa: E402
from litgpt.byte.megabyte_inference import megabyte_teacher_forced_sample  # noqa: E402
from litgpt.byte.reconstruction import _unwrap_model  # noqa: E402
from scripts.byte.eval import eval_fim_avclm as FIM  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import load_model  # noqa: E402
from scripts.hpc.zaratan.jpeglm_small_sample import eval_tf  # noqa: E402
from scripts.hpc.zaratan.jpeglm_small_sample.submit import (  # noqa: E402
    load_config,
    load_evaluation_config,
)


def ground_truth_masks(sample: FIM.WindowFimSample) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    """Walk the original byte order; never parse the rearranged PSM prompt."""
    state = FIM._seed_parser_state(sample, slice_max_mbs=None)
    masks: list[list[bool]] = []
    strict: list[bool] = []
    for position, byte in enumerate(sample.target_bytes):
        strict_before = state.strict_mask_calls
        allowed = HM.get_valid_byte_mask(state)
        if not allowed[byte]:
            return None, None, f"GT byte rejected at middle offset {position}: {byte:#04x}"
        masks.append(allowed)
        strict.append(state.strict_mask_calls > strict_before)
        HM.advance(state, byte)
        if state.automaton_unknown:
            return None, None, f"automaton unknown after middle offset {position}"
    if not HM.can_append_bytes(state, sample.bytes_after_hole, require_complete=True):
        return None, None, "GT suffix fails parser reconnection"
    return torch.tensor(masks, dtype=torch.bool), torch.tensor(strict, dtype=torch.bool), None


@torch.inference_mode()
def teacher_byte_logits(model: torch.nn.Module, sample: FIM.WindowFimSample, device: torch.device) -> torch.Tensor:
    """Use precisely the small-sample evaluator's MEGABYTE teacher-forcing layout."""
    raw = _unwrap_model(model)
    patch_size = int(raw.config.byte_patch_size)
    if patch_size > 1:
        patched = megabyte_teacher_forced_sample(
            sample.teacher_input_ids,
            sample.teacher_labels,
            sample.teacher_region_ids,
            sample.teacher_offset_ids,
            patch_size,
        )
        inputs = patched["input_ids"]
        labels = patched["labels"].to(device)
        regions = patched["region_ids"]
        offsets = patched["offset_ids"]
        extra = {"patch_targets": labels}
    else:
        inputs = sample.teacher_input_ids.unsqueeze(0)
        labels = sample.teacher_labels.unsqueeze(0).to(device)
        regions = sample.teacher_region_ids.unsqueeze(0)
        offsets = sample.teacher_offset_ids.unsqueeze(0)
        extra = {}
    if inputs.size(1) > raw.max_seq_length:
        raise RuntimeError(f"FIM sample exceeds model context: {inputs.size(1)} > {raw.max_seq_length}")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = raw(
            inputs.to(device),
            region_ids=regions.to(device),
            offset_ids=offsets.to(device),
            **extra,
        )
    selected_mask = FIM._teacher_forced_span_mask(labels, sample.target_length)
    if selected_mask is None:
        raise RuntimeError("Missing supervised FIM target positions")
    expected = list(sample.target_bytes) + [SEQ_EOS_ID]
    if labels[selected_mask].tolist() != expected:
        raise RuntimeError("FIM target labels differ from deleted bytes + EOS")
    return logits[selected_mask][:-1].float().cpu()


def score_byte_legality(logits: torch.Tensor, masks: torch.Tensor, strict: torch.Tensor) -> dict[str, Any]:
    """Match unmasked inference's available actions: 256 bytes plus EOS."""
    if logits.ndim != 2 or logits.size(0) != masks.size(0) or masks.shape != (logits.size(0), BYTE_VOCAB_SIZE):
        raise ValueError("logits and byte masks are misaligned")
    if strict.shape != (logits.size(0),):
        raise ValueError("strictness flags are misaligned")
    if not masks.any(dim=1).all():
        raise ValueError("empty legality mask")
    generation_logits = torch.cat(
        (logits[:, :BYTE_VOCAB_SIZE], logits[:, SEQ_EOS_ID : SEQ_EOS_ID + 1]),
        dim=-1,
    )
    probabilities = generation_logits.softmax(dim=-1)
    illegal_mass = (probabilities[:, :BYTE_VOCAB_SIZE] * ~masks).sum(dim=-1)
    byte_top = logits[:, :BYTE_VOCAB_SIZE].argmax(dim=-1)
    byte_top_illegal = ~masks.gather(1, byte_top[:, None]).squeeze(1)
    full_top = generation_logits.argmax(dim=-1)
    full_top_is_byte = full_top < BYTE_VOCAB_SIZE
    full_top_illegal = full_top_is_byte & ~masks.gather(1, full_top.clamp(max=BYTE_VOCAB_SIZE - 1)[:, None]).squeeze(1)
    return {
        "positions": logits.size(0),
        "strict_positions": int(strict.sum()),
        "constrained_positions": int((masks.sum(dim=1) < BYTE_VOCAB_SIZE).sum()),
        "illegal_mass_sum": float(illegal_mass.sum()),
        "byte_top_illegal_count": int(byte_top_illegal.sum()),
        "full_top_illegal_byte_count": int(full_top_illegal.sum()),
        "full_top_eos_count": int((full_top == BYTE_VOCAB_SIZE).sum()),
        "strict_illegal_mass_sum": float(illegal_mass[strict].sum()),
        "strict_byte_top_illegal_count": int(byte_top_illegal[strict].sum()),
        "strict_full_top_illegal_byte_count": int(full_top_illegal[strict].sum()),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_keys = (
        "positions", "strict_positions", "constrained_positions",
        "byte_top_illegal_count", "full_top_illegal_byte_count",
        "full_top_eos_count",
        "strict_byte_top_illegal_count", "strict_full_top_illegal_byte_count",
    )
    sums = {key: sum(int(row[key]) for row in rows) for key in count_keys}
    positions = sums["positions"]
    strict = sums["strict_positions"]
    return {
        "samples": len(rows),
        **sums,
        "strict_coverage": strict / positions if positions else None,
        "mean_illegal_byte_probability": sum(row["illegal_mass_sum"] for row in rows) / positions if positions else None,
        "byte_top_illegal_rate": sums["byte_top_illegal_count"] / positions if positions else None,
        "full_top_illegal_byte_rate": sums["full_top_illegal_byte_count"] / positions if positions else None,
        "full_top_eos_rate": sums["full_top_eos_count"] / positions if positions else None,
        "mean_illegal_byte_probability_strict": sum(row["strict_illegal_mass_sum"] for row in rows) / strict if strict else None,
        "byte_top_illegal_rate_strict": sums["strict_byte_top_illegal_count"] / strict if strict else None,
        "full_top_illegal_byte_rate_strict": sums["strict_full_top_illegal_byte_count"] / strict if strict else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Frozen small-sample YAML in the run directory")
    parser.add_argument("--checkpoint", default="final", help="final or step-XXXXXXXX")
    parser.add_argument("--split", choices=("val", "train"), default="val")
    args = parser.parse_args()
    values = load_config(args.config)
    evaluation = load_evaluation_config(args.config)
    run_dir = Path(values["OUT_DIR"])
    values["SMALL_SAMPLE_CONFIG"] = str(run_dir / "small_sample_config.yaml")
    record = json.loads((run_dir / "small_sample_config.json").read_text(encoding="utf-8"))
    if record != {"training_environment": values, "evaluation": evaluation}:
        raise RuntimeError("YAML differs from frozen training configuration")
    if args.checkpoint != "final" and not (
        args.checkpoint.startswith("step-") and len(args.checkpoint) == 13 and args.checkpoint[5:].isdigit()
    ):
        raise ValueError("--checkpoint must be final or step-XXXXXXXX")
    checkpoint = run_dir / args.checkpoint
    if not (checkpoint / "lit_model.pth").is_file():
        raise FileNotFoundError(checkpoint / "lit_model.pth")
    out = run_dir / "eval_syntax_prior" / args.checkpoint / eval_tf.EVAL_PROTOCOL_ID / args.split
    if out.exists():
        raise RuntimeError(f"Output already exists; refusing overwrite: {out}")

    samples = eval_tf.build_samples(values, evaluation, args.split)
    eval_tf.verify_shared_sample_set(samples, evaluation, args.split)
    prepared = []
    failures = []
    for index, sample in enumerate(samples):
        masks, strict, error = ground_truth_masks(sample)
        if error:
            failures.append({"sample_id": index, **eval_tf.sample_identity(sample, args.split), "reason": error})
        else:
            prepared.append((index, sample, masks, strict))
    if not prepared:
        raise RuntimeError(f"Parser accepted none of the {len(samples)} GT holes: {failures[:3]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.eval()
    rows = []
    for index, sample, masks, strict in prepared:
        logits = teacher_byte_logits(model, sample, device)
        row = {
            "sample_id": index,
            **eval_tf.sample_identity(sample, args.split),
            **score_byte_legality(logits, masks, strict),
        }
        rows.append(row)
        print(
            f"[{args.split}] {index + 1}/{len(samples)} {sample.corruption_frame_type} "
            f"{sample.gap}B strict={row['strict_positions']}/{row['positions']}",
            flush=True,
        )
    summary = {
        "checkpoint": str(checkpoint),
        "eval_split": args.split,
        "protocol": eval_tf.EVAL_PROTOCOL_ID,
        "interpretation": (
            "Teacher-forced ground-truth FIM prefixes; not model-generated prefixes. "
            "Illegal-byte mass uses the unmasked inference distribution over "
            "bytes 0..255 plus EOS, excluding FIM control tokens. "
            "byte_top_illegal_rate takes argmax over byte IDs 0..255 only; "
            "full_top_illegal_byte_rate uses the bytes-plus-EOS argmax. "
            "EOS is reported separately and is not classified as a byte. "
            "False mask entries are exclusions under the current H.264 automaton "
            "and dataset profile, not an exhaustive definition of decoder invalidity."
        ),
        "requested_samples": len(samples),
        "parser_rejected_samples": failures,
        "overall": aggregate(rows),
        "by_frame_type": {
            kind: aggregate([row for row in rows if row["frame_type"] == kind])
            for kind in sorted({row["frame_type"] for row in rows})
        },
    }
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "samples.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True), flush=True)
    print(f"Report: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
