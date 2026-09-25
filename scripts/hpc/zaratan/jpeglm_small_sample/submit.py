#!/usr/bin/env python3
"""Submit a small-sample JPEG-LM run from one immutable YAML config.

This is intentionally separate from the existing experiment launchers. It
translates only the settings listed below into the existing submit.sh path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

import yaml


FIELDS = {
    "run": {
        "staged_corpus": "STAGED_CORPUS",
        "model_tag": "MODEL_TAG",
        "out_dir": "OUT_DIR",
    },
    "model": {
        "architecture": "MODEL_ARCHITECTURE",
        "n_layer": "N_LAYER",
        "n_embd": "N_EMBD",
        "n_head": "N_HEAD",
        "byte_patch_size": "BYTE_PATCH_SIZE",
        "raw_context_bytes": "RAW_CONTEXT_BYTES",
        "megabyte_local_layers": "MEGABYTE_LOCAL_LAYERS",
        "megabyte_local_embd": "MEGABYTE_LOCAL_EMBD",
        "megabyte_local_heads": "MEGABYTE_LOCAL_HEADS",
    },
    "data": {
        "max_rows": "MAX_ROWS",
        "val_fraction": "VAL_FRACTION",
        "num_workers": "NUM_WORKERS",
        "window_min_frames": "WINDOW_MIN_FRAMES",
        "window_unit": "WINDOW_UNIT",
        "enable_length_bucketing": "ENABLE_LENGTH_BUCKETING",
        "length_bucket_pool_size": "LENGTH_BUCKET_POOL_SIZE",
    },
    "fim": {
        "p_fim": "P_FIM",
        "format": "FIM_FORMAT",
        "loss_scope": "FIM_LOSS_SCOPE",
        "span_loss_weight": "FIM_SPAN_LOSS_WEIGHT",
        "eos_aux_loss_weight": "EOS_AUX_LOSS_WEIGHT",
        "min_gap": "FIM_MIN_GAP",
        "max_gap": "FIM_MAX_GAP",
        "slice_header_guard_bytes": "SLICE_HEADER_GUARD_BYTES",
    },
    "training": {
        "steps": "STEPS",
        "warmup_steps": "WARMUP_STEPS",
        "global_batch_size": "GLOBAL_BATCH_SIZE",
        "micro_batch_size": "MICRO_BATCH_SIZE",
        "learning_rate": "LEARNING_RATE",
        "min_learning_rate": "MIN_LEARNING_RATE",
        "eval_interval": "EVAL_INTERVAL",
        "eval_iters": "EVAL_ITERS",
        "save_interval": "SAVE_INTERVAL",
        "latest_save_interval": "LATEST_SAVE_INTERVAL",
        "save_final": "SAVE_FINAL",
        "activation_checkpointing": "ACTIVATION_CHECKPOINTING",
        "compile": "COMPILE",
        "memory_profile_step": "SMALL_SAMPLE_MEMORY_PROFILE_STEP",
    },
    "preflight": {
        "min_manifest_rows": "MIN_MANIFEST_ROWS",
        "min_p_fim_eligibility": "MIN_P_FIM_ELIGIBILITY",
        "allow_low_fim_eligibility": "ALLOW_LOW_FIM_ELIGIBILITY",
    },
    "slurm": {
        "account": "SBATCH_ACCOUNT",
        "mem": "SBATCH_MEM",
        "cpus_per_task": "TRAIN_CPUS_PER_TASK",
    },
}

# Keep the required keys of previously frozen YAMLs unchanged. Only the new
# IDR-balanced run opts into this additional training setting.
OPTIONAL_FIELDS = {"fim": {"idr_sampling_probability": "FIM_IDR_SAMPLING_PROBABILITY"}}

BOOLEAN_KEYS = {
    "ENABLE_LENGTH_BUCKETING",
    "SAVE_FINAL",
    "ACTIVATION_CHECKPOINTING",
    "COMPILE",
    "ALLOW_LOW_FIM_ELIGIBILITY",
}
POSITIVE_INT_KEYS = {
    "N_LAYER", "N_EMBD", "N_HEAD", "BYTE_PATCH_SIZE", "RAW_CONTEXT_BYTES",
    "MEGABYTE_LOCAL_LAYERS", "MEGABYTE_LOCAL_EMBD", "MEGABYTE_LOCAL_HEADS",
    "MAX_ROWS", "WINDOW_MIN_FRAMES", "LENGTH_BUCKET_POOL_SIZE", "STEPS",
    "GLOBAL_BATCH_SIZE", "MICRO_BATCH_SIZE", "EVAL_INTERVAL", "EVAL_ITERS",
    "SAVE_INTERVAL", "LATEST_SAVE_INTERVAL", "FIM_MAX_GAP",
    "MIN_MANIFEST_ROWS", "TRAIN_CPUS_PER_TASK", "SMALL_SAMPLE_MEMORY_PROFILE_STEP",
}
NONNEGATIVE_INT_KEYS = {"NUM_WORKERS", "WARMUP_STEPS", "FIM_MIN_GAP", "SLICE_HEADER_GUARD_BYTES"}
NONNEGATIVE_FLOAT_KEYS = {"FIM_SPAN_LOSS_WEIGHT", "EOS_AUX_LOSS_WEIGHT"}
PROBABILITY_KEYS = {"VAL_FRACTION", "P_FIM", "MIN_P_FIM_ELIGIBILITY"}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a YAML mapping with schema_version: 1")
    unknown_groups = set(config) - {"schema_version", "evaluation", *FIELDS}
    if unknown_groups:
        raise ValueError(f"Unknown config sections: {sorted(unknown_groups)}")
    values = {}
    for group, names in FIELDS.items():
        section = config.get(group)
        if not isinstance(section, dict):
            raise ValueError(f"Missing or invalid section: {group}")
        allowed = set(names) | set(OPTIONAL_FIELDS.get(group, {}))
        if set(names) - set(section) or set(section) - allowed:
            raise ValueError(
                f"{group}: missing {sorted(set(names) - set(section))}; "
                f"unknown {sorted(set(section) - allowed)}"
            )
        present_optional = {
            key: env_name for key, env_name in OPTIONAL_FIELDS.get(group, {}).items()
            if key in section
        }
        for key, env_name in {**names, **present_optional}.items():
            value = section[key]
            if env_name in BOOLEAN_KEYS:
                if not isinstance(value, bool):
                    raise ValueError(f"{group}.{key} must be true or false")
                values[env_name] = "1" if value else "0"
            elif isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
                raise ValueError(f"{group}.{key} must be a scalar value")
            else:
                values[env_name] = str(value)

    for key in POSITIVE_INT_KEYS | NONNEGATIVE_INT_KEYS:
        try:
            value = int(values[key])
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if value < (0 if key in NONNEGATIVE_INT_KEYS else 1):
            raise ValueError(f"{key} is out of range")
    for key in NONNEGATIVE_FLOAT_KEYS | PROBABILITY_KEYS | {"LEARNING_RATE", "MIN_LEARNING_RATE"}:
        try:
            value = float(values[key])
        except ValueError as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
        if key in PROBABILITY_KEYS and value > 1:
            raise ValueError(f"{key} must be at most 1")
        if key in {"LEARNING_RATE", "MIN_LEARNING_RATE"} and value == 0:
            raise ValueError(f"{key} must be positive")
    if "FIM_IDR_SAMPLING_PROBABILITY" in values:
        probability = float(values["FIM_IDR_SAMPLING_PROBABILITY"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("fim.idr_sampling_probability must be in [0, 1]")
        if float(values["P_FIM"]) <= 0:
            raise ValueError("IDR-balanced sampling requires fim.p_fim > 0")
    if values["MODEL_ARCHITECTURE"] not in {"pythia", "qwen3"}:
        raise ValueError("model.architecture must be pythia or qwen3")
    if values["WINDOW_UNIT"] != "gop":
        raise ValueError("Small-sample comparison requires data.window_unit: gop")
    if values["FIM_FORMAT"] != "psm" or values["FIM_LOSS_SCOPE"] != "full":
        raise ValueError("Small-sample comparison requires fim.format: psm and loss_scope: full")
    if not re.fullmatch(r"[1-9][0-9]*[GM]", values["SBATCH_MEM"]):
        raise ValueError("slurm.mem must look like 320G")
    if int(values["RAW_CONTEXT_BYTES"]) % int(values["BYTE_PATCH_SIZE"]):
        raise ValueError("raw_context_bytes must be divisible by byte_patch_size")
    if int(values["N_EMBD"]) % int(values["N_HEAD"]) or int(values["N_EMBD"]) % int(values["BYTE_PATCH_SIZE"]):
        raise ValueError("n_embd must be divisible by n_head and byte_patch_size")
    if int(values["MEGABYTE_LOCAL_EMBD"]) % int(values["MEGABYTE_LOCAL_HEADS"]):
        raise ValueError("megabyte_local_embd must be divisible by megabyte_local_heads")
    if values["MODEL_ARCHITECTURE"] == "qwen3" and (
        int(values["N_HEAD"]) % 4 or int(values["MEGABYTE_LOCAL_HEADS"]) % 4
    ):
        raise ValueError("Qwen3 global and local head counts must be divisible by 4")
    if int(values["GLOBAL_BATCH_SIZE"]) % (2 * int(values["MICRO_BATCH_SIZE"])):
        raise ValueError("global_batch_size must be divisible by 2 GPUs × micro_batch_size")
    if int(values["FIM_MIN_GAP"]) > int(values["FIM_MAX_GAP"]):
        raise ValueError("fim.min_gap cannot exceed fim.max_gap")
    if int(values["WARMUP_STEPS"]) >= int(values["STEPS"]):
        raise ValueError("warmup_steps must be less than steps")
    if int(values["SMALL_SAMPLE_MEMORY_PROFILE_STEP"]) > int(values["STEPS"]):
        raise ValueError("training.memory_profile_step must not exceed steps")
    if not values["MODEL_TAG"] or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", values["MODEL_TAG"]):
        raise ValueError("run.model_tag must be a nonempty directory-safe name")
    if not Path(values["STAGED_CORPUS"]).is_absolute() or not Path(values["OUT_DIR"]).is_absolute():
        raise ValueError("run.staged_corpus and run.out_dir must be absolute")
    if float(values["MIN_LEARNING_RATE"]) > float(values["LEARNING_RATE"]):
        raise ValueError("min_learning_rate cannot exceed learning_rate")
    corpus = Path(values["STAGED_CORPUS"])
    values.update({
        "MANIFEST": str(corpus / "manifest.jsonl"),
        "NAL_INDEX": str(corpus / "nal_index.sqlite"),
        "BLOCK_SIZE": str(int(values["RAW_CONTEXT_BYTES"]) // int(values["BYTE_PATCH_SIZE"])),
        "DEVICES": "2",
        "NUM_NODES": "1",
        "INLINE_FINAL_EVAL": "0",
        "LOGGER_NAME": "tensorboard",
        "JOB_SCRIPT": str(Path(__file__).resolve().parent / "train_eval_a100_2gpu.sbatch"),
    })
    return values


def load_evaluation_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    evaluation = config.get("evaluation") if isinstance(config, dict) else None
    required = {
        "seed", "corruption_lengths", "samples_per_length",
        "corruption_position", "corruption_header_guard_bytes", "frame_types",
        "sample_set_dir", "num_heatmaps",
    }
    if not isinstance(evaluation, dict) or set(evaluation) != required:
        raise ValueError(
            f"evaluation: missing {sorted(required - set(evaluation or {}))}; "
            f"unknown {sorted(set(evaluation or {}) - required)}"
        )
    lengths = evaluation["corruption_lengths"]
    if not isinstance(lengths, list) or not lengths or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("evaluation.corruption_lengths must be positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("evaluation.corruption_lengths must not contain duplicates")
    if evaluation["frame_types"] != ["idr", "p"]:
        raise ValueError("evaluation.frame_types must be [idr, p]")
    for key in ("seed", "samples_per_length", "corruption_header_guard_bytes", "num_heatmaps"):
        value = evaluation[key]
        if type(value) is not int or value < (0 if key != "samples_per_length" else 1):
            raise ValueError(f"evaluation.{key} must be a nonnegative integer")
    position = evaluation["corruption_position"]
    if isinstance(position, bool) or not isinstance(position, (int, float)) or not 0 <= position <= 1:
        raise ValueError("evaluation.corruption_position must be in [0, 1]")
    if not isinstance(evaluation["sample_set_dir"], str) or not Path(evaluation["sample_set_dir"]).is_absolute():
        raise ValueError("evaluation.sample_set_dir must be an absolute path")
    return evaluation


def reserve_run_dir(out_dir: Path, values: dict, evaluation: dict, source_yaml: Path) -> None:
    """Never silently resume an old run or change a small-run configuration."""
    record = out_dir / "small_sample_config.json"
    if out_dir.exists() and not record.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Refusing existing run without a small-sample config: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {"training_environment": values, "evaluation": evaluation}
    if record.exists():
        saved = json.loads(record.read_text(encoding="utf-8"))
        if saved != resolved:
            raise ValueError(f"Configuration differs from existing run: {record}")
    else:
        with record.open("x", encoding="utf-8") as file:
            json.dump(resolved, file, indent=2, sort_keys=True)
            file.write("\n")
    frozen_yaml = out_dir / "small_sample_config.yaml"
    source_text = source_yaml.read_text(encoding="utf-8")
    if frozen_yaml.exists():
        if frozen_yaml.read_text(encoding="utf-8") != source_text:
            raise ValueError(f"YAML differs from frozen run configuration: {frozen_yaml}")
    else:
        with frozen_yaml.open("x", encoding="utf-8") as file:
            file.write(source_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Small-sample YAML configuration")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print settings without writing or submitting")
    args = parser.parse_args()
    source_yaml = args.config.resolve()
    values = load_config(source_yaml)
    evaluation = load_evaluation_config(source_yaml)
    values["SMALL_SAMPLE_CONFIG"] = str(Path(values["OUT_DIR"]) / "small_sample_config.yaml")
    submit = Path(__file__).resolve().parents[1] / "jpeglm_pretrain" / "submit.sh"
    if args.dry_run:
        print(json.dumps({"training_environment": values, "evaluation": evaluation}, indent=2, sort_keys=True))
        print(f"Would run: bash {submit}")
        return
    reserve_run_dir(Path(values["OUT_DIR"]), values, evaluation, source_yaml)
    environment = os.environ.copy()
    for key in (
        "AFTER_JOBID", "EXCLUDE_NODES", "DEPENDENCY_TYPE", "TRAINING_LOCK_WAIT_SEC",
        "FIM_IDR_SAMPLING_PROBABILITY",
    ):
        environment.pop(key, None)
    environment.update(values)
    subprocess.run(["bash", str(submit)], env=environment, check=True)


if __name__ == "__main__":
    main()
