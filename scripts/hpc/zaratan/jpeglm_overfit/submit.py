#!/usr/bin/env python3
"""Submit a JPEG-LM overfit run (plain byte transformer, one A100) from YAML.

Translates one complete YAML file into the environment read by
``jpeglm_pretrain/train.sh`` and submits ``train_eval_a100_1gpu.sbatch`` through
``jpeglm_pretrain/submit.sh``. The YAML is frozen into the run directory; a later
submission with a different YAML for the same directory is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

FIELDS = {
    "run": {"staged_corpus": "STAGED_CORPUS", "model_tag": "MODEL_TAG", "out_dir": "OUT_DIR"},
    "model": {
        "architecture": "MODEL_ARCHITECTURE",
        "n_layer": "N_LAYER",
        "n_embd": "N_EMBD",
        "n_head": "N_HEAD",
        "raw_context_bytes": "RAW_CONTEXT_BYTES",
    },
    "data": {
        "max_rows": "MAX_ROWS",
        "val_fraction": "VAL_FRACTION",
        "num_workers": "NUM_WORKERS",
        "window_min_frames": "WINDOW_MIN_FRAMES",
    },
    "fim": {
        "p_fim": "P_FIM",
        "fixed_holes_per_window": "FIXED_FIM_HOLES_PER_WINDOW",
        "span_loss_weight": "FIM_SPAN_LOSS_WEIGHT",
        "eos_aux_loss_weight": "EOS_AUX_LOSS_WEIGHT",
        "min_gap": "FIM_MIN_GAP",
        "max_gap": "FIM_MAX_GAP",
    },
    "training": {
        "steps": "STEPS",
        "warmup_steps": "WARMUP_STEPS",
        "global_batch_size": "GLOBAL_BATCH_SIZE",
        "micro_batch_size": "MICRO_BATCH_SIZE",
        "learning_rate": "LEARNING_RATE",
        "min_learning_rate": "MIN_LEARNING_RATE",
        "eval_interval": "EVAL_INTERVAL",
        "save_interval": "SAVE_INTERVAL",
        "activation_checkpointing": "ACTIVATION_CHECKPOINTING",
        "compile": "COMPILE",
    },
    "slurm": {"account": "SBATCH_ACCOUNT", "mem": "SBATCH_MEM", "time": "SBATCH_TIME"},
    "eval": {"ffmpeg_binary": "FFMPEG_BINARY", "ffprobe_binary": "FFPROBE_BINARY"},
}
# Optional keys, with the default used when a YAML omits them (older configs).
OPTIONAL_FIELDS = {
    "fim": {"format": ("FIM_FORMAT", "psm")},
    # Space-separated eval_fim_avclm stop modes: learned_eos and/or parser_reconnect.
    "eval": {"stop_modes": ("EVAL_STOP_MODES", "learned_eos")},
}
BOOLEAN_KEYS = {"ACTIVATION_CHECKPOINTING", "COMPILE"}

# Fixed by design: the simplest setup. Not configurable from YAML.
CONSTANTS = {
    "BYTE_PATCH_SIZE": "1",
    "MEGABYTE_LOCAL_LAYERS": "1",  # unused at patch 1, but train.sh passes them
    "MEGABYTE_LOCAL_EMBD": "64",
    "MEGABYTE_LOCAL_HEADS": "4",
    "WINDOW_UNIT": "gop",
    "FIM_LOSS_SCOPE": "full",
    "SLICE_HEADER_GUARD_BYTES": "0",
    "ENABLE_LENGTH_BUCKETING": "0",
    "LENGTH_BUCKET_POOL_SIZE": "1",
    "EVAL_ITERS": "20",
    "SAVE_FINAL": "1",
    "DEVICES": "1",
    "NUM_NODES": "1",
    "INLINE_FINAL_EVAL": "0",
    "LOGGER_NAME": "tensorboard",
    "MIN_MANIFEST_ROWS": "1",
    "MIN_P_FIM_ELIGIBILITY": "0.5",
    "ALLOW_LOW_FIM_ELIGIBILITY": "0",
    "TRAIN_CPUS_PER_TASK": "16",
}


def load_config(path: Path) -> dict[str, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a YAML mapping with schema_version: 1")
    unknown = set(config) - {"schema_version", *FIELDS}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    values: dict[str, str] = {}
    for group, names in FIELDS.items():
        section = config.get(group)
        optional = OPTIONAL_FIELDS.get(group, {})
        got = set(section or {}) if isinstance(section, dict) else set()
        if not isinstance(section, dict) or set(names) - got or got - set(names) - set(optional):
            raise ValueError(
                f"{group}: missing {sorted(set(names) - got)}; "
                f"unknown {sorted(got - set(names) - set(optional))}"
            )
        for key, (env_name, default) in optional.items():
            values[env_name] = str(section.get(key, default))
        for key, env_name in names.items():
            value = section[key]
            if env_name in BOOLEAN_KEYS:
                if not isinstance(value, bool):
                    raise ValueError(f"{group}.{key} must be true or false")
                values[env_name] = "1" if value else "0"
            elif isinstance(value, (bool, dict, list)) or value is None:
                raise ValueError(f"{group}.{key} must be a scalar")
            else:
                values[env_name] = str(value)

    stop_modes = values["EVAL_STOP_MODES"].split()
    if not stop_modes or set(stop_modes) - {"learned_eos", "parser_reconnect"}:
        raise ValueError("eval.stop_modes must list learned_eos and/or parser_reconnect")
    if values["FIM_FORMAT"] not in {"psm", "spm"}:
        raise ValueError("fim.format must be psm or spm")
    p_fim = float(values["P_FIM"])
    holes = int(values["FIXED_FIM_HOLES_PER_WINDOW"])
    if not 0.0 <= p_fim <= 1.0:
        raise ValueError("fim.p_fim must be in [0, 1]")
    if holes < 0:
        raise ValueError("fim.fixed_holes_per_window must be nonnegative")
    if p_fim == 0 and (
        holes or float(values["FIM_SPAN_LOSS_WEIGHT"]) or float(values["EOS_AUX_LOSS_WEIGHT"])
    ):
        raise ValueError("p_fim: 0 (pure AR) requires zero fixed holes and zero FIM loss weights")
    if int(values["N_EMBD"]) % int(values["N_HEAD"]):
        raise ValueError("n_embd must be divisible by n_head")
    if values["MODEL_ARCHITECTURE"] == "qwen3" and int(values["N_HEAD"]) % 4:
        raise ValueError("Qwen3 n_head must be divisible by 4")
    if int(values["GLOBAL_BATCH_SIZE"]) % int(values["MICRO_BATCH_SIZE"]):
        raise ValueError("global_batch_size must be divisible by micro_batch_size (one GPU)")
    if int(values["WARMUP_STEPS"]) >= int(values["STEPS"]):
        raise ValueError("warmup_steps must be less than steps")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", values["MODEL_TAG"]):
        raise ValueError("run.model_tag must be a directory-safe name")
    for key in ("STAGED_CORPUS", "OUT_DIR", "FFMPEG_BINARY", "FFPROBE_BINARY"):
        if not Path(values[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")

    corpus = Path(values["STAGED_CORPUS"])
    values.update(CONSTANTS)
    values.update({
        "MANIFEST": str(corpus / "manifest.jsonl"),
        "NAL_INDEX": str(corpus / "nal_index.sqlite"),
        "BLOCK_SIZE": values["RAW_CONTEXT_BYTES"],
        "LATEST_SAVE_INTERVAL": values["SAVE_INTERVAL"],
        "JOB_SCRIPT": str(HERE / "train_eval_a100_1gpu.sbatch"),
    })
    return values


def reserve_run_dir(out_dir: Path, source_yaml: Path) -> None:
    frozen = out_dir / "overfit_config.yaml"
    text = source_yaml.read_text(encoding="utf-8")
    if frozen.exists():
        if frozen.read_text(encoding="utf-8") != text:
            raise ValueError(f"YAML differs from the frozen run configuration: {frozen}")
        return
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Refusing a populated run directory without overfit_config.yaml: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.config.resolve()
    values = load_config(source)
    submit = HERE.parent / "jpeglm_pretrain" / "submit.sh"
    if args.dry_run:
        print(json.dumps(values, indent=2, sort_keys=True))
        print(f"Would run: bash {submit}")
        return
    reserve_run_dir(Path(values["OUT_DIR"]), source)
    environment = os.environ.copy()
    for key in ("AFTER_JOBID", "FIM_IDR_SAMPLING_PROBABILITY", "SYNTAX_MASK_DIR", "SYNTAX_LOSS_WEIGHT"):
        environment.pop(key, None)
    environment.update(values)
    subprocess.run(["bash", str(submit)], env=environment, check=True)


if __name__ == "__main__":
    main()
