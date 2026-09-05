#!/bin/bash
# Speed Pilot 5: test the maximum microbatch compatible with global batch=64
# on four GPUs. Each rank processes all 16 local samples in one optimizer step.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}

export MAX_ROWS=${MAX_ROWS:-256}
export STEPS=${STEPS:-20}
export WARMUP_STEPS=${WARMUP_STEPS:-2}
export EVAL_INTERVAL=${EVAL_INTERVAL:-10}
export EVAL_ITERS=${EVAL_ITERS:-8}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=16
export WINDOW_UNIT=${WINDOW_UNIT:-gop}
export SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-0}
export COMPILE=${COMPILE:-1}
export ACTIVATION_CHECKPOINTING=1

# This is a disposable speed/OOM test: write neither intermediate nor final
# 65 GB checkpoints. TensorBoard logs and the deterministic split are retained.
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
export LATEST_SAVE_INTERVAL=${LATEST_SAVE_INTERVAL:-1000000}
export SAVE_FINAL=0

export MIN_MANIFEST_ROWS=${MIN_MANIFEST_ROWS:-1}
export ALLOW_LOW_FIM_ELIGIBILITY=${ALLOW_LOW_FIM_ELIGIBILITY:-1}
export MODEL_TAG=${MODEL_TAG:-byte-jpeglm-7b-megabyte-patch8-gop-guard0-speed-microbatch16}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}

exec bash "${SCRIPT_DIR}/submit.sh"
