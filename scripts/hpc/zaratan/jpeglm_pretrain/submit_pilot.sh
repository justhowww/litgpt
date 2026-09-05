#!/bin/bash
# Exact-shape OOM/throughput pilot. It uses only a small corpus prefix and a
# separate output directory, but preserves the 7B/patch-8/16K/FSDP memory shape.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}

export MAX_ROWS=${MAX_ROWS:-256}
export STEPS=${STEPS:-20}
export WARMUP_STEPS=${WARMUP_STEPS:-2}
export EVAL_INTERVAL=${EVAL_INTERVAL:-10}
export EVAL_ITERS=${EVAL_ITERS:-2}
export SAVE_INTERVAL=${SAVE_INTERVAL:-20}
export COMPILE=${COMPILE:-0}
export MIN_MANIFEST_ROWS=${MIN_MANIFEST_ROWS:-1}
# This pilot tests the model's memory/throughput shape, not the final data
# distribution. Print low FIM coverage as a warning here; the full launcher
# still rejects it.
export ALLOW_LOW_FIM_ELIGIBILITY=${ALLOW_LOW_FIM_ELIGIBILITY:-1}
export MODEL_TAG=${MODEL_TAG:-byte-jpeglm-7b-megabyte-patch8-pilot}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}

exec bash "${SCRIPT_DIR}/submit.sh"
