#!/bin/bash
# Exact production-shape speed/OOM pilot for disabling activation checkpointing.
# It uses the complete JPEG-LM corpus so length bucketing sees the production GOP
# distribution, but writes no 65 GB checkpoint.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}

export BYTE_PATCH_SIZE=256
export RAW_CONTEXT_BYTES=131072

export P_FIM=0.5
export FIM_FORMAT=psm
export FIM_LOSS_SCOPE=full
export FIM_SPAN_LOSS_WEIGHT=1
export EOS_AUX_LOSS_WEIGHT=1
export FIM_MIN_GAP=64
export FIM_MAX_GAP=1400
export SLICE_HEADER_GUARD_BYTES=0
export WINDOW_UNIT=gop

export GLOBAL_BATCH_SIZE=64
export MICRO_BATCH_SIZE=8
export ENABLE_LENGTH_BUCKETING=1
export LENGTH_BUCKET_POOL_SIZE=8192
export ACTIVATION_CHECKPOINTING=0
export COMPILE=1

export MAX_ROWS=0
export STEPS=${STEPS:-200}
export WARMUP_STEPS=${WARMUP_STEPS:-20}
export EVAL_INTERVAL=1000000
export EVAL_ITERS=2
export SAVE_INTERVAL=1000000
export LATEST_SAVE_INTERVAL=1000000
export SAVE_FINAL=0

export MODEL_TAG=${MODEL_TAG:-byte-jpeglm-7b-patch256-no-activation-checkpointing-pilot}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
export STAGED_CORPUS

exec bash "${SCRIPT_DIR}/submit.sh"
