#!/bin/bash
# Submit the official JPEG-LM 7B patch-256 mixed AR/FIM pretraining run.
#
# This keeps the broad changing-hole distribution used by the JPEG-LM corpus,
# while applying the span and EOS auxiliary losses validated by the BSCV FIM
# experiments. Expensive free-run decoding remains an offline checkpoint eval.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}

export BYTE_PATCH_SIZE=${BYTE_PATCH_SIZE:-256}
export RAW_CONTEXT_BYTES=${RAW_CONTEXT_BYTES:-131072}

export P_FIM=${P_FIM:-0.5}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export FIM_LOSS_SCOPE=${FIM_LOSS_SCOPE:-full}
export FIM_SPAN_LOSS_WEIGHT=${FIM_SPAN_LOSS_WEIGHT:-1}
export EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-1}
export FIM_MIN_GAP=${FIM_MIN_GAP:-64}
export FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
export SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-0}
export WINDOW_UNIT=${WINDOW_UNIT:-gop}

export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
export NUM_WORKERS=${NUM_WORKERS:-8}
export ENABLE_LENGTH_BUCKETING=${ENABLE_LENGTH_BUCKETING:-1}
export LENGTH_BUCKET_POOL_SIZE=${LENGTH_BUCKET_POOL_SIZE:-8192}
export ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-1}
export COMPILE=${COMPILE:-1}

export STEPS=${STEPS:-1000000}
export WARMUP_STEPS=${WARMUP_STEPS:-2000}
export EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
export LATEST_SAVE_INTERVAL=${LATEST_SAVE_INTERVAL:-25000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-100000}

export MODEL_TAG=${MODEL_TAG:-byte-jpeglm-7b-patch256-fim-p050-changing-fullseq-spanw1-eosaux1-1m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
export STAGED_CORPUS

exec bash "${SCRIPT_DIR}/submit.sh"
