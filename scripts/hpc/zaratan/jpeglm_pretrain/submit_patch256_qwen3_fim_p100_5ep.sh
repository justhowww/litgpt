#!/bin/bash
# Submit the five-epoch JPEG-LM Qwen3-style patch-256 FIM pretraining run.
#
# The current corpus contains about 12.70M usable GOP windows. At global batch
# size 64, 1,000,000 optimizer steps are approximately 5.04 corpus passes.
# Each access still samples one view of one GOP; p_fim=1 only makes that view a
# freshly sampled FIM example instead of mixing AR and FIM examples.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}

export MODEL_ARCHITECTURE=${MODEL_ARCHITECTURE:-qwen3}
export BYTE_PATCH_SIZE=${BYTE_PATCH_SIZE:-256}
export RAW_CONTEXT_BYTES=${RAW_CONTEXT_BYTES:-131072}

export P_FIM=${P_FIM:-1.0}
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
export NUM_WORKERS=${NUM_WORKERS:-12}
# Allocate enough CPU headroom for the main rank in addition to its workers.
export TRAIN_CPUS_PER_TASK=${TRAIN_CPUS_PER_TASK:-16}
export ENABLE_LENGTH_BUCKETING=${ENABLE_LENGTH_BUCKETING:-1}
export LENGTH_BUCKET_POOL_SIZE=${LENGTH_BUCKET_POOL_SIZE:-8192}
export ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-1}
export COMPILE=${COMPILE:-1}

# About 5.04 passes over the current 12.70M-window training corpus.
export STEPS=${STEPS:-1000000}
export WARMUP_STEPS=${WARMUP_STEPS:-2000}
export EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
export EVAL_ITERS=${EVAL_ITERS:-20}
export LATEST_SAVE_INTERVAL=${LATEST_SAVE_INTERVAL:-25000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-200000}

# Reuse the four training H100s after the final checkpoint is written. Each GPU
# runs one train/validation x masked/unmasked evaluation quadrant.
export INLINE_FINAL_EVAL=${INLINE_FINAL_EVAL:-1}
export INLINE_EVAL_TAG=${INLINE_EVAL_TAG:-corrupt-gen-frame-final}
export INLINE_EVAL_NUM_CLIPS=${INLINE_EVAL_NUM_CLIPS:-20}
export INLINE_EVAL_NUM_VISUALIZATIONS=${INLINE_EVAL_NUM_VISUALIZATIONS:-8}

export MODEL_TAG=${MODEL_TAG:-byte-jpeglm-7b-qwen3-patch256-fim-p100-changing-fullseq-spanw1-eosaux1-5ep}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
export STAGED_CORPUS

exec bash "${SCRIPT_DIR}/submit.sh"
