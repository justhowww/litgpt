#!/usr/bin/env bash
# Controlled twin of:
#   byte-bscv-small-megabyte-patch8-ctx131k-fim-p100-changing-
#   fullseq-eos-20v-sweep38k-4gpu
#
# The data, architecture, changing-hole policy, batch size, schedule, and four-GPU
# launcher stay matched.  The only objective change is:
#   L = L_full + 1.0 * L_fim_span + 1.0 * L_balanced_eos
# where L_fim_span is normalized over missing raw bytes and excludes EOS.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STAGED_CORPUS=${STAGED_CORPUS:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data}
export NAL_INDEX=${NAL_INDEX:-${STAGED_CORPUS}/nal_index.sqlite}

export N_LAYER=${N_LAYER:-6}
export N_EMBD=${N_EMBD:-384}
export N_HEAD=${N_HEAD:-6}
export BLOCK_SIZE=${BLOCK_SIZE:-16384}
export BYTE_PATCH_SIZE=${BYTE_PATCH_SIZE:-8}
export MEGABYTE_LOCAL_LAYERS=${MEGABYTE_LOCAL_LAYERS:-4}
export MEGABYTE_LOCAL_EMBD=${MEGABYTE_LOCAL_EMBD:-512}
export MEGABYTE_LOCAL_HEADS=${MEGABYTE_LOCAL_HEADS:-8}

export MAX_ROWS=${MAX_ROWS:-21}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export VAL_FRACTION=${VAL_FRACTION:-0.05}
export P_FIM=${P_FIM:-1.0}
export FIXED_FIM_HOLES=${FIXED_FIM_HOLES:-0}
export FIXED_FIM_HOLES_PER_WINDOW=${FIXED_FIM_HOLES_PER_WINDOW:-0}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export FIM_LOSS_SCOPE=${FIM_LOSS_SCOPE:-full}
export FIM_SPAN_LOSS_WEIGHT=${FIM_SPAN_LOSS_WEIGHT:-1.0}
export USE_EOS=${USE_EOS:-1}
export EOS_LOSS_WEIGHT=${EOS_LOSS_WEIGHT:-1.0}
export EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-1.0}

export STEPS=${STEPS:-38000}
export WARMUP_STEPS=${WARMUP_STEPS:-760}
export EVAL_INTERVAL=${EVAL_INTERVAL:-2000}
export SAVE_INTERVAL=${SAVE_INTERVAL:-4750}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}
export MODEL_TAG=${MODEL_TAG:-byte-bscv-small-megabyte-patch8-ctx131k-fim-p100-changing-fullseq-spanw1-eosaux1-20v-sweep38k-4gpu}
export OUT_DIR=${OUT_DIR:-${STAGED_CORPUS}/runs/${MODEL_TAG}}

exec "${SCRIPT_DIR}/submit.sh"
