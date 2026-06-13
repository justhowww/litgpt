#!/bin/bash
# Launch the MRT-primary oracle-length diagnostic on matched 1-8 KiB FIM gaps.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STEPS=${STEPS:-1000}
export P_FIM=1
export FIM_FORMAT=psm
export FIM_MIN_GAP=1024
export FIM_MAX_GAP=8192

# Disable positive EOS supervision and remove EOS from generation actions.
# PSM vocab sizes with and without EOS both pad to 264, so the checkpoint
# tensors remain shape-compatible without a special vocabulary override.
export USE_EOS=0
export EOS_LOSS_WEIGHT=1
export EOS_AUX_LOSS_WEIGHT=0

# Decoder risk is primary on every step; CE remains a small syntax regularizer.
export CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-0.05}
# Restrict CE to byte logits so the unused EOS/control logits receive no CE
# gradient. MRT remains responsible for decoder-level candidate preference.
export CE_BYTE_ONLY=1
export MRT_INTERVAL=1
export MRT_START_STEP=0
export MRT_NUM_CANDIDATES=${MRT_NUM_CANDIDATES:-8}
export MRT_CONTEXT_POOL_SIZE=${MRT_CONTEXT_POOL_SIZE:-64}
export MRT_MAX_TARGET_BYTES=8192
export MRT_ORACLE_LENGTH=1
export MRT_WEIGHT=${MRT_WEIGHT:-1.0}
export MRT_RISK_MODE=${MRT_RISK_MODE:-smooth_mse}
export MRT_MSE_TAU=${MRT_MSE_TAU:-0.002}
export MRT_DECODE_FAILURE_WEIGHT=${MRT_DECODE_FAILURE_WEIGHT:-1.0}
export MRT_DECODE_WORKERS=${MRT_DECODE_WORKERS:-8}

export RECONSTRUCTION_TASK=fim
export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-250}
export RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-50}
export RECONSTRUCTION_MAX_TARGET_BYTES=8192
# With USE_EOS=0, the standard generation path already uses oracle length.
export RECONSTRUCTION_ORACLE_LENGTH=0
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-mrt-primary-oracle-1k-8k}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-mrt-primary-oracle-1k-8k"}
export INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-balanced-eos-1k-8k/step-00001000"}

if [[ ! -r "${INITIAL_CHECKPOINT_DIR}/lit_model.pth" ]]; then
    echo "Initial checkpoint is not readable: ${INITIAL_CHECKPOINT_DIR}/lit_model.pth" >&2
    exit 1
fi

exec bash "${BASE_DIR}/submit.sh"
