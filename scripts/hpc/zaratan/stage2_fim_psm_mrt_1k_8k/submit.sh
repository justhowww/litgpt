#!/bin/bash
# Launch online MRT from the 1K-step supervised PSM+EOS checkpoint.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STEPS=${STEPS:-10000}
export USE_EOS=1
export EOS_LOSS_WEIGHT=1
export EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-1.0}
export FIM_FORMAT=psm
export FIM_MIN_GAP=${FIM_MIN_GAP:-1024}
export FIM_MAX_GAP=${FIM_MAX_GAP:-8192}
export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-mrt-1k-8k}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-mrt-1k-8k"}
export INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-balanced-eos-1k-8k/step-00001000"}

# Seven optimizer steps use supervised CE+EOS only; every eighth adds one
# online MRT context. The new run starts at step zero after loading pretrained
# weights, so MRT begins on its eighth fine-tuning step.
export MRT_INTERVAL=${MRT_INTERVAL:-8}
export MRT_START_STEP=${MRT_START_STEP:-0}
export MRT_NUM_CANDIDATES=${MRT_NUM_CANDIDATES:-16}
export MRT_CONTEXT_POOL_SIZE=${MRT_CONTEXT_POOL_SIZE:-64}
export MRT_MAX_TARGET_BYTES=${MRT_MAX_TARGET_BYTES:-2048}
export MRT_TEMPERATURE=${MRT_TEMPERATURE:-1.0}
export MRT_CANDIDATE_ALPHA=${MRT_CANDIDATE_ALPHA:-1.0}
export MRT_WEIGHT=${MRT_WEIGHT:-4.0}
export MRT_MSE_WEIGHT=${MRT_MSE_WEIGHT:-1000.0}
export MRT_DECODE_FAILURE_WEIGHT=${MRT_DECODE_FAILURE_WEIGHT:-2.0}
export MRT_MAX_RISK=${MRT_MAX_RISK:-2.0}
export MRT_DECODE_WORKERS=${MRT_DECODE_WORKERS:-8}

export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-1000}
export RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-50}
export RECONSTRUCTION_MAX_TARGET_BYTES=${RECONSTRUCTION_MAX_TARGET_BYTES:-8192}
export RECONSTRUCTION_ORACLE_LENGTH=1
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

if [[ ! -r "${INITIAL_CHECKPOINT_DIR}/lit_model.pth" ]]; then
    echo "Initial MRT checkpoint is not readable: ${INITIAL_CHECKPOINT_DIR}/lit_model.pth" >&2
    exit 1
fi

exec bash "${BASE_DIR}/submit.sh"
