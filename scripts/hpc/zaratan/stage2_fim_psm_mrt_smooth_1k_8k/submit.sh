#!/bin/bash
# Launch smooth-risk online MRT from the matched supervised checkpoint.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLIPPED_MRT_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm_mrt_1k_8k" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export MRT_RISK_MODE=smooth_mse
export MRT_MSE_TAU=${MRT_MSE_TAU:-0.002}
export MRT_DECODE_FAILURE_WEIGHT=${MRT_DECODE_FAILURE_WEIGHT:-1.0}
export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-mrt-smooth-1k-8k}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-mrt-smooth-1k-8k"}

exec bash "${CLIPPED_MRT_DIR}/submit.sh"
