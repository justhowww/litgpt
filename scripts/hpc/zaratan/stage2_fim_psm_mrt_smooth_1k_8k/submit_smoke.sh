#!/bin/bash
# Submit a short smooth-risk MRT integration run.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLIPPED_MRT_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm_mrt_1k_8k" && pwd)

export MRT_RISK_MODE=smooth_mse
export MRT_MSE_TAU=${MRT_MSE_TAU:-0.002}
export MRT_DECODE_FAILURE_WEIGHT=${MRT_DECODE_FAILURE_WEIGHT:-1.0}
export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-mrt-smooth-smoke}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS:-/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data}/runs/byte-stage2-fim-psm-mrt-smooth-smoke"}

exec bash "${CLIPPED_MRT_DIR}/submit_smoke.sh"
