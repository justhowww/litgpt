#!/bin/bash
# Smoke-test balanced EOS loss and the 1-8 KiB FIM configuration.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export USE_EOS=1
export EOS_LOSS_WEIGHT=1
export EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-1.0}
export FIM_FORMAT=psm
export FIM_MIN_GAP=${FIM_MIN_GAP:-1024}
export FIM_MAX_GAP=${FIM_MAX_GAP:-8192}
export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-balanced-eos-1k-8k-smoke}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-balanced-eos-1k-8k-smoke"}
export RECONSTRUCTION_ORACLE_LENGTH=1
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

exec bash "${BASE_DIR}/submit_smoke.sh"
