#!/bin/bash
# Smoke-test weighted EOS training before launching the full experiment.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export USE_EOS=1
export EOS_LOSS_WEIGHT=${EOS_LOSS_WEIGHT:-10}
export FIM_FORMAT=psm
export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-eos-weight10-smoke}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-eos-weight10-smoke"}

exec bash "${BASE_DIR}/submit_smoke.sh"
