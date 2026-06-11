#!/bin/bash
# Submit the matched Stage 2.5 PSM run with 10x positive EOS loss.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export USE_EOS=1
export EOS_LOSS_WEIGHT=${EOS_LOSS_WEIGHT:-10}
export FIM_FORMAT=psm
export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-eos-weight10}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-eos-weight10"}
export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-5000}
export RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-50}
export RECONSTRUCTION_ORACLE_LENGTH=1
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

exec bash "${BASE_DIR}/submit.sh"
