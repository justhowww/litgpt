#!/bin/bash
# Submit the explicit Prefix-Suffix-Middle FIM ablation on an A100.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export FIM_FORMAT=psm
export MODEL_NAME=byte-stage2-fim-psm
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm"}

exec bash "${SCRIPT_DIR}/submit.sh"
