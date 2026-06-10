#!/bin/bash
# Submit a short integration run for the explicit PSM FIM representation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export FIM_FORMAT=psm
export MODEL_NAME=byte-stage2-fim-psm-smoke
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-smoke"}

exec bash "${SCRIPT_DIR}/submit_smoke.sh"
