#!/bin/bash
# Submit the Stage 2 PSM (+EOS) smoke run.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
FIM_FORMAT=${FIM_FORMAT:-psm}
USE_EOS=${USE_EOS:-1}
if [[ "${USE_EOS}" == "1" ]]; then
    EOS_TAG="-eos"
else
    EOS_TAG=""
fi
MODEL_NAME=${MODEL_NAME:-"byte-stage2-fim-psm${EOS_TAG}-smoke"}
OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm${EOS_TAG}-smoke"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}

MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
export REPO_ROOT MANIFEST NAL_INDEX OUT_DIR FIM_FORMAT MODEL_NAME USE_EOS

job_id=$(
    sbatch \
        --parsable \
        --export=ALL \
        --account="${SBATCH_ACCOUNT}" \
        "${SCRIPT_DIR}/smoke.sbatch"
)
echo "Submitted Stage 2 PSM FIM smoke job ${job_id} (USE_EOS=${USE_EOS})"
