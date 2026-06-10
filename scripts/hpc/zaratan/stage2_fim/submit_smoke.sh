#!/bin/bash
# Submit the Stage 2 mixed AR/FIM smoke run.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-smoke"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
export REPO_ROOT MANIFEST NAL_INDEX OUT_DIR

job_id=$(
    sbatch \
        --parsable \
        --export=ALL \
        --account="${SBATCH_ACCOUNT}" \
        "${SCRIPT_DIR}/smoke.sbatch"
)
echo "Submitted Stage 2 FIM smoke job ${job_id}"
