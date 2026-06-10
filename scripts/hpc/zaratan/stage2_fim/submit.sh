#!/bin/bash
# Submit Stage 2 mixed AR/FIM pretraining from scratch on Zaratan.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
JOB_SCRIPT=${JOB_SCRIPT:-"${SCRIPT_DIR}/train_a100.sbatch"}

MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
export REPO_ROOT MANIFEST NAL_INDEX OUT_DIR

mkdir -p "${OUT_DIR}"

job_id=$(
    sbatch \
        --parsable \
        --export=ALL \
        --account="${SBATCH_ACCOUNT}" \
        "${JOB_SCRIPT}"
)
echo "Submitted Stage 2 FIM job ${job_id}"
echo "Output directory: ${OUT_DIR}"
