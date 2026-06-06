#!/bin/bash
# Run on a Zaratan login node: stage the corpus, then submit the GPU smoke job.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/stage_corpus.sh"

SOURCE_CORPUS=${SOURCE_CORPUS:-"/home/${USER}/SHELL.metzler-prj/OpenVid-1M/h264"}
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
OUT_DIR=${OUT_DIR:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage1-smoke"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}

# stage_corpus "${SOURCE_CORPUS}" "${STAGED_CORPUS}"
mkdir -p "${OUT_DIR}"

sbatch_args=(--parsable)
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
    sbatch_args+=(--account="${SBATCH_ACCOUNT}")
fi

job_id=$(
    sbatch "${sbatch_args[@]}" \
        --export="ALL,REPO_ROOT=${REPO_ROOT},MANIFEST=${STAGED_CORPUS}/manifest.jsonl,OUT_DIR=${OUT_DIR}" \
        "${SCRIPT_DIR}/smoke_stage1.sbatch"
)

echo "Submitted smoke job ${job_id}"
echo "Staged corpus retained for the full training run"
