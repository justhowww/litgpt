#!/bin/bash
# Run this wrapper on a Zaratan login node. It stages data from SHELL storage,
# submits training against the compute-node-visible copy, and schedules cleanup.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/stage_corpus.sh"

SOURCE_CORPUS=${SOURCE_CORPUS:-"/home/${USER}/SHELL.metzler-prj/OpenVid-1M/h264"}
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
OUT_DIR=${OUT_DIR:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage1"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
JOB_SCRIPT=${JOB_SCRIPT:-"${SCRIPT_DIR}/train_stage1.sbatch"}
# CLEANUP=0 is the concise public override. Keep CLEANUP_AFTER_SUCCESS as the
# backward-compatible name used by existing scripts.
CLEANUP_AFTER_SUCCESS=${CLEANUP_AFTER_SUCCESS:-${CLEANUP:-1}}

# stage_corpus "${SOURCE_CORPUS}" "${STAGED_CORPUS}"
mkdir -p "${OUT_DIR}"

# Export values into the caller environment, then let sbatch copy that
# environment directly. Passing an explicit --export=ALL,VAR=... list can make
# Slurm invoke --get-user-env; a failure in that mechanism requeues and holds
# the job with "user env retrieval failed".
MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL)
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
    sbatch_args+=(--account="${SBATCH_ACCOUNT}")
fi

job_id=$(
    sbatch "${sbatch_args[@]}" \
        "${JOB_SCRIPT}"
)
echo "Submitted training job ${job_id}"

if [[ "${CLEANUP_AFTER_SUCCESS}" == "1" ]]; then
    cleanup_job_id=$(
        sbatch "${sbatch_args[@]}" \
            --dependency="afterok:${job_id}" \
            "${SCRIPT_DIR}/cleanup_stage1.sbatch"
    )
    echo "Scheduled cleanup job ${cleanup_job_id} after successful training"
else
    echo "Staged corpus retained because CLEANUP_AFTER_SUCCESS=${CLEANUP_AFTER_SUCCESS}"
fi
