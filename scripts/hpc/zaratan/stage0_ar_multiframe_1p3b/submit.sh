#!/bin/bash
# Run this wrapper on a Zaratan login node. It submits the 1.3 B Stage 0
# multi-frame AR (H0-1p3b) training against the compute-node-visible staged corpus.
# The corpus is already persistent on scratch, so cleanup is off by default.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
source "${SCRIPT_DIR}/../stage_corpus.sh"

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage0-ar-multiframe-1p3b"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
JOB_SCRIPT=${JOB_SCRIPT:-"${SCRIPT_DIR}/train_h100.sbatch"}

mkdir -p "${OUT_DIR}"

MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL)
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
    sbatch_args+=(--account="${SBATCH_ACCOUNT}")
fi
# Steer off nodes with a degraded scratch/Lustre mount, e.g. EXCLUDE_NODES=gpu-a6-2.
EXCLUDE_NODES=${EXCLUDE_NODES:-}
if [[ -n "${EXCLUDE_NODES}" ]]; then
    sbatch_args+=(--exclude="${EXCLUDE_NODES}")
fi

job_id=$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")
echo "Submitted Stage 0 multi-frame AR-1p3b training job ${job_id}"
echo "Out dir: ${OUT_DIR}"
