#!/bin/bash
# Login-node submitter for the phase-based AVC-LM reproduction runs. The per-phase
# wrappers (submit_phase0_overfit.sh, ...) set env knobs and exec this. Corpus defaults
# to the per-MB AVC-LM data; the manifest is assumed persistent on scratch.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-avclm"}
MODEL_TAG=${MODEL_TAG:-byte-phase}
OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
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
echo "Submitted phase training job ${job_id}"
echo "Out dir: ${OUT_DIR}"
