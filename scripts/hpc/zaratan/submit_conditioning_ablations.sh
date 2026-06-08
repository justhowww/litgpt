#!/bin/bash
# Stage once, train matched reference ablations, then evaluate every checkpoint.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
source "${SCRIPT_DIR}/stage_corpus.sh"

SOURCE_CORPUS=${SOURCE_CORPUS:-"/home/${USER}/SHELL.metzler-prj/OpenVid-1M/h264"}
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
ABLATION_ROOT=${ABLATION_ROOT:-"${STAGED_CORPUS}/runs/conditioning-ablations"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
STEPS=${STEPS:-1000}

stage_corpus "${SOURCE_CORPUS}" "${STAGED_CORPUS}"
mkdir -p "${ABLATION_ROOT}"

sbatch_args=(--parsable)
if [[ -n "${SBATCH_ACCOUNT}" ]]; then
    sbatch_args+=(--account="${SBATCH_ACCOUNT}")
fi

eval_job_ids=()
for mode in normal no_ref shuffled_ref; do
    out_dir="${ABLATION_ROOT}/${mode}"
    mkdir -p "${out_dir}"
    train_job_id=$(sbatch "${sbatch_args[@]}" \
        --export="ALL,REPO_ROOT=${REPO_ROOT},MANIFEST=${STAGED_CORPUS}/manifest.jsonl,OUT_DIR=${out_dir},REFERENCE_MODE=${mode},STEPS=${STEPS}" \
        "${SCRIPT_DIR}/train_stage1.sbatch")
    echo "Submitted ${mode} training job ${train_job_id}"

    eval_job_id=$(sbatch "${sbatch_args[@]}" \
        --dependency="afterok:${train_job_id}" \
        --export="ALL,REPO_ROOT=${REPO_ROOT},MANIFEST=${STAGED_CORPUS}/manifest.jsonl,CHECKPOINT_DIR=${out_dir}/final,RESULTS_PATH=${out_dir}/conditioning_eval.json" \
        "${SCRIPT_DIR}/evaluate_conditioning.sbatch")
    echo "Scheduled ${mode} evaluation job ${eval_job_id}"
    eval_job_ids+=("${eval_job_id}")
done

eval_dependency=$(IFS=:; echo "${eval_job_ids[*]}")
summary_job_id=$(sbatch "${sbatch_args[@]}" \
    --dependency="afterok:${eval_dependency}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},ABLATION_ROOT=${ABLATION_ROOT}" \
    "${SCRIPT_DIR}/summarize_conditioning.sbatch")
echo "Scheduled aggregate summary job ${summary_job_id}"
echo "Staged corpus retained so all ablations use the same immutable data snapshot."
