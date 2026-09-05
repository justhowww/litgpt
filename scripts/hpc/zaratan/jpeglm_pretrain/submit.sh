#!/bin/bash
# Submit the resumable JPEG-LM full-corpus AR+FIM pretraining run on 4xH100.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}
MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
MODEL_TAG=${MODEL_TAG:-byte-jpeglm-7b-megabyte-patch8-fim-p050-fullseq-eos-1m}
OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-metzler-prj-cmsc}
JOB_SCRIPT=${JOB_SCRIPT:-"${SCRIPT_DIR}/train_h100_4gpu.sbatch"}

FIM_MIN_GAP=${FIM_MIN_GAP:-64}
FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}
MIN_P_FIM_ELIGIBILITY=${MIN_P_FIM_ELIGIBILITY:-0.5}
MIN_MANIFEST_ROWS=${MIN_MANIFEST_ROWS:-900000}
ALLOW_LOW_FIM_ELIGIBILITY=${ALLOW_LOW_FIM_ELIGIBILITY:-0}
DEPENDENCY_TYPE=${DEPENDENCY_TYPE:-afterok}

preflight=(
    python "${REPO_ROOT}/scripts/byte/reports/check_jpeglm_pretrain_corpus.py"
    "${MANIFEST}"
    --nal-index-path "${NAL_INDEX}"
    --fim-min-gap "${FIM_MIN_GAP}"
    --frame-guard-bytes "${SLICE_HEADER_GUARD_BYTES}"
    --min-p-frame-eligibility "${MIN_P_FIM_ELIGIBILITY}"
    --min-manifest-rows "${MIN_MANIFEST_ROWS}"
)
if [[ "${ALLOW_LOW_FIM_ELIGIBILITY}" == "1" ]]; then
    preflight+=(--allow-low-fim-eligibility)
fi
if [[ -z "${AFTER_JOBID:-}" ]]; then
    "${preflight[@]}"
else
    echo "Deferring corpus preflight until dependency job ${AFTER_JOBID} has completed."
fi

mkdir -p "${OUT_DIR}/logs"
export REPO_ROOT STAGED_CORPUS MANIFEST NAL_INDEX MODEL_TAG OUT_DIR
export FIM_MIN_GAP FIM_MAX_GAP SLICE_HEADER_GUARD_BYTES
export MIN_P_FIM_ELIGIBILITY MIN_MANIFEST_ROWS ALLOW_LOW_FIM_ELIGIBILITY

sbatch_args=(
    --parsable
    --export=ALL
    --account="${SBATCH_ACCOUNT}"
    --output="${OUT_DIR}/logs/%x-%j.out"
    --error="${OUT_DIR}/logs/%x-%j.err"
)
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
    sbatch_args+=(--exclude="${EXCLUDE_NODES}")
fi
if [[ -n "${AFTER_JOBID:-}" ]]; then
    case "${DEPENDENCY_TYPE}" in
        afterok|afterany) ;;
        *) echo "DEPENDENCY_TYPE must be afterok or afterany" >&2; exit 2 ;;
    esac
    sbatch_args+=(--dependency="${DEPENDENCY_TYPE}:${AFTER_JOBID}")
fi

job_id=$(sbatch "${sbatch_args[@]}" "${JOB_SCRIPT}")
echo "Submitted JPEG-LM pretraining job ${job_id}"
echo "Output directory: ${OUT_DIR}"
[[ -n "${AFTER_JOBID:-}" ]] && echo "Dependency: ${DEPENDENCY_TYPE}:${AFTER_JOBID}"
