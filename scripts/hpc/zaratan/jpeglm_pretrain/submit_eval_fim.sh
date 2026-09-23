#!/bin/bash
# Submit train/validation x unmasked/masked JPEG-LM FIM evaluation jobs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}
RUN_NAME=${RUN_NAME:-"byte-jpeglm-7b-patch256-fim-p050-changing-fullseq-spanw1-eosaux1-1m"}
RUN_DIR=${RUN_DIR:-"${STAGED_CORPUS}/runs/${RUN_NAME}"}
EVAL_GPU_TYPE=${EVAL_GPU_TYPE:-h100}
EVAL_TASKS=${EVAL_TASKS:-0-3}

export REPO_ROOT STAGED_CORPUS RUN_NAME RUN_DIR
mkdir -p "${RUN_DIR}/logs"

job_id=$(
    sbatch \
        --parsable \
        --export=ALL \
        --account="${SBATCH_ACCOUNT}" \
        --gpus="${EVAL_GPU_TYPE}:1" \
        --array="${EVAL_TASKS}" \
        --output="${RUN_DIR}/logs/jpeglm-fim-eval-%A_%a.out" \
        --error="${RUN_DIR}/logs/jpeglm-fim-eval-%A_%a.err" \
        "${SCRIPT_DIR}/eval_fim_h100.sbatch"
)

echo "Submitted JPEG-LM FIM evaluation array ${job_id} (tasks ${EVAL_TASKS})"
echo "Run dir: ${RUN_DIR}"
echo "GPU type: ${EVAL_GPU_TYPE}"
echo "Evaluation tag: ${EVAL_TAG:-corrupt-gen-frame}"
