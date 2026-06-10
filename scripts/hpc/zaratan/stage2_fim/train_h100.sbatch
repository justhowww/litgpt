#!/bin/bash
# Stage 2 H100 run. Hyperparameters match train_a100.sbatch.

#SBATCH --job-name=byte-fim-h100
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"
: "${NAL_INDEX:?Set NAL_INDEX to the persistent SQLite NAL index}"

OUT_DIR=${OUT_DIR:-"$(dirname "${MANIFEST}")/runs/byte-stage2-fim"}
BLOCK_SIZE=${BLOCK_SIZE:-16384}
STEPS=${STEPS:-100000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
REFERENCE_MODE=${REFERENCE_MODE:-normal}
P_FIM=${P_FIM:-0.5}
FIM_MIN_GAP=${FIM_MIN_GAP:-64}
FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
    python scripts/train_byte_stage1.py
    "${MANIFEST}"
    --model-name byte-stage2-fim
    --nal-index-path "${NAL_INDEX}"
    --out-dir "${OUT_DIR}"
    --block-size "${BLOCK_SIZE}"
    --steps "${STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --precision bf16-mixed
    --logger-name "${LOGGER_NAME}"
    --num-workers "${NUM_WORKERS}"
    --reference-mode "${REFERENCE_MODE}"
    --p-fim "${P_FIM}"
    --fim-min-gap "${FIM_MIN_GAP}"
    --fim-max-gap "${FIM_MAX_GAP}"
    --slice-header-guard-bytes "${SLICE_HEADER_GUARD_BYTES}"
    --eval-interval 250
    --eval-iters 20
    --reconstruction-eval-interval 0
    --save-interval 500
    --compile
    --resume
)

# FIM decoder evaluation is intentionally disabled until the repair-specific
# probe is implemented. The existing reconstruction probe measures AR slices.
flock -n "${OUT_DIR}/.training.lock" srun "${cmd[@]}" || {
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
}
