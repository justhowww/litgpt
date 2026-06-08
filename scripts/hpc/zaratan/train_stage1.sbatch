#!/bin/bash
#SBATCH --job-name=byte-stage1
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Slurm executes a spool copy of this file, so BASH_SOURCE does not point into
# the repository. Resolve cluster helpers from the submitted repository root.
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

OUT_DIR=${OUT_DIR:-"${PROJECT_SCRATCH}/runs/byte-stage1"}
# Preliminary scan: 16K kept full context for 93.42% of samples with no target exclusion.
BLOCK_SIZE=${BLOCK_SIZE:-16384}
STEPS=${STEPS:-10000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
REFERENCE_MODE=${REFERENCE_MODE:-normal}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
    python scripts/train_byte_stage1.py
    "${MANIFEST}"
    --out-dir "${OUT_DIR}"
    --block-size "${BLOCK_SIZE}"
    --steps "${STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --precision bf16-mixed
    --num-workers "${NUM_WORKERS}"
    --reference-mode "${REFERENCE_MODE}"
    --eval-interval 250
    --eval-iters 20
    --save-interval 500
    --compile
    --resume
)

srun "${cmd[@]}"
