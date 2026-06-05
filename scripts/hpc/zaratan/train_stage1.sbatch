#!/bin/bash
#SBATCH --job-name=byte-stage1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

OUT_DIR=${OUT_DIR:-"${PROJECT_SCRATCH}/runs/byte-stage1"}
BLOCK_SIZE=${BLOCK_SIZE:-8192}
STEPS=${STEPS:-10000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}

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
    --eval-interval 250
    --eval-iters 20
    --save-interval 500
    --compile
    --resume
)

srun "${cmd[@]}"

