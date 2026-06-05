#!/bin/bash
#SBATCH --job-name=byte-stage1-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

OUT_DIR=${OUT_DIR:-"${PROJECT_SCRATCH}/runs/byte-stage1-smoke"}
BLOCK_SIZE=${BLOCK_SIZE:-8192}
STEPS=${STEPS:-10}
NUM_WORKERS=${NUM_WORKERS:-4}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

srun python scripts/train_byte_stage1.py \
    "${MANIFEST}" \
    --out-dir "${OUT_DIR}" \
    --block-size "${BLOCK_SIZE}" \
    --steps "${STEPS}" \
    --global-batch-size 8 \
    --micro-batch-size 1 \
    --precision bf16-mixed \
    --num-workers "${NUM_WORKERS}" \
    --eval-interval 5 \
    --eval-iters 2 \
    --save-interval 5

