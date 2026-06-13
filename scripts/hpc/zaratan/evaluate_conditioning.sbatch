#!/bin/bash
#SBATCH --job-name=byte-conditioning-eval
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the staged manifest.jsonl}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a directory containing lit_model.pth}"

RESULTS_PATH=${RESULTS_PATH:-"${CHECKPOINT_DIR}/conditioning_eval.json"}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-1000}
UNIGRAM_TRAIN_SAMPLES=${UNIGRAM_TRAIN_SAMPLES:-100000}
NUM_WORKERS=${NUM_WORKERS:-4}
BLOCK_SIZE=${BLOCK_SIZE:-16384}

srun python scripts/byte/evaluate_conditioning.py \
    "${CHECKPOINT_DIR}" \
    "${MANIFEST}" \
    --output "${RESULTS_PATH}" \
    --max-seq-length "${BLOCK_SIZE}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --unigram-train-samples "${UNIGRAM_TRAIN_SAMPLES}" \
    --num-workers "${NUM_WORKERS}"
