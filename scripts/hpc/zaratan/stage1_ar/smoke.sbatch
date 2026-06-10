#!/bin/bash
#SBATCH --job-name=byte-ar-smoke
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Slurm executes a spool copy of this file, so BASH_SOURCE does not point into
# the repository. Resolve cluster helpers from the submitted repository root.
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

OUT_DIR=${OUT_DIR:-"${PROJECT_SCRATCH}/runs/byte-stage1-smoke"}
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
# Preliminary scan: 16K kept full context for 93.42% of samples with no target exclusion.
BLOCK_SIZE=${BLOCK_SIZE:-16384}
STEPS=${STEPS:-10}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-100}
RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-10}
RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-1}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable on the compute node: ${NAL_INDEX}" >&2
    exit 1
fi
if [[ ! -x "${FFMPEG_BINARY}" ]]; then
    echo "FFmpeg executable is not available: ${FFMPEG_BINARY}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

srun python scripts/train_byte_stage1.py \
    "${MANIFEST}" \
    --nal-index-path "${NAL_INDEX}" \
    --out-dir "${OUT_DIR}" \
    --block-size "${BLOCK_SIZE}" \
    --steps "${STEPS}" \
    --global-batch-size 8 \
    --micro-batch-size 1 \
    --precision bf16-mixed \
    --logger-name "${LOGGER_NAME}" \
    --num-workers "${NUM_WORKERS}" \
    --max-manifest-rows "${MAX_MANIFEST_ROWS}" \
    --eval-interval 5 \
    --eval-iters 2 \
    --reconstruction-eval-interval "${RECONSTRUCTION_EVAL_INTERVAL}" \
    --reconstruction-eval-samples "${RECONSTRUCTION_EVAL_SAMPLES}" \
    --ffmpeg-binary "${FFMPEG_BINARY}" \
    --save-interval 5
