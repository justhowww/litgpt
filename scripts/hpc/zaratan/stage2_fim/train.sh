#!/bin/bash
# Shared Stage 2 mixed AR/FIM training body for A100 and H100 allocations.

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
FIM_FORMAT=${FIM_FORMAT:-bridge}
MODEL_NAME=${MODEL_NAME:-byte-stage2-fim}
FIM_MIN_GAP=${FIM_MIN_GAP:-64}
FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}
RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-1000}
RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-5}
RECONSTRUCTION_MAX_TARGET_BYTES=${RECONSTRUCTION_MAX_TARGET_BYTES:-1400}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    exit 1
fi
if [[ ! -x "${FFMPEG_BINARY}" ]]; then
    echo "FFmpeg executable is not available: ${FFMPEG_BINARY}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
    python scripts/byte/train.py
    "${MANIFEST}"
    --model-name "${MODEL_NAME}"
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
    --fim-format "${FIM_FORMAT}"
    --fim-min-gap "${FIM_MIN_GAP}"
    --fim-max-gap "${FIM_MAX_GAP}"
    --slice-header-guard-bytes "${SLICE_HEADER_GUARD_BYTES}"
    --eval-interval 250
    --eval-iters 20
    --reconstruction-task both
    --reconstruction-eval-interval "${RECONSTRUCTION_EVAL_INTERVAL}"
    --reconstruction-eval-samples "${RECONSTRUCTION_EVAL_SAMPLES}"
    --reconstruction-max-target-bytes "${RECONSTRUCTION_MAX_TARGET_BYTES}"
    --ffmpeg-binary "${FFMPEG_BINARY}"
    --save-interval 500
    --compile
    --resume
)

flock -n "${OUT_DIR}/.training.lock" srun "${cmd[@]}" || {
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
}
