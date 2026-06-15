#!/bin/bash
# Shared Stage 2 PSM (+EOS) training body for A100 and H100 allocations.

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"
: "${NAL_INDEX:?Set NAL_INDEX to the persistent SQLite NAL index}"

BLOCK_SIZE=${BLOCK_SIZE:-16384}
STEPS=${STEPS:-100000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
REFERENCE_MODE=${REFERENCE_MODE:-normal}
P_FIM=${P_FIM:-0.5}
FIM_FORMAT=${FIM_FORMAT:-psm}
# EOS is enabled by default for this experiment; set USE_EOS=0 for a PSM-only
# baseline (which writes to a distinct, untagged directory).
USE_EOS=${USE_EOS:-1}
CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-1}
CE_BYTE_ONLY=${CE_BYTE_ONLY:-0}
EOS_LOSS_WEIGHT=${EOS_LOSS_WEIGHT:-1}
EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-0}
if [[ "${USE_EOS}" == "1" ]]; then
    EOS_TAG="-eos"
else
    EOS_TAG=""
fi
MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm${EOS_TAG}}
OUT_DIR=${OUT_DIR:-"$(dirname "${MANIFEST}")/runs/byte-stage2-fim-psm${EOS_TAG}"}
FIM_MIN_GAP=${FIM_MIN_GAP:-64}
# Single-packet loss (~1 MTU/FU-A payload). Adjust for multi-packet schemes.
FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}
RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-1000}
RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-5}
RECONSTRUCTION_TASK=${RECONSTRUCTION_TASK:-both}
RECONSTRUCTION_MAX_TARGET_BYTES=${RECONSTRUCTION_MAX_TARGET_BYTES:-1400}
RECONSTRUCTION_ORACLE_LENGTH=${RECONSTRUCTION_ORACLE_LENGTH:-0}
RECONSTRUCTION_LEARNED_EOS=${RECONSTRUCTION_LEARNED_EOS:-0}
RECONSTRUCTION_ERROR_EXPLODING=${RECONSTRUCTION_ERROR_EXPLODING:-0}
RECONSTRUCTION_FIM_BASELINES=${RECONSTRUCTION_FIM_BASELINES:-0}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}
INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-}
MRT_INTERVAL=${MRT_INTERVAL:-0}
MRT_START_STEP=${MRT_START_STEP:-0}
MRT_NUM_CANDIDATES=${MRT_NUM_CANDIDATES:-16}
MRT_CONTEXT_POOL_SIZE=${MRT_CONTEXT_POOL_SIZE:-64}
MRT_MAX_TARGET_BYTES=${MRT_MAX_TARGET_BYTES:-2048}
MRT_ORACLE_LENGTH=${MRT_ORACLE_LENGTH:-0}
MRT_LEARNED_EOS=${MRT_LEARNED_EOS:-0}
MRT_TEMPERATURE=${MRT_TEMPERATURE:-1.0}
MRT_CANDIDATE_ALPHA=${MRT_CANDIDATE_ALPHA:-1.0}
MRT_WEIGHT=${MRT_WEIGHT:-4.0}
MRT_RISK_MODE=${MRT_RISK_MODE:-clipped_mse}
MRT_MSE_WEIGHT=${MRT_MSE_WEIGHT:-1000.0}
MRT_MSE_TAU=${MRT_MSE_TAU:-0.002}
MRT_DECODE_FAILURE_WEIGHT=${MRT_DECODE_FAILURE_WEIGHT:-2.0}
MRT_MAX_RISK=${MRT_MAX_RISK:-2.0}
MRT_DECODE_WORKERS=${MRT_DECODE_WORKERS:-8}
MRT_INCLUDE_GROUND_TRUTH=${MRT_INCLUDE_GROUND_TRUTH:-1}
VAL_FRACTION=${VAL_FRACTION:-0.01}
SPLIT_BY_VIDEO=${SPLIT_BY_VIDEO:-0}

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
    --ce-loss-weight "${CE_LOSS_WEIGHT}"
    --eval-interval 250
    --eval-iters 20
    --reconstruction-task "${RECONSTRUCTION_TASK}"
    --reconstruction-eval-interval "${RECONSTRUCTION_EVAL_INTERVAL}"
    --reconstruction-eval-samples "${RECONSTRUCTION_EVAL_SAMPLES}"
    --reconstruction-max-target-bytes "${RECONSTRUCTION_MAX_TARGET_BYTES}"
    --ffmpeg-binary "${FFMPEG_BINARY}"
    --save-interval 500
    --compile
)
if [[ -n "${INITIAL_CHECKPOINT_DIR}" ]]; then
    cmd+=(--initial-checkpoint-dir "${INITIAL_CHECKPOINT_DIR}")
else
    cmd+=(--resume)
fi
if (( MRT_INTERVAL > 0 )); then
    cmd+=(
        --mrt-interval "${MRT_INTERVAL}"
        --mrt-start-step "${MRT_START_STEP}"
        --mrt-num-candidates "${MRT_NUM_CANDIDATES}"
        --mrt-context-pool-size "${MRT_CONTEXT_POOL_SIZE}"
        --mrt-max-target-bytes "${MRT_MAX_TARGET_BYTES}"
        --mrt-temperature "${MRT_TEMPERATURE}"
        --mrt-candidate-alpha "${MRT_CANDIDATE_ALPHA}"
        --mrt-weight "${MRT_WEIGHT}"
        --mrt-risk-mode "${MRT_RISK_MODE}"
        --mrt-mse-weight "${MRT_MSE_WEIGHT}"
        --mrt-mse-tau "${MRT_MSE_TAU}"
        --mrt-decode-failure-weight "${MRT_DECODE_FAILURE_WEIGHT}"
        --mrt-max-risk "${MRT_MAX_RISK}"
        --mrt-decode-workers "${MRT_DECODE_WORKERS}"
    )
fi
if [[ "${USE_EOS}" == "1" ]]; then
    cmd+=(--use-eos)
fi
if [[ "${CE_BYTE_ONLY}" == "1" ]]; then
    cmd+=(--ce-byte-only)
fi
cmd+=(--eos-loss-weight "${EOS_LOSS_WEIGHT}")
cmd+=(--eos-aux-loss-weight "${EOS_AUX_LOSS_WEIGHT}")
if [[ "${RECONSTRUCTION_ORACLE_LENGTH}" == "1" ]]; then
    cmd+=(--reconstruction-oracle-length)
fi
if [[ "${MRT_ORACLE_LENGTH}" == "1" ]]; then
    cmd+=(--mrt-oracle-length)
fi
if [[ "${MRT_LEARNED_EOS}" == "1" ]]; then
    cmd+=(--mrt-learned-eos)
fi
if [[ "${MRT_INCLUDE_GROUND_TRUTH}" == "0" ]]; then
    cmd+=(--no-mrt-ground-truth)
fi
cmd+=(--val-fraction "${VAL_FRACTION}")
if [[ "${SPLIT_BY_VIDEO}" == "1" ]]; then
    cmd+=(--split-by-video)
fi
if [[ "${RECONSTRUCTION_LEARNED_EOS}" == "1" ]]; then
    cmd+=(--reconstruction-learned-eos)
fi
if [[ "${RECONSTRUCTION_ERROR_EXPLODING}" == "1" ]]; then
    cmd+=(--reconstruction-error-exploding)
fi
if [[ "${RECONSTRUCTION_FIM_BASELINES}" == "1" ]]; then
    cmd+=(--reconstruction-fim-baselines)
fi

exec 9>"${OUT_DIR}/.training.lock"
if ! flock -n 9; then
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
fi

srun "${cmd[@]}"
