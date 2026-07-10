#!/bin/bash
# Phase-based AVC-LM reproduction -- training body (NEW START, separate from the stageN
# ablations). Window AR (H0) over the per-MB AVC-LM corpus, encodings ON by default,
# SPS/PPS included in windows (data.py fix). Parameterized entirely by env vars so the
# per-phase submit wrappers (submit_phase0_overfit.sh, submit_phase1_repro.sh, ...) only
# set knobs. See scripts/byte/train.py for the full arg surface.
#
# Ladder: phase 0 = tiny overfit sanity (memorize + greedy free-run reproduce);
#         phase 1 = small AR reproduction; later phases add FIM.

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/vulcan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

# Model arch (head_dim 64 by convention: N_HEAD = N_EMBD/64).
N_LAYER=${N_LAYER:-12}
N_EMBD=${N_EMBD:-768}
N_HEAD=${N_HEAD:-12}
MODEL_TAG=${MODEL_TAG:-byte-phase}

OUT_DIR=${OUT_DIR:?Set OUT_DIR to the run directory}
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
BLOCK_SIZE=${BLOCK_SIZE:-16384}
STEPS=${STEPS:-100000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICES=${DEVICES:-1}
NUM_NODES=${NUM_NODES:-1}
WINDOW_MIN_FRAMES=${WINDOW_MIN_FRAMES:-2}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
MIN_LEARNING_RATE=${MIN_LEARNING_RATE:-3e-5}
VAL_FRACTION=${VAL_FRACTION:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
# MAX_ROWS = number of videos (0 = full corpus). SPLIT_BY_VIDEO=1 => held-out-video split
# (default is within-video, window-level). FREE_RUN_INTERVAL>0 turns on the in-training
# free-run (H0) probe -- set FREE_RUN_TEMP=0 for the greedy overfit-reproduction signal.
MAX_ROWS=${MAX_ROWS:-0}
SPLIT_BY_VIDEO=${SPLIT_BY_VIDEO:-0}
FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}
FREE_RUN_TEMP=${FREE_RUN_TEMP:-1.0}
FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-288}
FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-144}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    echo "Build nal_index.sqlite (scripts/byte build_byte_nal_index) before training." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
    python -u scripts/byte/train.py
    "${MANIFEST}"
    --nal-index-path "${NAL_INDEX}"
    --out-dir "${OUT_DIR}"
    --model-name "${MODEL_TAG}"
    --n-layer "${N_LAYER}"
    --n-embd "${N_EMBD}"
    --n-head "${N_HEAD}"
    --dataset-mode window
    --p-fim 0
    --window-min-frames "${WINDOW_MIN_FRAMES}"
    --block-size "${BLOCK_SIZE}"
    --steps "${STEPS}"
    --warmup-steps "${WARMUP_STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --devices "${DEVICES}"
    --num-nodes "${NUM_NODES}"
    --learning-rate "${LEARNING_RATE}"
    --min-learning-rate "${MIN_LEARNING_RATE}"
    --val-fraction "${VAL_FRACTION}"
    --precision bf16-mixed
    --logger-name "${LOGGER_NAME}"
    --num-workers "${NUM_WORKERS}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters 20
    --save-interval "${SAVE_INTERVAL}"
    --resume
)

if [[ "${MAX_ROWS}" != "0" ]]; then
    cmd+=(--max-manifest-rows "${MAX_ROWS}")
fi
if [[ "${SPLIT_BY_VIDEO}" == "1" ]]; then
    cmd+=(--split-by-video)
fi
if [[ "${NO_ENCODING:-0}" == "1" ]]; then
    cmd+=(--no-region-id --no-offset-id)
fi
if [[ "${FREE_RUN_INTERVAL}" != "0" ]]; then
    cmd+=(
        --free-run-eval-interval "${FREE_RUN_INTERVAL}"
        --free-run-temperature "${FREE_RUN_TEMP}"
        --free-run-eval-clips "${FREE_RUN_CLIPS}"
        --free-run-prefix-frames "${FREE_RUN_PREFIX_FRAMES}"
        --free-run-cont-frames "${FREE_RUN_CONT_FRAMES}"
    )
fi

COMPILE=${COMPILE:-1}
if [[ "${COMPILE}" == "1" ]]; then
    cmd+=(--compile)
fi

echo "[phase] model=${MODEL_TAG} n_layer=${N_LAYER} n_embd=${N_EMBD} n_head=${N_HEAD} block=${BLOCK_SIZE} steps=${STEPS} warmup=${WARMUP_STEPS} gbs=${GLOBAL_BATCH_SIZE} max_rows=${MAX_ROWS} split_by_video=${SPLIT_BY_VIDEO} no_encoding=${NO_ENCODING:-0} free_run_interval=${FREE_RUN_INTERVAL} free_run_temp=${FREE_RUN_TEMP}"

flock -n "${OUT_DIR}/.training.lock" srun --unbuffered "${cmd[@]}" || {
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
}
