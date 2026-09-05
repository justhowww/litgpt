#!/bin/bash
# Shared training body for JPEG-LM full-corpus AR+FIM MEGABYTE pretraining.

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the JPEG-LM manifest.jsonl}"
: "${NAL_INDEX:?Set NAL_INDEX to the matching persistent NAL index}"
: "${OUT_DIR:?Set OUT_DIR to the run directory}"

N_LAYER=${N_LAYER:-32}
N_EMBD=${N_EMBD:-4096}
N_HEAD=${N_HEAD:-64}
MODEL_TAG=${MODEL_TAG:-byte-jpeglm-7b-megabyte-patch8-fim-p050-fullseq-eos-1m}
BLOCK_SIZE=${BLOCK_SIZE:-16384}
BYTE_PATCH_SIZE=${BYTE_PATCH_SIZE:-8}
MEGABYTE_LOCAL_LAYERS=${MEGABYTE_LOCAL_LAYERS:-4}
MEGABYTE_LOCAL_EMBD=${MEGABYTE_LOCAL_EMBD:-512}
MEGABYTE_LOCAL_HEADS=${MEGABYTE_LOCAL_HEADS:-8}

STEPS=${STEPS:-1000000}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
MIN_LEARNING_RATE=${MIN_LEARNING_RATE:-3e-5}
MAX_ROWS=${MAX_ROWS:-0}
VAL_FRACTION=${VAL_FRACTION:-0.01}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICES=${DEVICES:-4}
NUM_NODES=${NUM_NODES:-1}

P_FIM=${P_FIM:-0.5}
FIM_FORMAT=${FIM_FORMAT:-psm}
FIM_LOSS_SCOPE=${FIM_LOSS_SCOPE:-full}
FIM_SPAN_LOSS_WEIGHT=${FIM_SPAN_LOSS_WEIGHT:-0}
EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-0}
FIM_MIN_GAP=${FIM_MIN_GAP:-64}
FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
# Deliberately expose start-code and header reconstruction during JPEG-LM FIM.
SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-0}
WINDOW_MIN_FRAMES=${WINDOW_MIN_FRAMES:-2}
WINDOW_UNIT=${WINDOW_UNIT:-gop}

EVAL_INTERVAL=${EVAL_INTERVAL:-250}
EVAL_ITERS=${EVAL_ITERS:-20}
# A full training-state checkpoint is ~65 GB. Keep permanent milestones sparse
# while updating one rolling recovery checkpoint at the old cadence.
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
LATEST_SAVE_INTERVAL=${LATEST_SAVE_INTERVAL:-1000}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}
COMPILE=${COMPILE:-1}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-1}
MIN_P_FIM_ELIGIBILITY=${MIN_P_FIM_ELIGIBILITY:-0.5}
MIN_MANIFEST_ROWS=${MIN_MANIFEST_ROWS:-900000}
ALLOW_LOW_FIM_ELIGIBILITY=${ALLOW_LOW_FIM_ELIGIBILITY:-0}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    exit 1
fi

# Repeat the corpus check inside the allocation. This is required for jobs
# submitted with an afterok dependency, where the index may not exist yet at
# submission time. It runs before model allocation and fails cheaply.
preflight=(
    python "${REPO_ROOT}/scripts/byte/reports/check_jpeglm_pretrain_corpus.py"
    "${MANIFEST}"
    --nal-index-path "${NAL_INDEX}"
    --fim-min-gap "${FIM_MIN_GAP}"
    --frame-guard-bytes "${SLICE_HEADER_GUARD_BYTES}"
    --min-p-frame-eligibility "${MIN_P_FIM_ELIGIBILITY}"
    --min-manifest-rows "${MIN_MANIFEST_ROWS}"
)
if [[ "${ALLOW_LOW_FIM_ELIGIBILITY}" == "1" ]]; then
    preflight+=(--allow-low-fim-eligibility)
fi
"${preflight[@]}"

if (( N_EMBD % N_HEAD != 0 )); then
    echo "N_EMBD=${N_EMBD} must be divisible by N_HEAD=${N_HEAD}" >&2
    exit 1
fi
if (( N_EMBD % BYTE_PATCH_SIZE != 0 )); then
    echo "N_EMBD=${N_EMBD} must be divisible by BYTE_PATCH_SIZE=${BYTE_PATCH_SIZE}" >&2
    exit 1
fi
if (( MEGABYTE_LOCAL_EMBD % MEGABYTE_LOCAL_HEADS != 0 )); then
    echo "MEGABYTE_LOCAL_EMBD=${MEGABYTE_LOCAL_EMBD} must be divisible by MEGABYTE_LOCAL_HEADS=${MEGABYTE_LOCAL_HEADS}" >&2
    exit 1
fi
WORLD_SIZE=$((DEVICES * NUM_NODES))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * WORLD_SIZE) != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE*WORLD_SIZE=$((MICRO_BATCH_SIZE * WORLD_SIZE))" >&2
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
    --window-min-frames "${WINDOW_MIN_FRAMES}"
    --window-unit "${WINDOW_UNIT}"
    --p-fim "${P_FIM}"
    --fim-format "${FIM_FORMAT}"
    --fim-loss-scope "${FIM_LOSS_SCOPE}"
    --fim-span-loss-weight "${FIM_SPAN_LOSS_WEIGHT}"
    --fim-min-gap "${FIM_MIN_GAP}"
    --fim-max-gap "${FIM_MAX_GAP}"
    --slice-header-guard-bytes "${SLICE_HEADER_GUARD_BYTES}"
    --use-eos
    --eos-aux-loss-weight "${EOS_AUX_LOSS_WEIGHT}"
    --no-region-id
    --no-offset-id
    --split-by-video
    --block-size "${BLOCK_SIZE}"
    --byte-patch-size "${BYTE_PATCH_SIZE}"
    --megabyte-local-layers "${MEGABYTE_LOCAL_LAYERS}"
    --megabyte-local-embd "${MEGABYTE_LOCAL_EMBD}"
    --megabyte-local-heads "${MEGABYTE_LOCAL_HEADS}"
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
    --eval-iters "${EVAL_ITERS}"
    --save-interval "${SAVE_INTERVAL}"
    --latest-save-interval "${LATEST_SAVE_INTERVAL}"
    --resume
)

if [[ "${MAX_ROWS}" != "0" ]]; then
    cmd+=(--max-manifest-rows "${MAX_ROWS}")
fi
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
    cmd+=(--activation-checkpointing)
fi
if [[ "${COMPILE}" == "1" ]]; then
    cmd+=(--compile)
fi

echo "[jpeglm] model=${N_LAYER}L/${N_EMBD}D/${N_HEAD}H patch=${BYTE_PATCH_SIZE} raw_capacity~=$((BLOCK_SIZE * BYTE_PATCH_SIZE))B"
echo "[jpeglm] rows=${MAX_ROWS} steps=${STEPS} gbs=${GLOBAL_BATCH_SIZE} micro=${MICRO_BATCH_SIZE} devices=${DEVICES} activation_checkpointing=${ACTIVATION_CHECKPOINTING} compile=${COMPILE}"
echo "[jpeglm] p_fim=${P_FIM} format=${FIM_FORMAT} loss=${FIM_LOSS_SCOPE} gap=[${FIM_MIN_GAP},${FIM_MAX_GAP}] guard=${SLICE_HEADER_GUARD_BYTES} window=${WINDOW_UNIT} EOS=on split=held-out-video"
echo "[jpeglm] checkpoints: rolling latest every ${LATEST_SAVE_INTERVAL} steps; permanent milestone every ${SAVE_INTERVAL} steps"

exec {training_lock_fd}>"${OUT_DIR}/.training.lock"
if ! flock -n "${training_lock_fd}"; then
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
fi
srun --unbuffered "${cmd[@]}"
