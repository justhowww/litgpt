#!/bin/bash
# Submit matched JPEG-LM MEGABYTE patch-size throughput pilots.
#
# Every run sees the same corpus prefix, optimizer-step count, global batch,
# local-model shape, length bucketing, and raw-byte context. Only the patch
# size (and therefore the number of global Transformer positions) changes.
# The existing patch-8 length-bucketing pilot can be reused as the baseline;
# include 8 in PATCH_SIZES to rerun it from the current code.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}
RAW_CONTEXT_BYTES=${RAW_CONTEXT_BYTES:-131072}
N_EMBD=${N_EMBD:-4096}
PATCH_SIZES=${PATCH_SIZES:-"32 64 128"}

for patch_size in ${PATCH_SIZES}; do
    if [[ ! "${patch_size}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid patch size: ${patch_size}" >&2
        exit 2
    fi
    if (( N_EMBD % patch_size != 0 )); then
        echo "Skipping patch ${patch_size}: N_EMBD=${N_EMBD} is not divisible by it" >&2
        continue
    fi
    if (( RAW_CONTEXT_BYTES % patch_size != 0 )); then
        echo "Skipping patch ${patch_size}: RAW_CONTEXT_BYTES=${RAW_CONTEXT_BYTES} is not divisible by it" >&2
        continue
    fi

    model_tag="byte-jpeglm-7b-patch${patch_size}-length-bucket-on-pool8192-pilot"
    echo "Submitting patch=${patch_size}, global_positions=$((RAW_CONTEXT_BYTES / patch_size)), raw_context=${RAW_CONTEXT_BYTES}B"
    STAGED_CORPUS="${STAGED_CORPUS}" \
    N_EMBD="${N_EMBD}" \
    BYTE_PATCH_SIZE="${patch_size}" \
    RAW_CONTEXT_BYTES="${RAW_CONTEXT_BYTES}" \
    MAX_ROWS="${MAX_ROWS:-10000}" \
    STEPS="${STEPS:-200}" \
    WARMUP_STEPS="${WARMUP_STEPS:-2}" \
    EVAL_INTERVAL="${EVAL_INTERVAL:-100}" \
    EVAL_ITERS="${EVAL_ITERS:-20}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" \
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}" \
    COMPILE="${COMPILE:-1}" \
    ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}" \
    ENABLE_LENGTH_BUCKETING=1 \
    LENGTH_BUCKET_POOL_SIZE="${LENGTH_BUCKET_POOL_SIZE:-8192}" \
    SAVE_INTERVAL=1000000 \
    LATEST_SAVE_INTERVAL=1000000 \
    SAVE_FINAL=0 \
    MODEL_TAG="${model_tag}" \
    OUT_DIR="${STAGED_CORPUS}/runs/${model_tag}" \
    bash "${SCRIPT_DIR}/submit_pilot.sh"
done
