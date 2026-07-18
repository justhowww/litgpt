#!/bin/bash
# Stage 0 multi-frame AR (H0) training body.
#
# Verifies the AVC-LM/JPEG-LM generation legacy on our setup: contiguous
# multi-frame stream windows (parameter sets + IDR + P-frames), next-byte loss
# across the whole window, window-level (within-video) split. AR only (p_fim=0);
# the slice/FIM reconstruction probe is disabled -- continuation quality is
# measured post-hoc by scripts/byte/eval/eval_ar_continuation.py. See 0616.md.

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

OUT_DIR=${OUT_DIR:-"${PROJECT_SCRATCH}/runs/byte-stage0-ar-multiframe"}
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
# Reuse the Stage 1 context budget; in window mode this sets how many frames a
# window holds, not a per-slice cap.
BLOCK_SIZE=${BLOCK_SIZE:-16384}
STEPS=${STEPS:-200000}           # extended run; resumes across job restarts until reached
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
WINDOW_MIN_FRAMES=${WINDOW_MIN_FRAMES:-2}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    echo "Build it with scripts/hpc/zaratan/submit_byte_nal_index.sh before training." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
    python -u scripts/byte/train.py
    "${MANIFEST}"
    --nal-index-path "${NAL_INDEX}"
    --out-dir "${OUT_DIR}"
    --model-name byte-stage0-ar-multiframe
    --dataset-mode window
    --p-fim 0
    --window-min-frames "${WINDOW_MIN_FRAMES}"
    --block-size "${BLOCK_SIZE}"
    --steps "${STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --precision bf16-mixed
    --logger-name "${LOGGER_NAME}"
    --num-workers "${NUM_WORKERS}"
    --eval-interval 250
    --eval-iters 20
    --save-interval 500
    --resume
)

# torch.compile is the prime suspect for a silent multi-minute first step; toggle
# it off with COMPILE=0 to isolate startup hangs.
COMPILE=${COMPILE:-1}
if [[ "${COMPILE}" == "1" ]]; then
    cmd+=(--compile)
fi

# Window-level random split (within-video) is the default (no --split-by-video).
# The reconstruction-eval probe is left at its default of 0 (off): it is a
# slice/FIM probe and does not apply to stream-window AR.

# -u (python) and --unbuffered (srun) stream manifest/indexing/step logs live so
# a startup stall is visible instead of a 0-byte slurm file.
flock -n "${OUT_DIR}/.training.lock" srun --unbuffered "${cmd[@]}" || {
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
}
