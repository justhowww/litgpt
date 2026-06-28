#!/bin/bash
# Stage 0 multi-frame AR (H0) training body -- 1.3 B model ("1p3b").
#
# Third rung of the scaling curve: 33.9 M -> 200 M (xl) -> 1.3 B. Held identical
# to the xl/H0 runs in EVERYTHING except model arch (objective window AR p_fim=0,
# within-video split, block size, global/micro batch, LR schedule) so the only
# moving variable is scale. head_dim is held at 64 (n_embd 2048 / n_head 32 = 64),
# the same head_dim as the 33.9 M (512/8) and 200 M (1024/16) runs -- a clean
# single-variable comparison.
#
# Two things this run answers (see 0616.md):
#   1. Does content-layer CE (mb_type/mvd/coeff) keep dropping with scale, i.e. is
#      free-run validity on the AVC-LM scaling curve? Read it with the free-run
#      survival-length metric, not val CE alone.
#   2. Is the CURRENT corpus big enough for 1.3 B, or does it overfit? 1.3 B is
#      ~6.5x the 200 M that the data is ~matched to, so watch the TRAIN/VAL GAP.
#      Gap blows open early -> data-limited (need more OpenVid); gap stays small
#      and val dips below xl -> scale still helping.

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

# 1.3 B arch: n_layer 24 / n_embd 2048 / n_head 32 (head_dim 64). Override for a
# memory-driven shrink, but note any arch change breaks single-variable scaling.
N_LAYER=${N_LAYER:-24}
N_EMBD=${N_EMBD:-2048}
N_HEAD=${N_HEAD:-32}
MODEL_TAG=${MODEL_TAG:-byte-stage0-ar-multiframe-1p3b}

OUT_DIR=${OUT_DIR:-"${PROJECT_SCRATCH}/runs/byte-stage0-ar-multiframe-1p3b"}
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
# Held identical to H0/xl for comparability. MEMORY NOTE: 1.3 B at 16384 context
# is heavy; on an 80 GB H100 with micro batch 1 / bf16 it is expected to fit but
# is close. If it OOMs, drop BLOCK_SIZE (e.g. 8192) -- this changes how many
# frames a window holds and weakens strict comparability, so note it if used.
BLOCK_SIZE=${BLOCK_SIZE:-16384}
# Bigger model converges slower in tokens; give headroom. Resumes across restarts.
STEPS=${STEPS:-400000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
WINDOW_MIN_FRAMES=${WINDOW_MIN_FRAMES:-2}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}

# Free-run (H0 generation) probe during validation. Off by default elsewhere; on
# here because val CE does not determine free-run validity (exposure-bias / the
# illegal-token tail). Logs parser-based survival-length + validity on a fixed
# SPS-anchored continuation set. Set FREE_RUN_INTERVAL=0 to disable.
FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-1000}
FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-8}
FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-4}

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
    --model-name "${MODEL_TAG}"
    --n-layer "${N_LAYER}"
    --n-embd "${N_EMBD}"
    --n-head "${N_HEAD}"
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
    --free-run-eval-interval "${FREE_RUN_INTERVAL}"
    --free-run-eval-clips "${FREE_RUN_CLIPS}"
    --free-run-prefix-frames "${FREE_RUN_PREFIX_FRAMES}"
    --free-run-cont-frames "${FREE_RUN_CONT_FRAMES}"
    --resume
)

# torch.compile toggle (COMPILE=0 to isolate startup hangs).
COMPILE=${COMPILE:-1}
if [[ "${COMPILE}" == "1" ]]; then
    cmd+=(--compile)
fi

echo "[stage0-1p3b] model=${MODEL_TAG} n_layer=${N_LAYER} n_embd=${N_EMBD} n_head=${N_HEAD} (head_dim $((N_EMBD / N_HEAD))) block=${BLOCK_SIZE} steps=${STEPS} gbs=${GLOBAL_BATCH_SIZE} free_run_interval=${FREE_RUN_INTERVAL}"

flock -n "${OUT_DIR}/.training.lock" srun --unbuffered "${cmd[@]}" || {
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
}
