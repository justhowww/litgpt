#!/bin/bash
# Phase 1 -- small AVC-LM reproduction (road list "phase 1"). ~1000 videos, ~85M model,
# AR-only, encodings ON, within-video split (window-level). Goal: reproduce basic AR
# behavior and debug desync at a size where memorization is no longer trivial.
# Signal: (1) training-set eval and (2) within-video eval -- both via
# eval_stream_continuation.py (train-set = --train-split-file OUT_DIR/train_split.json).
#
# Run on a Zaratan login node. AVC-LM (per-MB) corpus.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-avclm"}

# ~85M model: n_layer 12, n_embd 768, n_head 12 (head_dim 64). Bump N_EMBD=896 for ~115M.
export N_LAYER=${N_LAYER:-12}
export N_EMBD=${N_EMBD:-768}
export N_HEAD=${N_HEAD:-12}

export MAX_ROWS=${MAX_ROWS:-1000}              # 1000 videos
export STEPS=${STEPS:-100000}
export WARMUP_STEPS=${WARMUP_STEPS:-2000}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
export VAL_FRACTION=${VAL_FRACTION:-0.01}      # within-video: ~1% of windows held out
export EVAL_INTERVAL=${EVAL_INTERVAL:-250}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
# encodings ON; AR-only (train.sh sets --p-fim 0); within-video (SPLIT_BY_VIDEO unset).

# In-training free-run probe (greedy) on the within-video val split.
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-1000}
export FREE_RUN_TEMP=${FREE_RUN_TEMP:-0.0}
export FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
export FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-288}
export FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-144}

export MODEL_TAG=${MODEL_TAG:-byte-phase1-repro-1000v-85m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}

echo "[phase1-repro] ${MAX_ROWS} videos / ~85M (${N_LAYER}/${N_EMBD}/${N_HEAD}) / enc=ON / AR-only / within-video"
echo "  OUT_DIR=${OUT_DIR}"
exec bash "${SCRIPT_DIR}/submit.sh"
