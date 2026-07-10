#!/bin/bash
# Phase 0 -- tiny overfit sanity check (road list "phase 0"). ~10 videos, ~10M model,
# encodings ON, SPS/PPS-in-window fix live. Goal: prove the model can MEMORIZE and, with
# GREEDY free-run, reproduce the training videos (the L0->L3 ladder; L2 = byte-exact
# greedy reproduction is the pass/fail line). Trains + evals the same videos;
# train_split.json (written to OUT_DIR by train.py) records exactly what was trained, so
# the training-set eval can target it with eval_stream_continuation.py --train-split-file.
#
# Run on a Zaratan login node. Uses the AVC-LM (per-MB, slice-max-mbs=1) corpus.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# AVC-LM corpus (per-MB slices). Override STAGED_CORPUS to point elsewhere.
export STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-avclm"}

# ~10.6M model: n_layer 6, n_embd 384, n_head 6 (head_dim 64). Bump for more headroom.
export N_LAYER=${N_LAYER:-6}
export N_EMBD=${N_EMBD:-384}
export N_HEAD=${N_HEAD:-6}

export MAX_ROWS=${MAX_ROWS:-10}                # 10 videos
export STEPS=${STEPS:-30000}                   # overfit: run until train loss ~0
export WARMUP_STEPS=${WARMUP_STEPS:-50}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
export VAL_FRACTION=${VAL_FRACTION:-0.02}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
# encodings ON (NO_ENCODING left unset); SPS/PPS are included via the data.py fix.

# Live L2 signal: greedy free-run probe. Under slice-max-mbs=1 prefix/cont "frames" count
# MB-slices (2 frames context -> generate 1 frame at 144 MBs/frame for 256x144).
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-500}
export FREE_RUN_TEMP=${FREE_RUN_TEMP:-0.0}      # greedy = deterministic reproduction test
export FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
export FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-288}
export FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-144}

export MODEL_TAG=${MODEL_TAG:-byte-phase0-overfit-10v-10m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}

echo "[phase0-overfit] ${MAX_ROWS} videos / ~10M (${N_LAYER}/${N_EMBD}/${N_HEAD}) / enc=ON / greedy free-run probe"
echo "  OUT_DIR=${OUT_DIR}"
exec bash "${SCRIPT_DIR}/submit.sh"
