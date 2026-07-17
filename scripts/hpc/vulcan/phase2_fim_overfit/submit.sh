#!/bin/bash
# Phase 2 (overfit sanity) -- tiny AR+FIM run on UMD Vulcan (single a6000). ~10 videos,
# ~10.6M model (6/384/6), NO encodings (AVC-LM-style), SPS/PPS in windows -- the same
# scale-down phase0_overfit applies to phase1_repro, applied here to phase2_fim.
#
# Goal: cheaply prove the FIM objective is LEARNABLE at all -- val_loss_fim actually
# drops, a FIM sample's teacher-forced span gets predicted correctly, AR free-run still
# reproduces the training clips -- before spending the full 1000v/85M/3-day budget.
# This is a sanity gate, not phase 2's result: 10 videos is far too little to read
# objective INTERFERENCE from (that needs the full run against phase 1's baseline).
#
# Deliberately reuses phase2_fim/train.sh rather than a copy (train.sbatch sources it
# directly) -- that file just picked up two real fixes (the free-run/p_fim crash, and
# resampling the hole per epoch instead of freezing it per window) and a duplicate
# here would silently drift out of sync with the next one. Run on a Vulcan login node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm"}

# --- Overfit scale, mirroring phase0_overfit -------------------------------------
export N_LAYER=${N_LAYER:-6}
export N_EMBD=${N_EMBD:-384}
export N_HEAD=${N_HEAD:-6}
export DEVICES=${DEVICES:-1}                     # single a6000 is plenty at this scale
export NO_ENCODING=${NO_ENCODING:-1}             # AVC-LM-faithful; also required by window FIM
export NUM_WORKERS=${NUM_WORKERS:-3}             # vulcan-ampere caps 4 CPU/GPU

export MAX_ROWS=${MAX_ROWS:-10}                  # 10 videos -- ~10 windows (1 IDR/clip)
export STEPS=${STEPS:-1000000}                   # ceiling; overfit converges early, resumable
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
export VAL_FRACTION=${VAL_FRACTION:-0.02}        # rounds up to 1 window (max(1, ...) in setup())
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-500}
export FREE_RUN_TEMP=${FREE_RUN_TEMP:-0.0}
export FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
# REAL frames (first_mb_in_slice == 0), not macroblocks -- see phase2_fim/submit.sh's
# note on the stale 288/144 macroblock-era numbers. 4/2 matches eval.sh below.
export FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-4}
export FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-2}

# --- FIM objective, held identical to phase2_fim's defaults ----------------------
# This is meant to be a SMALLER instance of the same experiment, not a differently
# tuned toy -- override via env vars if you want to sanity-check a different mixture.
export P_FIM=${P_FIM:-0.5}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export USE_EOS=${USE_EOS:-1}
export FIM_MIN_GAP=${FIM_MIN_GAP:-64}
export FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
export SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}

export MODEL_TAG=${MODEL_TAG:-byte-phase2-fim-overfit-10v-10m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
mkdir -p "${OUT_DIR}/logs"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL
    --output="${OUT_DIR}/logs/%x-%j.out" --error="${OUT_DIR}/logs/%x-%j.err")
[[ -n "${EXCLUDE_NODES:-}" ]] && sbatch_args+=(--exclude="${EXCLUDE_NODES}")

echo "[phase2-fim-overfit/vulcan] ${MAX_ROWS} videos / ~10M (${N_LAYER}/${N_EMBD}/${N_HEAD}) / ${DEVICES}x a6000 / p_fim=${P_FIM} / steps=${STEPS}"
echo "  OUT_DIR=${OUT_DIR}"
job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train.sbatch")
echo "Submitted phase2-fim-overfit job ${job_id}"
