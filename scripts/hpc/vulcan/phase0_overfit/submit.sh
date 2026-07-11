#!/bin/bash
# Phase 0 -- tiny overfit sanity check on UMD Vulcan (single a6000). ~10 videos, ~10.6M
# model (6/384/6), NO encodings (AVC-LM-style), SPS/PPS in windows. Goal: memorize + GREEDY free-run
# reproduce the training videos (L0->L3; L2 = byte-exact reproduction = pass/fail line).
# STEPS is a high ceiling (1M); overfit converges far earlier and the run is resumable.
# SLURM placement is in train.sbatch (vulcan-ampere / vulcan-medium / vulcan-metzler).
# Run on a Vulcan login node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm"}

# ~10.6M model (head_dim 64); single GPU is plenty for the overfit.
export N_LAYER=${N_LAYER:-6}
export N_EMBD=${N_EMBD:-384}
export N_HEAD=${N_HEAD:-6}
export DEVICES=${DEVICES:-1}
# AVC-LM-faithful: NO additive region/offset-id encodings (--no-region-id --no-offset-id).
export NO_ENCODING=${NO_ENCODING:-1}
# vulcan-ampere caps 4 CPU / 48 G per GPU -> the sbatch requests 4 CPU / 48 G for 1 GPU.
export NUM_WORKERS=${NUM_WORKERS:-3}

export MAX_ROWS=${MAX_ROWS:-10}                 # 10 videos
export STEPS=${STEPS:-1000000}                  # ceiling; overfit converges early, resumable
# WARMUP_STEPS defaults to 2% of STEPS (=20000 at STEPS=1M) in train.sh.
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
export VAL_FRACTION=${VAL_FRACTION:-0.02}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
# no region/offset encodings (AVC-LM-style). Greedy free-run probe = live L2 signal
# (per-MB: prefix/cont are MB-slices).
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-500}
export FREE_RUN_TEMP=${FREE_RUN_TEMP:-0.0}
export FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
export FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-288}
export FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-144}

export MODEL_TAG=${MODEL_TAG:-byte-phase0-overfit-10v-10m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
mkdir -p "${OUT_DIR}/logs"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL
    --output="${OUT_DIR}/logs/%x-%j.out" --error="${OUT_DIR}/logs/%x-%j.err")
[[ -n "${EXCLUDE_NODES:-}" ]] && sbatch_args+=(--exclude="${EXCLUDE_NODES}")

echo "[phase0-overfit/vulcan] ${MAX_ROWS} videos / ~10M (${N_LAYER}/${N_EMBD}/${N_HEAD}) / ${DEVICES}x a6000 / no_encoding=${NO_ENCODING} / steps=${STEPS}"
echo "  OUT_DIR=${OUT_DIR}"
job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train.sbatch")
echo "Submitted phase0-overfit job ${job_id}"
