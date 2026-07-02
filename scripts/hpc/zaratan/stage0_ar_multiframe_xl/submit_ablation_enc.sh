#!/bin/bash
# XL ablation A: correct STEPS (~4 epochs of the ~3.3B-token corpus) with a real
# LR anneal, region/offset encoding ON.
#
# Purpose: isolate the *schedule*. vs the original XL run (encoding on, 300k-step
# horizon that never annealed) this changes only STEPS + warmup, so any CE/survival
# delta is the annealing/right-sizing effect. Pairs with submit_ablation_noenc.sh,
# which differs from THIS run only by NO_ENCODING=1 (isolating the encoding).
#
# ~4 epochs: 3.3B tokens / (64 x 16384 tokens per step) ≈ 3,140 steps/epoch → 12,500.
# Cosine horizon = STEPS automatically (max_tokens = STEPS x gbs x block), so LR
# anneals 3e-4 → 3e-5 within the run.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STAGED_CORPUS
export STEPS=${STEPS:-12500}
export WARMUP_STEPS=${WARMUP_STEPS:-300}
export MODEL_TAG=${MODEL_TAG:-byte-stage0-xl-abl-enc}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage0-xl-abl-enc"}
# encoding ON: NO_ENCODING left unset (default 0)

echo "[xl-ablation] encoding=ON STEPS=${STEPS} WARMUP=${WARMUP_STEPS} OUT_DIR=${OUT_DIR}"
exec bash "${SCRIPT_DIR}/submit.sh"
