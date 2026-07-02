#!/bin/bash
# XL ablation A: correct STEPS with a real LR anneal, region/offset encoding ON.
#
# Purpose: isolate the *schedule*. vs the original XL run (encoding on, 300k-step
# horizon that never annealed) this changes only STEPS + warmup, so any CE/survival
# delta is the annealing/right-sizing effect. Pairs with submit_ablation_noenc.sh,
# which differs from THIS run only by NO_ENCODING=1 (isolating the encoding).
#
# STEPS≈40,000, NOT 4 epochs (~12,500): XL's CE kept dropping past epoch 4
# (4.67 @ ~12.5k → 4.56 @ ~44k, gap ~0.09 so real, not memorization), so 4 epochs
# undertrains it. ~40k matches where the curve flattened AND the original run's
# token budget → enc-vs-old-XL is a clean "same tokens, now annealed vs constant-LR"
# test. Cosine horizon = STEPS (max_tokens = STEPS x gbs x block), so LR anneals
# 3e-4 → 3e-5 within the run.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STAGED_CORPUS
export STEPS=${STEPS:-40000}
export WARMUP_STEPS=${WARMUP_STEPS:-800}
export MODEL_TAG=${MODEL_TAG:-byte-stage0-xl-abl-enc}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage0-xl-abl-enc"}
# encoding ON: NO_ENCODING left unset (default 0)

echo "[xl-ablation] encoding=ON STEPS=${STEPS} WARMUP=${WARMUP_STEPS} OUT_DIR=${OUT_DIR}"
exec bash "${SCRIPT_DIR}/submit.sh"
