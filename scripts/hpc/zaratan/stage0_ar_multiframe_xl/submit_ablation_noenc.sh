#!/bin/bash
# XL ablation B: correct STEPS (~40k, matching where XL's loss flattened) with LR
# anneal, region/offset encoding OFF -- vanilla byte-AR (RoPE + explicit start codes).
#
# The ONLY difference from submit_ablation_enc.sh is NO_ENCODING=1, so the A-vs-B
# comparison cleanly isolates the additive region/offset embeddings (same STEPS,
# warmup, seed=42, within-video split, data order). Same-token comparison, so read
# CE (val_loss_ar) AND survival/tf_byte_acc between the two.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STAGED_CORPUS
export STEPS=${STEPS:-40000}
export WARMUP_STEPS=${WARMUP_STEPS:-800}
export MODEL_TAG=${MODEL_TAG:-byte-stage0-xl-abl-noenc}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage0-xl-abl-noenc"}
export NO_ENCODING=1

echo "[xl-ablation] encoding=OFF STEPS=${STEPS} WARMUP=${WARMUP_STEPS} OUT_DIR=${OUT_DIR}"
exec bash "${SCRIPT_DIR}/submit.sh"
