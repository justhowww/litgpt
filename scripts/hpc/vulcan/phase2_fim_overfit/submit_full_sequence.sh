#!/usr/bin/env bash
# Literature-matched causal-FIM loss-scope ablation.
#
# Matches byte-fim-changing-p050-ar-eos-matched-20v-10m in data mixture,
# changing holes, EOS, and update count. The only intended change is that FIM
# samples supervise the complete reordered sequence instead of only the bridge.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STAGED_CORPUS=${STAGED_CORPUS:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm}
export MAX_ROWS=${MAX_ROWS:-21}
export VAL_FRACTION=${VAL_FRACTION:-0.05}
export P_FIM=0.5
export FIXED_FIM_HOLES=0
export FIM_LOSS_SCOPE=full
export USE_EOS=1
export STEPS=${STEPS:-10000}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}
export MODEL_TAG=${MODEL_TAG:-byte-fim-changing-p050-fullseq-20v-10m}
export OUT_DIR=${OUT_DIR:-${STAGED_CORPUS}/runs/${MODEL_TAG}}

exec "${SCRIPT_DIR}/submit.sh"
