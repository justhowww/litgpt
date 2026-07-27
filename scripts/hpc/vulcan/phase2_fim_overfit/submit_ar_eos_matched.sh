#!/usr/bin/env bash
# Controlled AR/FIM termination experiment.
#
# The 5k P_FIM=1 run received about 5k steps of FIM supervision. This 10k
# P_FIM=0.5 run receives the same expected number of FIM examples while also
# training AR. Unlike the earlier mixed run, window AR now appends SEQ_EOS, so
# both objectives teach an explicit stopping token.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STAGED_CORPUS=${STAGED_CORPUS:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm}
export MAX_ROWS=${MAX_ROWS:-21}
export VAL_FRACTION=${VAL_FRACTION:-0.05}
export P_FIM=0.5
export FIXED_FIM_HOLES=0
export USE_EOS=1
export STEPS=${STEPS:-10000}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}
export MODEL_TAG=${MODEL_TAG:-byte-fim-changing-p050-ar-eos-matched-20v-10m}
export OUT_DIR=${OUT_DIR:-${STAGED_CORPUS}/runs/${MODEL_TAG}}

exec "${SCRIPT_DIR}/submit.sh"
