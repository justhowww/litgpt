#!/usr/bin/env bash
# Continue the Phase 3 BSCV MEGABYTE patch-8 run with the exact full-sequence
# FIM + EOS setup used by:
#   byte-phase3-bscv-341m-megabyte-patch8-fim-p050-fullseq-eos
#
# Usage:
#   scripts/hpc/vulcan/phase3/resubmit_bscv_megabyte_fullseq.sh <AFTER_JOBID>
#
# The generic resubmit wrapper supplies the afterany dependency. train.sh passes
# --resume, so the submitted segment resumes OUT_DIR's latest checkpoint.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <AFTER_JOBID>" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STAGED_CORPUS=${STAGED_CORPUS:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data}
export NAL_INDEX=${NAL_INDEX:-${STAGED_CORPUS}/nal_index.sqlite}
export MAX_ROWS=${MAX_ROWS:-45000}

export BYTE_PATCH_SIZE=${BYTE_PATCH_SIZE:-8}
export P_FIM=${P_FIM:-0.5}
export FIXED_FIM_HOLES=${FIXED_FIM_HOLES:-0}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export FIM_LOSS_SCOPE=${FIM_LOSS_SCOPE:-full}
export USE_EOS=${USE_EOS:-1}

export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}
export FREE_RUN_SLICE_LAYOUT=${FREE_RUN_SLICE_LAYOUT:-frame}
export STEPS=${STEPS:-1000000}

export MODEL_TAG=${MODEL_TAG:-byte-phase3-bscv-341m-megabyte-patch8-fim-p050-fullseq-eos}
export OUT_DIR=${OUT_DIR:-${STAGED_CORPUS}/runs/${MODEL_TAG}}

exec "${SCRIPT_DIR}/resubmit.sh" "$1"
