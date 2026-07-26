#!/usr/bin/env bash
# Phase 3 BSCV AR evaluation: same paired greedy/full-mask sweep as eval.sh,
# pointed at the one-progressive-picture-per-slice corpus. The online mask resolves
# each slice's macroblock extent from SPS; no resolution-specific number is supplied.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DATA=${DATA:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data}
export OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-phase3-bscv-341m}
export NAL_INDEX=${NAL_INDEX:-${DATA}/nal_index.sqlite}
export MAX_GEN_MULTIPLE=${MAX_GEN_MULTIPLE:-4}
export SLICE_LAYOUT=frame
export EVAL_INTRA=${EVAL_INTRA:-0}

exec "${SCRIPT_DIR}/eval.sh"
