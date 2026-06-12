#!/bin/bash
# Exercise one complete MRT update with the smallest useful candidate set.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STEPS=${STEPS:-8}
export MRT_INTERVAL=${MRT_INTERVAL:-8}
export MRT_NUM_CANDIDATES=${MRT_NUM_CANDIDATES:-2}
export MRT_CONTEXT_POOL_SIZE=${MRT_CONTEXT_POOL_SIZE:-2}
export MRT_MAX_TARGET_BYTES=${MRT_MAX_TARGET_BYTES:-256}
export MRT_DECODE_WORKERS=${MRT_DECODE_WORKERS:-2}
export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-0}
export OUT_DIR=${OUT_DIR:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage2-fim-psm-mrt-1k-8k-smoke"}

exec bash "${SCRIPT_DIR}/submit_h100.sh"
