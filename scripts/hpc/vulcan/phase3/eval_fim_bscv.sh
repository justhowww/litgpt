#!/usr/bin/env bash
# Phase 3 BSCV FIM evaluation using one progressive picture per slice.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DATA=${DATA:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data}
export OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-phase3-45kv-341m-megabyte-patch8}
export MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-45000}
export MAX_WINDOW_BYTES=${MAX_WINDOW_BYTES:-131072}
export SLICE_LAYOUT=frame

exec "${SCRIPT_DIR}/../phase2_fim/eval_fim.sh"
