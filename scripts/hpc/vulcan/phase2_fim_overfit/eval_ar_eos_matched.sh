#!/usr/bin/env bash
# Evaluate the matched-FIM-update AR+EOS experiment with learned-EOS and
# parser-reconnection stopping, both unconstrained and fully constrained.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA=${DATA:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm}
OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-fim-changing-p050-ar-eos-matched-20v-10m}
CKPT=${CKPT:-step-00010000}
NUM_CLIPS=${NUM_CLIPS:-20}

export DATA OUT_DIR CKPT NUM_CLIPS
exec "${SCRIPT_DIR}/../phase2_fim/eval_fim.sh"
