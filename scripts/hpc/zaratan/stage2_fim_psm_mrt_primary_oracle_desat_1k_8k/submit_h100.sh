#!/bin/bash
# Submit MRT-primary oracle-length desaturated run (alpha) to an H100.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
export JOB_SCRIPT="${BASE_DIR}/train_h100.sbatch"

exec bash "${SCRIPT_DIR}/submit.sh"
