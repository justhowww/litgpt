#!/bin/bash
# Submit the 1.3 B Stage 0 multi-frame AR (H0-1p3b) run to an H100 via the wrapper.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export JOB_SCRIPT="${SCRIPT_DIR}/train_h100.sbatch"

exec bash "${SCRIPT_DIR}/submit.sh"
