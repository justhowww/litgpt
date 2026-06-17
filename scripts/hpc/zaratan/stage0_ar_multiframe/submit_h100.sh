#!/bin/bash
# Submit the Stage 0 multi-frame AR (H0) run to an H100 via the common wrapper.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export JOB_SCRIPT="${SCRIPT_DIR}/train_h100.sbatch"

exec bash "${SCRIPT_DIR}/submit.sh"
