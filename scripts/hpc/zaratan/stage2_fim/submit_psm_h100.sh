#!/bin/bash
# Submit the explicit Prefix-Suffix-Middle FIM ablation on an H100.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export JOB_SCRIPT="${SCRIPT_DIR}/train_h100.sbatch"

exec bash "${SCRIPT_DIR}/submit_psm.sh"
