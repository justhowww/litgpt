#!/bin/bash
# Submit the Stage 1 convergence run to an H100 while reusing the common submit
# wrapper, scratch corpus, checkpoint directory, and training hyperparameters.
#
# The A100 and H100 jobs can be queued together, but the output-directory lock
# permits only one to train. Cancel the redundant pending job after one starts.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export JOB_SCRIPT="${SCRIPT_DIR}/train_stage1_h100.sbatch"
# The H.264 corpus is already persistent on scratch and must not be removed
# after training.
export CLEANUP_AFTER_SUCCESS=${CLEANUP_AFTER_SUCCESS:-0}

exec bash "${SCRIPT_DIR}/submit_stage1.sh"
