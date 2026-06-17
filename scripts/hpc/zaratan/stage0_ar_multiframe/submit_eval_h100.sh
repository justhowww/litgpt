#!/bin/bash
# Submit the Stage 0 multi-frame AR (H0) continuation evaluation to an H100.
#
# Post-hoc probe over checkpoints of the Stage 0 run. Set CHECKPOINT_STEPS to the
# step numbers you want to evaluate; everything else has sensible defaults.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
export REPO_ROOT

job_id=$(
    sbatch \
        --parsable \
        --export=ALL \
        --account="${SBATCH_ACCOUNT}" \
        "${SCRIPT_DIR}/eval_h100.sbatch"
)
echo "Submitted Stage 0 continuation eval job ${job_id}"
echo "Run dir: ${RUN_DIR:-/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage0-ar-multiframe}"
echo "Output dir: ${OUT_DIR:-/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage0-ar-multiframe/continuation_eval}"
