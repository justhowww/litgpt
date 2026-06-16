#!/bin/bash
# Submit the offline FIM sampling-diagnostic sweep to an H100.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}

job_id=$(
    sbatch \
        --parsable \
        --export=ALL \
        --account="${SBATCH_ACCOUNT}" \
        "${SCRIPT_DIR}/sampling_sweep_h100.sbatch"
)
echo "Submitted byte sampling sweep job ${job_id}"
echo "Run dir: ${RUN_DIR:-/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage2-fim-psm-ce-only-within-video-1k-8k}"
echo "Output root: ${OUT_ROOT:-/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage2-fim-psm-ce-only-within-video-1k-8k/offline_sampling_sweep}"
