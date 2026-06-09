#!/bin/bash
# Run on a Zaratan login node to submit the resumable CPU NAL-index build.
#
# Optional overrides:
#   MANIFEST=/path/manifest.jsonl NAL_INDEX=/path/nal_index.sqlite \
#   INDEX_WORKERS=16 REBUILD_INDEX=0 bash scripts/hpc/zaratan/submit_byte_nal_index.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
MANIFEST=${MANIFEST:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
INDEX_WORKERS=${INDEX_WORKERS:-16}
REBUILD_INDEX=${REBUILD_INDEX:-0}

export REPO_ROOT MANIFEST NAL_INDEX INDEX_WORKERS REBUILD_INDEX

job_id=$(
    sbatch \
        --parsable \
        --export=ALL \
        --account="${SBATCH_ACCOUNT}" \
        "${SCRIPT_DIR}/build_byte_nal_index.sbatch"
)
echo "Submitted NAL-index job ${job_id}"
echo "Output: slurm-byte-nal-index-${job_id}.out"
