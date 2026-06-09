#!/bin/bash
# Build the persistent H.264 NAL-offset cache on a CPU node. This removes the
# 153k-file scan from every GPU training startup.
# Submit through submit_byte_nal_index.sh so MANIFEST and NAL_INDEX are set.

#SBATCH --job-name=byte-nal-index
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident manifest.jsonl}"
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
INDEX_WORKERS=${INDEX_WORKERS:-"${SLURM_CPUS_PER_TASK}"}
REBUILD_INDEX=${REBUILD_INDEX:-0}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi

cmd=(
    python scripts/build_byte_nal_index.py
    "${MANIFEST}"
    --output "${NAL_INDEX}"
    --workers "${INDEX_WORKERS}"
)
if [[ "${REBUILD_INDEX}" == "1" ]]; then
    cmd+=(--rebuild)
fi

echo "Manifest: ${MANIFEST}"
echo "NAL index: ${NAL_INDEX}"
echo "Workers: ${INDEX_WORKERS}"
srun "${cmd[@]}"
