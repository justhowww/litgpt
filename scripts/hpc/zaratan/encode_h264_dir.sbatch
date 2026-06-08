#!/bin/bash
#SBATCH --job-name=encode-h264
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# This job intentionally accesses only the repository and project scratch.
# SHELL.metzler-prj is unavailable from Zaratan compute nodes.
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${STAGED_SOURCE_ROOT:?Set STAGED_SOURCE_ROOT to the scratch source root}"
: "${SCRATCH_CORPUS:?Set SCRATCH_CORPUS to the persistent scratch H.264 corpus}"
: "${PART_NAME:?Set PART_NAME to the staged source directory name}"

CONFIG=${CONFIG:-"${REPO_ROOT}/preprocessing/h264_preprocess_config.json"}
PART_MANIFEST="${SCRATCH_CORPUS}/.manifest-${PART_NAME}.jsonl"

if [[ ! -d "${STAGED_SOURCE_ROOT}/${PART_NAME}" ]]; then
    echo "Staged source directory is missing: ${STAGED_SOURCE_ROOT}/${PART_NAME}" >&2
    exit 1
fi

mkdir -p "${SCRATCH_CORPUS}"

# Rebuild this directory's shard on every run. --skip-existing reconstructs
# rows for completed outputs and encodes only files that are still missing.
rm -f "${PART_MANIFEST}"

srun python preprocessing/parse_mp4_to_h264.py \
    "${STAGED_SOURCE_ROOT}" \
    "${SCRATCH_CORPUS}" \
    --config "${CONFIG}" \
    --manifest "${PART_MANIFEST}" \
    --skip-existing \
    --fast-skip-existing

if [[ ! -s "${PART_MANIFEST}" ]]; then
    echo "Encoding completed without a non-empty manifest: ${PART_MANIFEST}" >&2
    exit 1
fi
