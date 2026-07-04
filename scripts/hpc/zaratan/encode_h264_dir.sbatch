#!/bin/bash
#SBATCH --job-name=encode-h264
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# This job intentionally accesses only the repository and project scratch.
# SHELL.metzler-prj is unavailable from Zaratan compute nodes.
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

# ffmpeg/ffprobe are required before any source file can be probed. Prefer the
# cluster module when available; conda-provided binaries still work as fallback.
FFMPEG_MODULE=${FFMPEG_MODULE:-ffmpeg}
if command -v module >/dev/null 2>&1; then
    module load "${FFMPEG_MODULE}" || true
fi
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null

: "${STAGED_SOURCE_ROOT:?Set STAGED_SOURCE_ROOT to the scratch source root}"
: "${SCRATCH_CORPUS:?Set SCRATCH_CORPUS to the persistent scratch H.264 corpus}"
: "${PART_NAME:?Set PART_NAME to the staged source directory name}"

CONFIG=${CONFIG:-"${REPO_ROOT}/preprocessing/h264_preprocess_config.json"}
# Encode this many videos concurrently (one single-threaded ffmpeg each).
# Defaults to the allocation's cpus-per-task so cores are not left idle.
ENCODE_WORKERS=${ENCODE_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
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
    --fast-skip-existing \
    --workers "${ENCODE_WORKERS}"

if [[ ! -s "${PART_MANIFEST}" ]]; then
    echo "Encoding completed without a non-empty manifest: ${PART_MANIFEST}" >&2
    exit 1
fi
