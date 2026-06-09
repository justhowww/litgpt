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
: "${MANIFEST:?Set MANIFEST to the scratch-resident manifest.jsonl}"
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
INDEX_WORKERS=${INDEX_WORKERS:-"${SLURM_CPUS_PER_TASK}"}
REBUILD_INDEX=${REBUILD_INDEX:-0}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-"/home/${USER}/scratch.metzler-prj"}
CONDA_ROOT=${CONDA_ROOT:-"${PROJECT_SCRATCH}/miniforge3"}
CONDA_ENV=${CONDA_ENV:-"litpt"}
PYTHON_BIN=${PYTHON_BIN:-"${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"}

# Use the environment's interpreter directly. `conda activate` can block on
# shared-filesystem initialization before the job emits any diagnostics.
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

echo "Job started on $(/usr/bin/hostname) at $(/usr/bin/date --iso-8601=seconds)"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Manifest: ${MANIFEST}"
echo "NAL index: ${NAL_INDEX}"
echo "Workers: ${INDEX_WORKERS}"

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

cmd=(
    "${PYTHON_BIN}" scripts/build_byte_nal_index.py
    "${MANIFEST}"
    --output "${NAL_INDEX}"
    --workers "${INDEX_WORKERS}"
)
if [[ "${REBUILD_INDEX}" == "1" ]]; then
    cmd+=(--rebuild)
fi

srun "${cmd[@]}"
