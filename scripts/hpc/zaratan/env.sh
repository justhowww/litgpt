#!/bin/bash
# Shared Zaratan environment for byte-model Slurm jobs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

PROJECT_SCRATCH=${PROJECT_SCRATCH:-"/scratch/zt1/project/metzler-prj/user/${USER}"}
CONDA_ROOT=${CONDA_ROOT:-"${PROJECT_SCRATCH}/miniforge3"}
CONDA_ENV=${CONDA_ENV:-"litpt"}
CACHE_ROOT=${CACHE_ROOT:-"${PROJECT_SCRATCH}/cache"}

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "Conda initialization script not found under CONDA_ROOT=${CONDA_ROOT}" >&2
    exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HF_HOME=${HF_HOME:-"${CACHE_ROOT}/huggingface"}
export TORCH_HOME=${TORCH_HOME:-"${CACHE_ROOT}/torch"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${CACHE_ROOT}/xdg"}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"${CACHE_ROOT}/pip"}
export PYTHONUNBUFFERED=1

# Prevent each worker/library from spawning a full node's worth of BLAS threads.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"
cd "${REPO_ROOT}"

