#!/bin/bash
# Shared Vulcan (UMD Nexus) environment for byte-model Slurm jobs. Mirrors the Zaratan
# env.sh but for the Nexus project filesystem + a6000 nodes.
#
# >>> SET THESE for your Vulcan account if the defaults are wrong <<<
#   CONDA_ROOT  -- path to your conda/miniforge install on Vulcan
#   CONDA_ENV   -- the env with torch/lightning (this repo's deps)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-"$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"}

PROJECT_ROOT=${PROJECT_ROOT:-"/fs/nexus-projects/time-control-videogen"}
CONDA_ROOT=${CONDA_ROOT:-"${PROJECT_ROOT}/miniforge3"}   # <-- verify / override
CONDA_ENV=${CONDA_ENV:-"litgpt"}                          # <-- verify / override
CACHE_ROOT=${CACHE_ROOT:-"${PROJECT_ROOT}/cache"}


eval "$(conda shell.bash hook)"
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
