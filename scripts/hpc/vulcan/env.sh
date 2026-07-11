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
CONDA_ROOT=${CONDA_ROOT:-"/vulcanscratch/${USER}/miniconda3"}
CONDA_ENV=${CONDA_ENV:-"litgpt"}
CACHE_ROOT=${CACHE_ROOT:-"/vulcanscratch/${USER}/cache"}


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

# --- Single-node multi-GPU NCCL (only matters when DEVICES>1) ---------------------------
# The first collective (a barrier) hung on 2x a6000: classic single-node NCCL failure --
# PCIe P2P with ACS enabled, or NCCL probing a broken/absent IB interface. Forcing SHM
# transport (P2P + IB disabled) is the standard fix; it's a bit slower than NVLink/P2P but
# reliable. Set NCCL_P2P_DISABLE=0 to try P2P once you confirm the node has working NVLink.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# NCCL_DEBUG=INFO logs the chosen transport + ring/tree setup so the *reason* is visible in
# the .err file (grep 'NCCL INFO'). Drop to WARN once it's working to quiet the logs.
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"
cd "${REPO_ROOT}"
