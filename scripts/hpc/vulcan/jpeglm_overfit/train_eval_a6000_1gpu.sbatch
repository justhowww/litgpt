#!/bin/bash
# UMD Vulcan (Nexus) allocation for the JPEG-LM overfit / generalization runs: one
# RTX A6000 (48 GB). Only the allocation and the environment differ from Zaratan.
# The training and evaluation body is the Zaratan job script itself, sourced below
# (its #SBATCH lines are plain comments there), so both clusters run the same
# train -> eval sequence. Submit through zaratan/jpeglm_overfit/submit.py with
# `slurm.cluster: vulcan` in the YAML; submit.sh overrides --account/--mem/--time
# and the log paths.
#
# vulcan-ampere caps each GPU at 4 CPUs / 48 GB (submit.py sets 4 CPUs for
# vulcan; keep data.num_workers <= 3 and slurm.mem <= 48G). vulcan-medium allows
# the 72 h wall time phase 0 used; an A6000 trains roughly half as fast as an
# A100, so budget about twice the Zaratan time.

#SBATCH --job-name=jpeglm-overfit
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-medium
#SBATCH --account=vulcan-metzler
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=72:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
# train.sh sources this instead of zaratan/env.sh; eval-only jobs skip train.sh,
# so activate it here as well.
export CLUSTER_ENV_SH="${REPO_ROOT}/scripts/hpc/vulcan/env.sh"
source "${CLUSTER_ENV_SH}"
# FFmpeg comes from the module system on Vulcan (strict decode check + concealment
# baseline). Match the version used on Zaratan so baselines stay comparable.
set +u
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        [[ -f "${init}" ]] && source "${init}"
    done
fi
module load "${VULCAN_FFMPEG_MODULE:-ffmpeg/7.1}"
set -u
echo "[ffmpeg] $(command -v ffmpeg): $(ffmpeg -version | head -1)"
echo "[gpu-check] host=$(hostname) $(nvidia-smi -L | wc -l) GPU(s): $(nvidia-smi -L | head -1)"

source "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_overfit/train_eval_a100_1gpu.sbatch"
