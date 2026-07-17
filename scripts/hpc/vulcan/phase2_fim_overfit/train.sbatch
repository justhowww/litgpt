#!/usr/bin/env bash
# UMD Vulcan (Nexus) allocation for the phase-2 overfit training body. Single a6000
# (48 GB) -- the ~10.6M model + 10-video overfit fits with huge headroom, same as
# phase0_overfit. Sources phase2_fim/train.sh directly (not a copy of its own) so the
# FIM training body has one source of truth across the full run and this sanity-scale
# one. submit.sh may override --output/--error and --gres/--exclude.

#SBATCH --job-name=byte-phase2-fim-overfit
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-medium
#SBATCH --account=vulcan-metzler
#SBATCH --time=72:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
source "${REPO_ROOT}/scripts/hpc/vulcan/phase2_fim/train.sh"
