#!/usr/bin/env bash
# UMD Vulcan (Nexus) allocation for the phase-0 overfit training body.
# Single a5000 -- the ~10.6M model + 10-video overfit fits comfortably on 24 GB.
# submit.sh may override --output/--error (to OUT_DIR/logs) and --gres/--exclude.

#SBATCH --job-name=byte-phase0-overfit
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-medium
#SBATCH --account=vulcan-metzler
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
source "${REPO_ROOT}/scripts/hpc/vulcan/phase0_overfit/train.sh"
