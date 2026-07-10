#!/usr/bin/env bash
# UMD Vulcan (Nexus) allocation for the phase-1 reproduction training body.
# 2x a5000 (FSDP) -- the ~85M model @ 16384 context needs the sharding to fit 24 GB
# cards. vulcan-medium allows up to 2 GPUs / 3-day walltime, which fits this.
# submit.sh may override --output/--error (to OUT_DIR/logs) and --gres/--exclude.

#SBATCH --job-name=byte-phase1-repro
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-medium
#SBATCH --account=vulcan-metzler
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
source "${REPO_ROOT}/scripts/hpc/vulcan/phase1_repro/train.sh"
