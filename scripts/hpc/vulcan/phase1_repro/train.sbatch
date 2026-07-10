#!/usr/bin/env bash
# UMD Vulcan (Nexus) allocation for the phase-1 reproduction training body.
# 2x a6000 (48 GB) -- data-parallel (FSDP) for 2x throughput / double batch. The 85M @
# 16384 model fits on ONE a6000, so 2 GPUs is a throughput choice, not a memory need
# (set --gres=gpu:rtxa6000:1 + DEVICES=1 to run single-GPU). vulcan-medium allows up to
# 2 GPUs / 3-day walltime. submit.sh may override --output/--error and --gres/--exclude.

#SBATCH --job-name=byte-phase1-repro
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-medium
#SBATCH --account=vulcan-metzler
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
source "${REPO_ROOT}/scripts/hpc/vulcan/phase1_repro/train.sh"
