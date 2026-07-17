#!/usr/bin/env bash
# UMD Vulcan (Nexus) allocation for the phase-2 AR+FIM training body. Identical to
# phase1_repro/train.sbatch -- 2x a6000 (48 GB), FSDP data-parallel, 3-day walltime --
# so phase 2 differs from phase 1 in objective only, not in hardware or batch shape.
# FIM samples are SHORTER than AR ones (the window is truncated at the hole's frame),
# so this is not a memory increase over phase 1. submit.sh may override
# --output/--error and --gres/--exclude.

#SBATCH --job-name=byte-phase2-fim
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-medium
#SBATCH --account=vulcan-metzler
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
source "${REPO_ROOT}/scripts/hpc/vulcan/phase2_fim/train.sh"
