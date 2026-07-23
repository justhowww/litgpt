#!/usr/bin/env bash
#SBATCH --job-name=fim-fixed-overfit
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-medium
#SBATCH --account=vulcan-metzler
#SBATCH --time=3:00:00
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
