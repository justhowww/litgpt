#!/bin/bash
# H100 allocation for the 1.3 B Stage 0 multi-frame AR (H0-1p3b) training body.

#SBATCH --job-name=byte-stage0-ar-1p3b-h100
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
source "${REPO_ROOT}/scripts/hpc/zaratan/stage0_ar_multiframe_1p3b/train.sh"
