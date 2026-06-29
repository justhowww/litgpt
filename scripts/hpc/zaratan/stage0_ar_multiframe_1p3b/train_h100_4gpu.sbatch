#!/bin/bash
# 4xH100 allocation for the 1.3 B Stage 0 multi-frame AR (H0-1p3b) training body.
#
# Multi-GPU FSDP: litgpt switches to FSDPStrategy (HYBRID_SHARD, per-Block)
# automatically when devices*num_nodes>1. One srun task per GPU is REQUIRED
# (--ntasks-per-node=4) or Fabric only sees a single rank.

#SBATCH --job-name=byte-stage0-ar-1p3b-4gpu
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
# One rank per GPU; FSDP shards params/grads/optimizer across them.
export DEVICES=${DEVICES:-4}
export NUM_NODES=${NUM_NODES:-1}
source "${REPO_ROOT}/scripts/hpc/zaratan/stage0_ar_multiframe_1p3b/train.sh"
