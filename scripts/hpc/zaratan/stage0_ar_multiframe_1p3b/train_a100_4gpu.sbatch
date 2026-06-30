#!/bin/bash
# 4xA100 allocation for the 1.3 B Stage 0 multi-frame AR (H0-1p3b) training body.
#
# Same FSDP setup as the H100 4-GPU sbatch, just a different GRES. Multi-GPU FSDP:
# litgpt switches to FSDPStrategy (HYBRID_SHARD, per-Block) automatically when
# devices*num_nodes>1. One srun task per GPU is REQUIRED (--ntasks-per-node=4) or
# Fabric only sees a single rank.
#
# MEMORY: HYBRID_SHARD shards params/grads/optimizer across the 4 ranks, so the
# binding constraint is per-GPU ACTIVATION memory at block_size 16384 (not sharded).
# On 80 GB A100 this is comfortable; on 40 GB A100 it is tight -- do the OOM
# shakeout (STEPS=50 COMPILE=0) first and drop BLOCK_SIZE (e.g. 8192) if it OOMs.
# A100 bf16 throughput is ~3x lower than H100, so expect fewer steps in 2 days.

#SBATCH --job-name=byte-stage0-ar-1p3b-4gpu-a100
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
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
