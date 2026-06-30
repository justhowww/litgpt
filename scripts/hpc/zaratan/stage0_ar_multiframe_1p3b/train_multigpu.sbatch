#!/bin/bash
# Generic N-GPU allocation for the 1.3 B Stage 0 multi-frame AR (H0-1p3b) body.
#
# GPU type and count are NOT fixed here: submit_multigpu.sh passes them as sbatch
# CLI overrides (--gpus=<type>:<n> --ntasks-per-node=<n>), which take precedence
# over the #SBATCH defaults below, and exports DEVICES=<n>. So this one file
# serves any (h100|a100) x N on a single node -- no per-config copy needed.
#
# The #SBATCH --gpus / --ntasks-per-node lines are only defaults for a bare
# `sbatch train_multigpu.sbatch` (1 GPU is wrong for FSDP, so they default to 4).
#
# Single node only: NGPU must be <= GPUs-per-node (4 on Zaratan). For more GPUs
# than one node holds, also set NODES and split ntasks accordingly.
#
# MEMORY: FSDP HYBRID_SHARD shards params/grads/optimizer across ranks, so the
# binding term is per-GPU activation memory at block_size 16384 (not sharded).
# Comfortable on 80 GB cards, tight on 40 GB A100 -- OOM-shake (STEPS=50) first.

#SBATCH --job-name=byte-stage0-ar-1p3b-multigpu
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
# One rank per GPU; FSDP shards params/grads/optimizer across them. DEVICES is
# set by the submit wrapper (= NGPU); fall back to 4 for a bare sbatch.
export DEVICES=${DEVICES:-4}
export NUM_NODES=${NUM_NODES:-1}
source "${REPO_ROOT}/scripts/hpc/zaratan/stage0_ar_multiframe_1p3b/train.sh"
