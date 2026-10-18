#!/bin/bash
# Multi-GPU training job for the JPEG-LM generalization runs (one node, N A100s,
# one task per GPU for FSDP). Training only: submit.py chains a dependent 1-GPU
# eval job (train_eval_a100_1gpu.sbatch with OVERFIT_EVAL_ONLY=1), so the extra
# GPUs are not held idle during the single-GPU evaluation.
# submit.sh overrides --account/--mem/--time/--cpus-per-task and the log paths;
# submit.py sets --gpus/--ntasks-per-node through SBATCH_GPUS (see below).

#SBATCH --job-name=jpeglm-train-ngpu
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
export DEVICES=${DEVICES:-${SLURM_NTASKS_PER_NODE:-4}}
export NUM_NODES=1

echo "[gpu-check] host=$(hostname)"
srun --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-4}" bash -c \
    'echo "[gpu-check] rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} $(nvidia-smi -L | wc -l) GPU(s) visible"'

source "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_pretrain/train.sh"
