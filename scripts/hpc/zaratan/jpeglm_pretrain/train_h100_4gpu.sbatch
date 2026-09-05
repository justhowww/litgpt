#!/bin/bash
# 7B JPEG-LM pretraining allocation. One task per H100 is required for FSDP.

#SBATCH --job-name=jpeglm-7b
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
export DEVICES=${DEVICES:-4}
export NUM_NODES=${NUM_NODES:-1}

echo "[gpu-check] host=$(hostname)"
srun --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-4}" bash -c \
    'echo "[gpu-check] rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} $(nvidia-smi -L | wc -l) GPU(s) visible"'

source "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_pretrain/train.sh"
