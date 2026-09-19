#!/bin/bash
# Allocate one H100 for one JPEG-LM FIM evaluation quadrant.

#SBATCH --job-name=jpeglm-fim-eval
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
export EVAL_TASK_ID=${SLURM_ARRAY_TASK_ID}

exec srun --unbuffered bash \
    "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_pretrain/eval_fim_task.sh"
