#!/bin/bash
# Read-only syntax-prior audit for one completed small-sample checkpoint.

#SBATCH --job-name=jpeglm-syntax
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00

set -euo pipefail

: "${SMALL_SAMPLE_CONFIG:?Set SMALL_SAMPLE_CONFIG to the frozen run YAML}"
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
CHECKPOINT=${SMALL_SAMPLE_CHECKPOINT:-final}
SPLIT=${SMALL_SAMPLE_SPLIT:-val}

cd "${REPO_ROOT}"
srun python -u scripts/hpc/zaratan/jpeglm_small_sample/eval_syntax_prior.py \
    "${SMALL_SAMPLE_CONFIG}" --checkpoint "${CHECKPOINT}" --split "${SPLIT}"
