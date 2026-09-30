#!/bin/bash
# Global-signal ablation for one MEGABYTE checkpoint (scripts/byte/eval/megabyte_global_ablation.py).
#
#   RUN_DIR=/path/to/runs/<run> CHECKPOINT=step-00200000 \
#     sbatch scripts/hpc/zaratan/jpeglm_pretrain/eval_global_ablation_a100.sbatch
#
# Writes RUN_DIR/eval_global_ablation/<CHECKPOINT>.json (never overwrites).

#SBATCH --job-name=megabyte-global-ablation
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

: "${RUN_DIR:?Set RUN_DIR to the training run directory}"
CHECKPOINT=${CHECKPOINT:-latest}
NUM_VIDEOS=${NUM_VIDEOS:-200}
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

if [[ ! -r "${RUN_DIR}/${CHECKPOINT}/lit_model.pth" ]]; then
    echo "Missing checkpoint: ${RUN_DIR}/${CHECKPOINT}/lit_model.pth" >&2
    exit 1
fi

cd "${REPO_ROOT}"
srun python -u scripts/byte/eval/megabyte_global_ablation.py \
    "${RUN_DIR}/${CHECKPOINT}" \
    --train-split-file "${RUN_DIR}/train_split.json" \
    --num-videos "${NUM_VIDEOS}" \
    --out "${RUN_DIR}/eval_global_ablation/${CHECKPOINT}.json"
