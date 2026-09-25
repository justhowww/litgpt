#!/bin/bash
# Isolated small-sample job: two A100s train, then one GPU per eval split.

#SBATCH --job-name=jpeglm-small
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00

set -euo pipefail

: "${SMALL_SAMPLE_CONFIG:?Missing YAML config path}"
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
export DEVICES=2 NUM_NODES=1

echo "[small-sample] two-A100 job; config=${SMALL_SAMPLE_CONFIG} eval_only=${SMALL_SAMPLE_EVAL_ONLY:-0}"
srun --ntasks=2 --ntasks-per-node=2 bash -c \
    'echo "[gpu-check] rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"'

if [[ "${SMALL_SAMPLE_EVAL_ONLY:-0}" == "1" ]]; then
    echo "[small-sample] evaluation only; reusing ${OUT_DIR}/${SMALL_SAMPLE_EVAL_CHECKPOINT:-final}"
else
    source "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_pretrain/train.sh"
fi

EVAL_CHECKPOINT=${SMALL_SAMPLE_EVAL_CHECKPOINT:-final}
if [[ ! "${EVAL_CHECKPOINT}" =~ ^(final|step-[0-9]{8})$ ]]; then
    echo "SMALL_SAMPLE_EVAL_CHECKPOINT must be final or step-XXXXXXXX" >&2
    exit 1
fi
export EVAL_CHECKPOINT
if [[ ! -r "${OUT_DIR}/${EVAL_CHECKPOINT}/lit_model.pth" ]]; then
    echo "Checkpoint is not readable: ${OUT_DIR}/${EVAL_CHECKPOINT}/lit_model.pth" >&2
    exit 1
fi
if [[ ! -r "${OUT_DIR}/train_split.json" ]]; then
    echo "Training split is missing: ${OUT_DIR}/train_split.json" >&2
    exit 1
fi

if [[ "${SMALL_SAMPLE_EVAL_ONLY:-0}" != "1" ]]; then
    python "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_small_sample/write_learning_curve_aliases.py" "${OUT_DIR}"
fi

echo "[small-sample] teacher-forced evaluation: rank 0=train, rank 1=val"
srun \
    --exclusive \
    --kill-on-bad-exit=1 \
    --ntasks=2 \
    --ntasks-per-node=2 \
    --gpus-per-task=a100:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --output="${OUT_DIR}/logs/small-sample-eval-%t.out" \
    --error="${OUT_DIR}/logs/small-sample-eval-%t.err" \
    bash -c '
        split=train
        if [[ "${SLURM_PROCID}" == "1" ]]; then split=val; fi
        exec python -u "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_small_sample/eval_tf.py" \
            "${SMALL_SAMPLE_CONFIG}" --split "${split}" --checkpoint "${EVAL_CHECKPOINT}"
    '

echo "[small-sample] evaluation complete"
