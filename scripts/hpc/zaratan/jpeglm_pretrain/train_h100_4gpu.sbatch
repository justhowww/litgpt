#!/bin/bash
# 7B JPEG-LM pretraining allocation. One task per H100 is required for FSDP.

#SBATCH --job-name=jpeglm-7b
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=3-00:00:00
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

if [[ "${INLINE_FINAL_EVAL:-0}" == "1" ]]; then
    final_checkpoint="${OUT_DIR}/final/lit_model.pth"
    if [[ ! -r "${final_checkpoint}" ]]; then
        echo "Inline final evaluation requested, but checkpoint is missing: ${final_checkpoint}" >&2
        exit 1
    fi

    export RUN_NAME="${MODEL_TAG}"
    export RUN_DIR="${OUT_DIR}"
    export CHECKPOINT_NAMES=final
    export EVAL_TAG=${INLINE_EVAL_TAG:-corrupt-gen-frame-final}
    export NUM_CLIPS=${INLINE_EVAL_NUM_CLIPS:-20}
    export NUM_VISUALIZATIONS=${INLINE_EVAL_NUM_VISUALIZATIONS:-8}
    export SKIP_COMPLETED_EVAL=1

    mkdir -p "${OUT_DIR}/logs"
    echo "[jpeglm] starting inline final FIM evaluation on four H100s"
    echo "[jpeglm] task map: 0=train/unmasked 1=train/masked 2=val/unmasked 3=val/masked"
    echo "[jpeglm] evaluation tag=${EVAL_TAG} clips=${NUM_CLIPS} visualizations=${NUM_VISUALIZATIONS}"

    srun \
        --exclusive \
        --kill-on-bad-exit=1 \
        --ntasks=4 \
        --ntasks-per-node=4 \
        --gpus-per-task=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
        --output="${OUT_DIR}/logs/jpeglm-inline-eval-%t.out" \
        --error="${OUT_DIR}/logs/jpeglm-inline-eval-%t.err" \
        bash -c \
        'export EVAL_TASK_ID=${SLURM_PROCID}; exec bash "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_pretrain/eval_fim_task.sh"'

    echo "[jpeglm] inline final FIM evaluation complete"
fi
