#!/bin/bash
# 7B JPEG-LM pretraining on 2x4 A100 40GB while the 4xH100 job waits in queue.
# One task per A100 is required for FSDP. Resumes from and writes to the same
# OUT_DIR as the H100 run; set YIELD_TO_JOBID to that queued H100 job so this
# job cancels itself as soon as the H100 job starts running.

#SBATCH --job-name=jpeglm-7b-a100
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
export DEVICES=${DEVICES:-4}
export NUM_NODES=${NUM_NODES:-${SLURM_JOB_NUM_NODES:-2}}
# Final evaluation belongs to the H100 run; this job only bridges training.
export INLINE_FINAL_EVAL=0

echo "[gpu-check] nodes=${SLURM_JOB_NODELIST}"
srun --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-4}" bash -c \
    'echo "[gpu-check] host=$(hostname) rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} $(nvidia-smi -L | wc -l) GPU(s) visible"'

if [[ -n "${YIELD_TO_JOBID:-}" ]]; then
    # This job may itself have waited in queue; if the H100 job already started
    # (or already finished), the handover has happened and bridging is moot.
    state=$(squeue -h -j "${YIELD_TO_JOBID}" -o %T 2>/dev/null || true)
    if [[ "${state}" != "PENDING" ]]; then
        echo "[yield] job ${YIELD_TO_JOBID} is ${state:-no longer queued}; nothing to bridge, exiting"
        exit 0
    fi
    (
        while sleep "${YIELD_POLL_SEC:-60}"; do
            state=$(squeue -h -j "${YIELD_TO_JOBID}" -o %T 2>/dev/null || true)
            if [[ -z "${state}" ]]; then
                echo "[yield] job ${YIELD_TO_JOBID} left the queue; no longer watching"
                exit 0
            fi
            if [[ "${state}" == "RUNNING" ]]; then
                echo "[yield] job ${YIELD_TO_JOBID} is running; cancelling ${SLURM_JOB_ID} to release ${OUT_DIR}"
                scancel "${SLURM_JOB_ID}"
                exit 0
            fi
        done
    ) &
    echo "[yield] will cancel this job once job ${YIELD_TO_JOBID} starts running"
fi

source "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_pretrain/train.sh"
