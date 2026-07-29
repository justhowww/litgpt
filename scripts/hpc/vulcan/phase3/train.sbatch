#!/usr/bin/env bash
# UMD Vulcan allocation for phase 3's AR+FIM training body, on the vulcan-high QOS:
# 4x a6000, 16 cpus, 128G mem, 1-12:00:00 wall (that QOS's per-job caps). Sources
# phase2_fim/train.sh directly (not a copy) -- the training body (objective, dataset
# mode, FIM knobs) is identical between phase 2 and phase 3; only scale (arch, video
# count, device count) differs, and that's all set via env vars from submit.sh. A
# duplicated copy here would silently drift out of sync with phase 2's bug fixes.
# submit.sh may override --output/--error and --gres/--exclude.
#
# qos=vulcan-high is a JOB QoS (cpu=16, gres/gpu=4, mem=128G, MaxWall=1-12:00:00 --
# confirmed via `show_qos --all`), orthogonal to the partition; vulcan-ampere is the
# partition (the a6000 node pool). Confirmed correct, not a guess.

#SBATCH --job-name=byte-phase3
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-high
#SBATCH --account=vulcan-metzler
#SBATCH --time=1-12:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT           :-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT

# GPU-visibility sanity check -- fails fast (seconds) instead of a 30-minute NCCL
# rendezvous timeout if the node doesn't actually expose 4 GPUs to every task (the
# failure mode hit on the first phase-3 submission: 2/4 ranks got no visible device).
echo "[gpu-check] host=$(hostname)"
srun --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-1}" bash -c \
    'echo "[gpu-check] rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} $(nvidia-smi -L | wc -l) GPU(s) visible"'

source "${REPO_ROOT}/scripts/hpc/vulcan/phase2_fim/train.sh"
