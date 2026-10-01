#!/bin/bash
# Experiment E: regenerate from a cut to the end of its slice
# (scripts/byte/eval/eval_slice_end.py).
#
#   RUN_DIR=/path/to/runs/<run> CHECKPOINT=latest \
#     sbatch scripts/hpc/zaratan/jpeglm_pretrain/eval_slice_end_a100.sbatch
#
# Smoke test (2 holes, 1 sample, ~minutes):
#   RUN_DIR=... LIMIT_HOLES=2 SAMPLES_PER_HOLE=1 OUT_NAME=smoke \
#     sbatch --time=0-01:00:00 scripts/hpc/zaratan/jpeglm_pretrain/eval_slice_end_a100.sbatch
#
# Held-out holes are pinned corpus-wide (STAGED_CORPUS/eval_sets/), so every run
# and checkpoint is scored on the same holes; train-split holes are per run.
# Resumable: resubmitting with the same OUT_NAME skips finished generations.
# Writes RUN_DIR/eval_slice_end/<OUT_NAME or CHECKPOINT>[-train]/.

#SBATCH --job-name=slice-end-regen
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

: "${RUN_DIR:?Set RUN_DIR to the training run directory}"
CHECKPOINT=${CHECKPOINT:-latest}
SPLIT=${SPLIT:-val}
OUT_NAME=${OUT_NAME:-"${CHECKPOINT}"}
SAMPLES_PER_HOLE=${SAMPLES_PER_HOLE:-3}
HOLES_PER_STRATUM=${HOLES_PER_STRATUM:-20}
LIMIT_HOLES=${LIMIT_HOLES:-0}
CONDITIONS=${CONDITIONS:-"unmasked masked_old masked_new random_legal"}
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

STAGED_CORPUS=${STAGED_CORPUS:-"${PROJECT_SCRATCH}/OpenVid-1M_Data/data-jpeglm"}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}
if [[ ! -x "${FFMPEG_BINARY}" ]]; then
    echo "ffmpeg not executable: ${FFMPEG_BINARY}" >&2
    exit 1
fi
if [[ ! -r "${RUN_DIR}/${CHECKPOINT}/lit_model.pth" ]]; then
    echo "Missing checkpoint: ${RUN_DIR}/${CHECKPOINT}/lit_model.pth" >&2
    exit 1
fi

if [[ "${SPLIT}" == "val" ]]; then
    HOLES_FILE=${HOLES_FILE:-"${STAGED_CORPUS}/eval_sets/slice_end_v1_val.json"}
    OUT_DIR="${RUN_DIR}/eval_slice_end/${OUT_NAME}"
else
    HOLES_FILE=${HOLES_FILE:-"${RUN_DIR}/eval_slice_end/holes_v1_train.json"}
    OUT_DIR="${RUN_DIR}/eval_slice_end/${OUT_NAME}-train"
fi

cd "${REPO_ROOT}"
# shellcheck disable=SC2086
srun python -u scripts/byte/eval/eval_slice_end.py \
    "${RUN_DIR}/${CHECKPOINT}" \
    --train-split-file "${RUN_DIR}/train_split.json" \
    --split "${SPLIT}" \
    --holes-file "${HOLES_FILE}" \
    --holes-per-stratum "${HOLES_PER_STRATUM}" \
    --samples-per-hole "${SAMPLES_PER_HOLE}" \
    --limit-holes "${LIMIT_HOLES}" \
    --conditions ${CONDITIONS} \
    --ffmpeg-binary "${FFMPEG_BINARY}" \
    --out-dir "${OUT_DIR}"
