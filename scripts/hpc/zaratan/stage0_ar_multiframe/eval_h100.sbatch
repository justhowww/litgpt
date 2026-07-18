#!/bin/bash
# Stage 0 multi-frame AR (H0) clean-prefix continuation evaluation.
#
# Post-hoc probe for the Stage 0 run: give the model the first N frames of a
# stream window, free-run generate the next M frames, strict-decode, and compare
# against the ground-truth continuation (plus an intra IDR-from-scratch mode).
# See scripts/byte/eval/eval_ar_continuation.py and 0616.md (H0).

#SBATCH --job-name=byte-stage0-cont-eval
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
RUN_NAME=${RUN_NAME:-"byte-stage0-ar-multiframe"}
RUN_DIR=${RUN_DIR:-"${STAGED_CORPUS}/runs/${RUN_NAME}"}
MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}
OUT_DIR=${OUT_DIR:-"${RUN_DIR}/continuation_eval"}

# Checkpoints to evaluate, as bare step numbers under RUN_DIR (step-XXXXXXXX).
# Override per run; these are placeholders and the guard below fails loudly if a
# named checkpoint does not exist yet.
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"2000 4000 6000"}

# Continuation probe geometry (see eval_ar_continuation.py defaults).
NUM_CLIPS=${NUM_CLIPS:-20}
NUM_VISUALIZATIONS=${NUM_VISUALIZATIONS:-8}
PREFIX_FRAMES=${PREFIX_FRAMES:-8}
CONT_FRAMES=${CONT_FRAMES:-4}
MAX_WINDOW_BYTES=${MAX_WINDOW_BYTES:-16384}
MAX_GEN_MULTIPLE=${MAX_GEN_MULTIPLE:-2.0}
MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-0}
TEMPERATURE=${TEMPERATURE:-0.0}
SEED=${SEED:-42}
VIZ_FPS=${VIZ_FPS:-6}
TIMEOUT_SEC=${TIMEOUT_SEC:-60}
# Intra (IDR-from-scratch) mode is on by default in the eval; set EVAL_INTRA=0 to
# restrict to clean-prefix continuation only.
EVAL_INTRA=${EVAL_INTRA:-1}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    exit 1
fi
if [[ ! -x "${FFMPEG_BINARY}" ]]; then
    echo "FFmpeg executable is not available: ${FFMPEG_BINARY}" >&2
    exit 1
fi

checkpoint_dirs=()
for step in ${CHECKPOINT_STEPS}; do
    printf -v step_dir "step-%08d" "${step}"
    checkpoint_dir="${RUN_DIR}/${step_dir}"
    if [[ ! -r "${checkpoint_dir}/lit_model.pth" ]]; then
        echo "Checkpoint is missing: ${checkpoint_dir}/lit_model.pth" >&2
        exit 1
    fi
    checkpoint_dirs+=("${checkpoint_dir}")
done

mkdir -p "${OUT_DIR}"
echo "Repository: ${REPO_ROOT}"
echo "Run dir: ${RUN_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Checkpoints: ${checkpoint_dirs[*]}"
echo "NUM_CLIPS=${NUM_CLIPS} PREFIX_FRAMES=${PREFIX_FRAMES} CONT_FRAMES=${CONT_FRAMES} TEMPERATURE=${TEMPERATURE} EVAL_INTRA=${EVAL_INTRA}"

intra_flag="--eval-intra"
if [[ "${EVAL_INTRA}" == "0" ]]; then
    intra_flag="--no-eval-intra"
fi

srun python scripts/byte/eval/eval_ar_continuation.py \
    "${MANIFEST}" \
    --nal-index-path "${NAL_INDEX}" \
    --checkpoint-dirs "${checkpoint_dirs[@]}" \
    --out-dir "${OUT_DIR}" \
    --num-clips "${NUM_CLIPS}" \
    --num-visualizations "${NUM_VISUALIZATIONS}" \
    --prefix-frames "${PREFIX_FRAMES}" \
    --cont-frames "${CONT_FRAMES}" \
    --max-window-bytes "${MAX_WINDOW_BYTES}" \
    --max-gen-multiple "${MAX_GEN_MULTIPLE}" \
    --max-manifest-rows "${MAX_MANIFEST_ROWS}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --viz-fps "${VIZ_FPS}" \
    --timeout-sec "${TIMEOUT_SEC}" \
    --ffmpeg-binary "${FFMPEG_BINARY}" \
    "${intra_flag}"

echo "Stage 0 continuation evaluation complete: ${OUT_DIR}"
