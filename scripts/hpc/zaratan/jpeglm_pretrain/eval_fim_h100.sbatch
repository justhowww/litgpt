#!/bin/bash
# JPEG-LM FIM evaluation matrix: train/validation x unmasked/masked.

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
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}
RUN_NAME=${RUN_NAME:-"byte-jpeglm-7b-patch256-fim-p050-changing-fullseq-spanw1-eosaux1-1m"}
RUN_DIR=${RUN_DIR:-"${STAGED_CORPUS}/runs/${RUN_NAME}"}
MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
TRAIN_SPLIT_FILE=${TRAIN_SPLIT_FILE:-"${RUN_DIR}/train_split.json"}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}

CHECKPOINT_NAMES=${CHECKPOINT_NAMES:-"step-00100000 step-00200000 step-00300000 final"}
EVAL_TAG=${EVAL_TAG:-"corrupt-gen-frame"}
NUM_CLIPS=${NUM_CLIPS:-100}
NUM_VISUALIZATIONS=${NUM_VISUALIZATIONS:-8}
MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-0}
MAX_WINDOW_BYTES=${MAX_WINDOW_BYTES:-131071}
MAX_GEN_BYTES=${MAX_GEN_BYTES:-4096}
CORR_LEN_BYTES=${CORR_LEN_BYTES:-600}
CORR_POS=${CORR_POS:-0.4}
CORR_FRAME_TYPE=${CORR_FRAME_TYPE:-any}
CORR_HEADER_GUARD_BYTES=${CORR_HEADER_GUARD_BYTES:-0}
SEED=${SEED:-42}

case "${SLURM_ARRAY_TASK_ID}" in
    0) EVAL_SPLIT=train; MASK_TAG=unmasked; MASK_FLAG=() ;;
    1) EVAL_SPLIT=train; MASK_TAG=masked; MASK_FLAG=(--mask-illegal-bytes) ;;
    2) EVAL_SPLIT=val; MASK_TAG=unmasked; MASK_FLAG=() ;;
    3) EVAL_SPLIT=val; MASK_TAG=masked; MASK_FLAG=(--mask-illegal-bytes) ;;
    *) echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

for required in "${MANIFEST}" "${NAL_INDEX}" "${TRAIN_SPLIT_FILE}"; do
    if [[ ! -r "${required}" ]]; then
        echo "Required input is not readable: ${required}" >&2
        exit 1
    fi
done
if [[ ! -x "${FFMPEG_BINARY}" ]]; then
    echo "FFmpeg executable is not available: ${FFMPEG_BINARY}" >&2
    exit 1
fi

checkpoint_dirs=()
for checkpoint_name in ${CHECKPOINT_NAMES}; do
    checkpoint_dir="${RUN_DIR}/${checkpoint_name}"
    if [[ ! -r "${checkpoint_dir}/lit_model.pth" ]]; then
        echo "Checkpoint is missing: ${checkpoint_dir}/lit_model.pth" >&2
        exit 1
    fi
    checkpoint_dirs+=("${checkpoint_dir}")
done

OUT_DIR="${RUN_DIR}/eval_fim/${EVAL_TAG}/${EVAL_SPLIT}/${MASK_TAG}"
if [[ -e "${OUT_DIR}/summary.csv" || -e "${OUT_DIR}/metrics.jsonl" ]]; then
    echo "Evaluation output already exists; choose a new EVAL_TAG: ${OUT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"

echo "JPEG-LM FIM evaluation"
echo "  split=${EVAL_SPLIT} mask=${MASK_TAG} clips=${NUM_CLIPS}"
echo "  corruption=${CORR_LEN_BYTES}B position=${CORR_POS} frame_type=${CORR_FRAME_TYPE}"
echo "  checkpoints=${checkpoint_dirs[*]}"
echo "  output=${OUT_DIR}"

srun python -u scripts/byte/eval/eval_fim_avclm.py \
    "${MANIFEST}" \
    --nal-index-path "${NAL_INDEX}" \
    --checkpoint-dirs "${checkpoint_dirs[@]}" \
    --train-split-file "${TRAIN_SPLIT_FILE}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --eval-split "${EVAL_SPLIT}" \
    --hole-set sampled \
    --num-clips "${NUM_CLIPS}" \
    --num-visualizations "${NUM_VISUALIZATIONS}" \
    --max-manifest-rows "${MAX_MANIFEST_ROWS}" \
    --max-window-bytes "${MAX_WINDOW_BYTES}" \
    --window-min-frames 2 \
    --window-unit auto \
    --fim-format psm \
    --fim-loss-scope auto \
    --use-eos \
    --fim-min-gap 64 \
    --fim-max-gap 1400 \
    --slice-header-guard-bytes 0 \
    --hole-placement corrupt_gen_frame \
    --corr-len-bytes "${CORR_LEN_BYTES}" \
    --corr-pos "${CORR_POS}" \
    --corr-frame-type "${CORR_FRAME_TYPE}" \
    --corr-header-guard-bytes "${CORR_HEADER_GUARD_BYTES}" \
    --stop-modes learned_eos \
    --temperature 0 \
    --top-k 0 \
    --top-p 1.0 \
    --max-gen-bytes "${MAX_GEN_BYTES}" \
    --slice-layout frame \
    --seed "${SEED}" \
    --ffmpeg-binary "${FFMPEG_BINARY}" \
    --save-streams \
    "${MASK_FLAG[@]}"

echo "JPEG-LM FIM evaluation complete: ${OUT_DIR}"
