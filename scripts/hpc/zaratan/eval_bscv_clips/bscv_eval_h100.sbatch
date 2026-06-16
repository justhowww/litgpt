#!/bin/bash
# Whole-clip BSCV-style byte reconstruction evaluation.

#SBATCH --job-name=byte-bscv-eval
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
RUN_NAME=${RUN_NAME:-"byte-stage2-fim-psm-ce-only-within-video-1k-8k"}
RUN_DIR=${RUN_DIR:-"${STAGED_CORPUS}/runs/${RUN_NAME}"}
MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}
OUT_ROOT=${OUT_ROOT:-"${RUN_DIR}/offline_bscv_eval"}

CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"5000 5500 6000 6500"}
NUM_CLIPS=${NUM_CLIPS:-20}
NUM_VISUALIZATIONS=${NUM_VISUALIZATIONS:-8}
# BSCV only ever evaluates corr_prob 1 (and one config at 2), so we reproduce the
# faithful single-hit point rather than the synthetic 4/8 density stress test.
CORR_PROB_SWEEP=${CORR_PROB_SWEEP:-"1"}
# random | early (worst-case forward propagation) | late (minimal propagation).
GOP_POSITION_MODE=${GOP_POSITION_MODE:-random}
CORR_POS=${CORR_POS:-0.4}
CORR_LEN_BYTES=${CORR_LEN_BYTES:-2048}
MAX_SPANS_PER_CLIP=${MAX_SPANS_PER_CLIP:-0}
MAX_FRAMES=${MAX_FRAMES:-96}
BLOCK_SIZE=${BLOCK_SIZE:-16384}
NUM_REF_SLICES=${NUM_REF_SLICES:-1}
# P-only by default: the checkpoint is FIM-trained on P-slices (nal type 1) only, so
# corrupting the IDR is out-of-distribution and unfair to the model. "1" covers both
# 0x41 and 0x01 non-IDR slices. Set "1 5" for the keyframe-inclusive BSCV-faithful
# variant (reproduces BSCV's ~6-12% catastrophic IDR-hit tail).
# corr_len 2048 BYTES = BSCV's 4096 hex chars = their `_144096` config.
TARGET_NAL_TYPES=${TARGET_NAL_TYPES:-"1"}
SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-0}
TIMEOUT_SEC=${TIMEOUT_SEC:-60}

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

mkdir -p "${OUT_ROOT}"
echo "Repository: ${REPO_ROOT}"
echo "Run dir: ${RUN_DIR}"
echo "Output root: ${OUT_ROOT}"
echo "Checkpoints: ${checkpoint_dirs[*]}"
echo "NUM_CLIPS=${NUM_CLIPS} CORR_PROB_SWEEP=${CORR_PROB_SWEEP} GOP_POSITION_MODE=${GOP_POSITION_MODE} TARGET_NAL_TYPES=${TARGET_NAL_TYPES} CORR_LEN_BYTES=${CORR_LEN_BYTES} CORR_POS=${CORR_POS}"

for corr_prob in ${CORR_PROB_SWEEP}; do
    out_dir="${OUT_ROOT}/corrprob_${corr_prob}_${GOP_POSITION_MODE}"
    mkdir -p "${out_dir}"
    echo "=== corr_prob=${corr_prob} mode=${GOP_POSITION_MODE} -> ${out_dir} ==="
    srun python scripts/byte/eval/eval_bscv_clips.py \
        "${MANIFEST}" \
        --nal-index-path "${NAL_INDEX}" \
        --checkpoint-dirs "${checkpoint_dirs[@]}" \
        --out-dir "${out_dir}" \
        --num-clips "${NUM_CLIPS}" \
        --num-visualizations "${NUM_VISUALIZATIONS}" \
        --corr-prob "${corr_prob}" \
        --corr-pos "${CORR_POS}" \
        --corr-len-bytes "${CORR_LEN_BYTES}" \
        --max-spans-per-clip "${MAX_SPANS_PER_CLIP}" \
        --max-frames "${MAX_FRAMES}" \
        --block-size "${BLOCK_SIZE}" \
        --num-ref-slices "${NUM_REF_SLICES}" \
        --target-nal-types ${TARGET_NAL_TYPES} \
        --gop-position-mode "${GOP_POSITION_MODE}" \
        --slice-header-guard-bytes "${SLICE_HEADER_GUARD_BYTES}" \
        --timeout-sec "${TIMEOUT_SEC}" \
        --ffmpeg-binary "${FFMPEG_BINARY}"
done

echo "Whole-clip BSCV-style evaluation complete: ${OUT_ROOT}"
