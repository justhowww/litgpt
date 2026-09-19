#!/bin/bash
# Run one quadrant of the JPEG-LM FIM evaluation matrix.
#
# Task mapping:
#   0: train, unmasked
#   1: train, masked
#   2: validation, unmasked
#   3: validation, masked
#
# This body is shared by the standalone Slurm array and the four-GPU inline
# post-training evaluation. The caller supplies EVAL_TASK_ID, or it is inferred
# from SLURM_ARRAY_TASK_ID / SLURM_PROCID.

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}
RUN_NAME=${RUN_NAME:-"byte-jpeglm-7b-patch256-fim-p050-changing-fullseq-spanw1-eosaux1-1m"}
RUN_DIR=${RUN_DIR:-"${STAGED_CORPUS}/runs/${RUN_NAME}"}
MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
TRAIN_SPLIT_FILE=${TRAIN_SPLIT_FILE:-"${RUN_DIR}/train_split.json"}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}

CHECKPOINT_NAMES=${CHECKPOINT_NAMES:-"step-00100000 step-00200000 step-00300000 final latest"}
EVAL_TAG=${EVAL_TAG:-"corrupt-gen-frame"}
NUM_CLIPS=${NUM_CLIPS:-20}
NUM_VISUALIZATIONS=${NUM_VISUALIZATIONS:-8}
MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-0}
MAX_WINDOW_BYTES=${MAX_WINDOW_BYTES:-131071}
MAX_GEN_BYTES=${MAX_GEN_BYTES:-4096}
CORR_LEN_BYTES=${CORR_LEN_BYTES:-600}
# Optional whitespace-separated severity schedule. When set, the evaluator uses
# one distinct clip per length and ignores the scalar CORR_LEN_BYTES value.
CORR_LEN_BYTES_LIST=${CORR_LEN_BYTES_LIST:-}
CORR_SAMPLES_PER_LENGTH=${CORR_SAMPLES_PER_LENGTH:-1}
CORR_POS=${CORR_POS:-0.4}
CORR_FRAME_TYPE=${CORR_FRAME_TYPE:-any}
CORR_HEADER_GUARD_BYTES=${CORR_HEADER_GUARD_BYTES:-0}
SEED=${SEED:-42}

EVAL_TASK_ID=${EVAL_TASK_ID:-${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-}}}
if [[ -z "${EVAL_TASK_ID}" ]]; then
    echo "Set EVAL_TASK_ID to one of 0, 1, 2, or 3" >&2
    exit 2
fi

case "${EVAL_TASK_ID}" in
    0) EVAL_SPLIT=train; MASK_TAG=unmasked; MASK_FLAG=() ;;
    1) EVAL_SPLIT=train; MASK_TAG=masked; MASK_FLAG=(--mask-illegal-bytes) ;;
    2) EVAL_SPLIT=val; MASK_TAG=unmasked; MASK_FLAG=() ;;
    3) EVAL_SPLIT=val; MASK_TAG=masked; MASK_FLAG=(--mask-illegal-bytes) ;;
    *) echo "Unexpected EVAL_TASK_ID=${EVAL_TASK_ID}; expected 0-3" >&2; exit 2 ;;
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
checkpoint_targets=()
for checkpoint_name in ${CHECKPOINT_NAMES}; do
    checkpoint_dir="${RUN_DIR}/${checkpoint_name}"
    if [[ ! -r "${checkpoint_dir}/lit_model.pth" ]]; then
        echo "Skipping unavailable checkpoint: ${checkpoint_dir}/lit_model.pth" >&2
        continue
    fi
    canonical_checkpoint_dir=$(cd -P -- "${checkpoint_dir}" && pwd)
    duplicate=0
    for existing_checkpoint_target in "${checkpoint_targets[@]}"; do
        if [[ "${existing_checkpoint_target}" == "${canonical_checkpoint_dir}" ]]; then
            duplicate=1
            break
        fi
    done
    if (( duplicate )); then
        echo "Skipping duplicate checkpoint target: ${checkpoint_dir} -> ${canonical_checkpoint_dir}" >&2
        continue
    fi
    checkpoint_dirs+=("${checkpoint_dir}")
    checkpoint_targets+=("${canonical_checkpoint_dir}")
done
if (( ${#checkpoint_dirs[@]} == 0 )); then
    echo "None of the requested checkpoints are readable under ${RUN_DIR}: ${CHECKPOINT_NAMES}" >&2
    exit 1
fi

corr_length_args=()
if [[ -n "${CORR_LEN_BYTES_LIST}" ]]; then
    read -r -a corr_length_values <<< "${CORR_LEN_BYTES_LIST}"
    if (( ${#corr_length_values[@]} == 0 )); then
        echo "CORR_LEN_BYTES_LIST did not contain any lengths" >&2
        exit 2
    fi
    if [[ ! "${CORR_SAMPLES_PER_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
        echo "CORR_SAMPLES_PER_LENGTH must be a positive integer" >&2
        exit 2
    fi
    NUM_CLIPS=$(( ${#corr_length_values[@]} * CORR_SAMPLES_PER_LENGTH ))
    corr_length_args=(--corr-len-bytes-list "${corr_length_values[@]}")
else
    corr_length_args=(--corr-len-bytes "${CORR_LEN_BYTES}")
fi

OUT_DIR="${RUN_DIR}/eval_fim/${EVAL_TAG}/${EVAL_SPLIT}/${MASK_TAG}"
if [[ -e "${OUT_DIR}/.complete" && "${SKIP_COMPLETED_EVAL:-0}" == "1" ]]; then
    echo "Evaluation already completed; skipping: ${OUT_DIR}"
    exit 0
fi
if [[ -e "${OUT_DIR}/summary.csv" || -e "${OUT_DIR}/metrics.jsonl" ]]; then
    echo "Evaluation output already exists; choose a new EVAL_TAG: ${OUT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"

echo "JPEG-LM FIM evaluation"
echo "  task=${EVAL_TASK_ID} split=${EVAL_SPLIT} mask=${MASK_TAG} clips=${NUM_CLIPS}"
echo "  corruption=${CORR_LEN_BYTES_LIST:-${CORR_LEN_BYTES}}B position=${CORR_POS} frame_type=${CORR_FRAME_TYPE}"
echo "  checkpoints=${checkpoint_dirs[*]}"
echo "  output=${OUT_DIR}"

cd "${REPO_ROOT}"
python -u scripts/byte/eval/eval_fim_avclm.py \
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
    "${corr_length_args[@]}" \
    --corr-samples-per-length "${CORR_SAMPLES_PER_LENGTH}" \
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

touch "${OUT_DIR}/.complete"
echo "JPEG-LM FIM evaluation complete: ${OUT_DIR}"
