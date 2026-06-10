#!/bin/bash
# Short integration check for PSM (+EOS) mixed AR/FIM sample construction.

#SBATCH --job-name=byte-fim-psm-smoke
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"
: "${NAL_INDEX:?Set NAL_INDEX to the persistent SQLite NAL index}"

BLOCK_SIZE=${BLOCK_SIZE:-16384}
MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-100}
P_FIM=${P_FIM:-0.5}
FIM_FORMAT=${FIM_FORMAT:-psm}
USE_EOS=${USE_EOS:-1}
if [[ "${USE_EOS}" == "1" ]]; then
    EOS_TAG="-eos"
else
    EOS_TAG=""
fi
MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm${EOS_TAG}-smoke}
OUT_DIR=${OUT_DIR:-"$(dirname "${MANIFEST}")/runs/byte-stage2-fim-psm${EOS_TAG}-smoke"}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}

if [[ ! -x "${FFMPEG_BINARY}" ]]; then
    echo "FFmpeg executable is not available: ${FFMPEG_BINARY}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
    python scripts/train_byte_stage1.py
    "${MANIFEST}"
    --model-name "${MODEL_NAME}"
    --nal-index-path "${NAL_INDEX}"
    --out-dir "${OUT_DIR}"
    --block-size "${BLOCK_SIZE}"
    --steps 10
    --global-batch-size 8
    --micro-batch-size 1
    --precision bf16-mixed
    --logger-name tensorboard
    --num-workers 4
    --max-manifest-rows "${MAX_MANIFEST_ROWS}"
    --p-fim "${P_FIM}"
    --fim-format "${FIM_FORMAT}"
    --eval-interval 5
    --eval-iters 2
    --reconstruction-task both
    --reconstruction-eval-interval 10
    --reconstruction-eval-samples 1
    --reconstruction-max-target-bytes 1400
    --ffmpeg-binary "${FFMPEG_BINARY}"
    --save-interval 5
)
if [[ "${USE_EOS}" == "1" ]]; then
    cmd+=(--use-eos)
fi

srun "${cmd[@]}"
