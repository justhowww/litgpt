#!/bin/bash
# Free-run FIM repair on the fixed holes for one completed small-sample checkpoint.

#SBATCH --job-name=jpeglm-freerun
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

: "${SMALL_SAMPLE_CONFIG:?Set SMALL_SAMPLE_CONFIG to the frozen run YAML}"
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

CHECKPOINT=${SMALL_SAMPLE_CHECKPOINT:-final}
SPLIT=${SMALL_SAMPLE_SPLIT:-train}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}
FFPROBE_BINARY=${FFPROBE_BINARY:-"${CONDA_ROOT}/bin/ffprobe"}

for binary in "${FFMPEG_BINARY}" "${FFPROBE_BINARY}"; do
    if [[ ! -x "${binary}" ]]; then
        echo "Required FFmpeg executable is not available: ${binary}" >&2
        exit 1
    fi
    if ! "${binary}" -version >/dev/null 2>&1; then
        echo "Required FFmpeg executable cannot run: ${binary}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
srun python -u scripts/hpc/zaratan/jpeglm_small_sample/eval_free_run.py \
    "${SMALL_SAMPLE_CONFIG}" --checkpoint "${CHECKPOINT}" --split "${SPLIT}" \
    --ffmpeg-binary "${FFMPEG_BINARY}" --ffprobe-binary "${FFPROBE_BINARY}"
