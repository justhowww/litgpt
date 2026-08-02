#!/usr/bin/env bash
#SBATCH --job-name=bscv-storage-pilot
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

FFMPEG_MODULE=${FFMPEG_MODULE:-ffmpeg}
if command -v module >/dev/null 2>&1; then
    module load "${FFMPEG_MODULE}" || true
fi
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null

: "${SOURCE_ROOT:?Set SOURCE_ROOT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${CONFIG:?Set CONFIG}"

SAMPLE_SIZE=${SAMPLE_SIZE:-100}
SEED=${SEED:-42}
PROJECTION_VIDEOS=${PROJECTION_VIDEOS:-1000000}
ENCODE_TIMEOUT=${ENCODE_TIMEOUT:-1800}

cd "${REPO_ROOT}"
srun python scripts/byte/reports/report_bscv_storage_pilot.py \
    "${SOURCE_ROOT}" \
    "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --sample-size "${SAMPLE_SIZE}" \
    --seed "${SEED}" \
    --gop-counts 1 2 4 \
    --projection-videos "${PROJECTION_VIDEOS}" \
    --encode-timeout "${ENCODE_TIMEOUT}"
