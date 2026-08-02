#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-"$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"}

SOURCE_ROOT=${SOURCE_ROOT:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/video"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/bscv-storage-pilot"}
CONFIG=${CONFIG:-"${REPO_ROOT}/preprocessing/h264_preprocess_config_bscv.json"}
SAMPLE_SIZE=${SAMPLE_SIZE:-100}
SEED=${SEED:-42}
PROJECTION_VIDEOS=${PROJECTION_VIDEOS:-1000000}
ENCODE_TIMEOUT=${ENCODE_TIMEOUT:-1800}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
SBATCH_PARTITION=${SBATCH_PARTITION:-""}

for path in "${SOURCE_ROOT}" "${CONFIG}"; do
    if [[ ! -r "${path}" ]]; then
        echo "Required path is not readable: ${path}" >&2
        exit 1
    fi
done
mkdir -p "${OUTPUT_DIR}"

args=(
    --parsable
    --account="${SBATCH_ACCOUNT}"
    --export="ALL,REPO_ROOT=${REPO_ROOT},SOURCE_ROOT=${SOURCE_ROOT},OUTPUT_DIR=${OUTPUT_DIR},CONFIG=${CONFIG},SAMPLE_SIZE=${SAMPLE_SIZE},SEED=${SEED},PROJECTION_VIDEOS=${PROJECTION_VIDEOS},ENCODE_TIMEOUT=${ENCODE_TIMEOUT}"
)
if [[ -n "${SBATCH_PARTITION}" ]]; then
    args+=(--partition="${SBATCH_PARTITION}")
fi

job_id=$(sbatch "${args[@]}" "${SCRIPT_DIR}/pilot.sbatch")
echo "Submitted BSCV storage pilot job ${job_id}"
echo "  samples=${SAMPLE_SIZE} seed=${SEED}"
echo "  output=${OUTPUT_DIR}"
