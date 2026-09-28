#!/bin/bash
# Run on a Zaratan login node: submit the sharded syntax-mask scan as a Slurm
# array plus a merge job that runs after all shards (even failed ones) end.
#
# Pilot (one shard, 2000 random files) to measure throughput:
#   PILOT=1 bash scripts/hpc/zaratan/syntax_mask/submit_scan.sh
# Full corpus:
#   NUM_SHARDS=32 bash scripts/hpc/zaratan/syntax_mask/submit_scan.sh
# Resubmit only some shards into the same run (they resume):
#   RUN_DIR=... ARRAY=3,17 NUM_SHARDS=32 bash scripts/hpc/zaratan/syntax_mask/submit_scan.sh
#
# Overrides: MANIFEST LAYOUT(frame|mb) NUM_SHARDS CPUS TIME RUN_DIR ARRAY
#            REFERENCE_RATE PROBE_RATE PROBE_LEN LIMIT MAX_MANIFEST_ROWS DUMP_DIR
#            SBATCH_ACCOUNT
# Masks for the small-sample runs (data.max_rows: 256):
#   MAX_MANIFEST_ROWS=256 NUM_SHARDS=1 CPUS=32 TIME=01:00:00 \
#   DUMP_DIR=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm/syntax_masks/first256 \
#   bash scripts/hpc/zaratan/syntax_mask/submit_scan.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
MANIFEST=${MANIFEST:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm/manifest.jsonl"}
LAYOUT=${LAYOUT:-frame}
CPUS=${CPUS:-64}
TIME=${TIME:-12:00:00}
if [[ "${PILOT:-0}" == 1 ]]; then
    NUM_SHARDS=1
    LIMIT=${LIMIT:-2000}
    TIME=${TIME_PILOT:-02:00:00}
    REFERENCE_RATE=${REFERENCE_RATE:-0.01}
    PROBE_RATE=${PROBE_RATE:-0.005}
fi
NUM_SHARDS=${NUM_SHARDS:-32}
ARRAY=${ARRAY:-"0-$((NUM_SHARDS - 1))"}
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR=${RUN_DIR:-"$(dirname "${MANIFEST}")/syntax_mask_scan/${STAMP}"}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest not readable: ${MANIFEST}" >&2
    exit 1
fi
mkdir -p "${RUN_DIR}"

export REPO_ROOT MANIFEST LAYOUT NUM_SHARDS RUN_DIR
export REFERENCE_RATE=${REFERENCE_RATE:-0.001}
export PROBE_RATE=${PROBE_RATE:-0.0005}
export PROBE_LEN=${PROBE_LEN:-64}
export LIMIT=${LIMIT:-0}
export MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-0}
export DUMP_DIR=${DUMP_DIR:-}

scan_id=$(
    sbatch --parsable --export=ALL --account="${SBATCH_ACCOUNT}" \
        --array="${ARRAY}" --cpus-per-task="${CPUS}" --time="${TIME}" \
        --chdir="${REPO_ROOT}" "${SCRIPT_DIR}/scan_array.sbatch"
)
merge_id=$(
    sbatch --parsable --export=ALL --account="${SBATCH_ACCOUNT}" \
        --dependency="afterany:${scan_id}" --chdir="${REPO_ROOT}" \
        "${SCRIPT_DIR}/merge.sbatch"
)
cat <<EOF
Scan array job: ${scan_id}  (shards ${ARRAY} of ${NUM_SHARDS}, ${CPUS} CPUs each, ${TIME})
Merge job:      ${merge_id} (after all shards)
Run dir:        ${RUN_DIR}
Logs:           ${REPO_ROOT}/slurm-syntax-mask-scan-${scan_id}_*.out
Progress:       tail -f ${REPO_ROOT}/slurm-syntax-mask-scan-${scan_id}_0.out
Result:         ${RUN_DIR}/summary.txt  and  summary.json
EOF
