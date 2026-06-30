#!/bin/bash
# Submit the 1.3 B Stage 0 multi-frame AR (H0-1p3b) run on N GPUs of a chosen type.
# Generalizes submit_h100_4gpu.sh / submit_a100_4gpu.sh to any (type x count).
#
#   GPU_TYPE=h100|a100   (default h100)
#   NGPU=<int>           (default 4; must be <= GPUs-per-node, 4 on Zaratan)
#
# Example:
#   GPU_TYPE=a100 NGPU=2 FREE_RUN_INTERVAL=0 \
#   OUT_DIR=$HOME/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage0-1p3b-smoke \
#   bash scripts/hpc/zaratan/stage0_ar_multiframe_1p3b/submit_multigpu.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

GPU_TYPE=${GPU_TYPE:-h100}
NGPU=${NGPU:-4}

case "${GPU_TYPE}" in
    h100|a100) ;;
    *) echo "GPU_TYPE must be h100 or a100 (got '${GPU_TYPE}')" >&2; exit 1 ;;
esac
if ! [[ "${NGPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NGPU must be a positive integer (got '${NGPU}')" >&2; exit 1
fi

export JOB_SCRIPT="${SCRIPT_DIR}/train_multigpu.sbatch"
# One rank per GPU; FSDP shards across them. DEVICES is read by train.sh.
export DEVICES="${NGPU}"
# Override the job script's GRES + task count on the sbatch CLI (handled in submit.sh).
export SBATCH_OVERRIDES="--gpus=${GPU_TYPE}:${NGPU} --ntasks-per-node=${NGPU}"

echo "[multigpu] GPU_TYPE=${GPU_TYPE} NGPU=${NGPU} (1 rank/GPU, single node)"
exec bash "${SCRIPT_DIR}/submit.sh"
