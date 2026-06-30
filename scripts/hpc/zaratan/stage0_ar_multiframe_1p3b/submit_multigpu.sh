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
# Override the job script's GRES + task count on the sbatch CLI (handled in
# submit.sh). A config-specific job name makes concurrent variants (e.g. a race)
# distinguishable in squeue. Override with JOB_NAME=... if desired.
JOB_NAME=${JOB_NAME:-"byte-stage0-1p3b-${GPU_TYPE}-${NGPU}gpu"}

# Host RAM scales ~per-rank: each rank runs its own dataloaders and loads the
# manifest/NAL-index, plus a rank-0 spike when the fp32 state is gathered for
# checkpointing. A full 1-GPU run measured 61.78 GB peak, so default to ~56 GB
# per rank + 16 GB headroom. NUM_WORKERS is per-rank; 4 is plenty once NGPU>1 and
# meaningfully cuts host RAM vs the single-GPU default of 8. Tune MEM/NUM_WORKERS
# after `seff`-ing a real multi-GPU run; all three are overridable via env.
MEM=${MEM:-"$(( NGPU * 56 + 16 ))G"}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
export NUM_WORKERS=${NUM_WORKERS:-4}
export SBATCH_OVERRIDES="--gpus=${GPU_TYPE}:${NGPU} --ntasks-per-node=${NGPU} --job-name=${JOB_NAME} --mem=${MEM} --cpus-per-task=${CPUS_PER_TASK}"

echo "[multigpu] GPU_TYPE=${GPU_TYPE} NGPU=${NGPU} mem=${MEM} cpus/task=${CPUS_PER_TASK} workers/rank=${NUM_WORKERS} (1 rank/GPU, single node)"
exec bash "${SCRIPT_DIR}/submit.sh"
