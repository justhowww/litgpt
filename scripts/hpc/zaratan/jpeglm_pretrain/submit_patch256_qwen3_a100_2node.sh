#!/bin/bash
# Continue the five-epoch Qwen3 patch-256 run on 2x4 A100 40GB while its
# 4xH100 job waits in queue. Same OUT_DIR and hyperparameters; only the
# per-GPU microbatch shrinks, and gradient accumulation keeps the global batch.
#
#   YIELD_TO_JOBID=<queued H100 job id> bash submit_patch256_qwen3_a100_2node.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${YIELD_TO_JOBID:?Set YIELD_TO_JOBID to the queued H100 job this run hands over to}"
export YIELD_TO_JOBID

export JOB_SCRIPT="${SCRIPT_DIR}/train_a100_2node.sbatch"
# The 4xH100 run peaks at ~82 GB with microbatch 8. Eight-way FSDP halves the
# sharded state, and microbatch 2 quarters activations, which fits 40 GB cards.
# 64 / (2 x 8 ranks) = 4 accumulation steps per optimizer step.
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
# Bound work lost at handover; the H100 job restarts from the latest checkpoint.
export LATEST_SAVE_INTERVAL=${LATEST_SAVE_INTERVAL:-5000}
export INLINE_FINAL_EVAL=0

exec bash "${SCRIPT_DIR}/submit_patch256_qwen3_fim_p100_5ep.sh"
