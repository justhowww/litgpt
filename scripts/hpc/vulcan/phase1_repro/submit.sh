#!/bin/bash
# Phase 1 -- small AVC-LM reproduction on UMD Vulcan (2x a5000, FSDP). ~1000 videos,
# ~85M model (12/768/12), AR-only, NO encodings (AVC-LM-style), within-video split. Signal: training-set
# eval (--train-split-file OUT_DIR/train_split.json) + within-video eval. STEPS=1M ceiling,
# resumable. 2 GPUs shard the 85M @ 16384 model to fit 24 GB a5000s. SLURM placement is in
# train.sbatch (vulcan-ampere / vulcan-medium / vulcan-metzler, 2 GPUs / 3-day walltime).
# Run on a Vulcan login node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm"}

# ~85M model (head_dim 64). N_EMBD=896 -> ~115M.
export N_LAYER=${N_LAYER:-12}
export N_EMBD=${N_EMBD:-768}
export N_HEAD=${N_HEAD:-12}
export DEVICES=${DEVICES:-2}                     # 2x a5000 FSDP; keep in sync with --gres
# AVC-LM-faithful: NO additive region/offset-id encodings (--no-region-id --no-offset-id).
export NO_ENCODING=${NO_ENCODING:-1}

export MAX_ROWS=${MAX_ROWS:-1000}               # 1000 videos
export STEPS=${STEPS:-1000000}                  # ceiling; resumable
export WARMUP_STEPS=${WARMUP_STEPS:-2000}
# global_batch_size is the TOTAL effective batch across both GPUs (FSDP data-parallel).
# 64 keeps the effective batch of a 1-GPU run while training ~2x faster. To DOUBLE the
# effective batch, set GLOBAL_BATCH_SIZE=128 (consider a higher LR / more warmup then).
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}  # per-GPU; raise once 'peak mem' shows headroom
export VAL_FRACTION=${VAL_FRACTION:-0.01}
export EVAL_INTERVAL=${EVAL_INTERVAL:-250}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
# no region/offset encodings (AVC-LM-style); AR-only (train.sh sets --p-fim 0);
# within-video (SPLIT_BY_VIDEO unset).
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-1000}
export FREE_RUN_TEMP=${FREE_RUN_TEMP:-0.0}
export FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
export FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-288}
export FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-144}

export MODEL_TAG=${MODEL_TAG:-byte-phase1-repro-1000v-85m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
mkdir -p "${OUT_DIR}/logs"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL
    --output="${OUT_DIR}/logs/%x-%j.out" --error="${OUT_DIR}/logs/%x-%j.err")
[[ -n "${EXCLUDE_NODES:-}" ]] && sbatch_args+=(--exclude="${EXCLUDE_NODES}")

echo "[phase1-repro/vulcan] ${MAX_ROWS} videos / ~85M (${N_LAYER}/${N_EMBD}/${N_HEAD}) / ${DEVICES}x a5000 / no_encoding=${NO_ENCODING} / steps=${STEPS}"
echo "  OUT_DIR=${OUT_DIR}"
job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train.sbatch")
echo "Submitted phase1-repro job ${job_id}"
