#!/bin/bash
# Phase 3 -- AR + masked-span FIM at scale, on UMD Vulcan (4x a6000, FSDP, vulcan-high).
# Same objective as phase 2 (P_FIM=0.5, FIM_FORMAT=psm, USE_EOS=1); the change here is
# scale: model and corpus both grow, sized from two empirical results, not guesses:
#
#   Model size (27 layers / 1024 embd / 16 head, ~341M params): chosen by bisecting
#   peak GPU memory at block_size=16384, micro_batch=1, 2x a6000 FSDP -- 16L/202M used
#   21.4GB, 24L/303M used 37.1GB (safe), 30L/378M and 32L/404M both hit ~48-50GB (over
#   budget -- the card's real ceiling, not a smooth continuation of the trend). 27L/341M
#   measured 39.5GB, ~8.5GB of margin on TWO GPUs. On four (this run), FSDP shards
#   weights/opt/grads across twice as many ranks, so per-GPU headroom is larger still.
#
#   Video count (45,000, rounding up from a ~42k estimate): phase 1's real anchor is
#   1000 videos / 85M params; phase 3's own road-list spec is 150,000 videos / 300M-1B.
#   Log-linear interpolation between those two anchors, evaluated at 341M params, gives
#   ~42k videos -- i.e. data needs to grow much faster than params for this task, not
#   proportionally (consistent with phase 1's train/val gap on residual coding being a
#   DIVERSITY problem, not a token-budget one -- see 260712 - phase 1 result.md). The
#   staged corpus has 128,440 videos, so 45k needs no new data collection.
#
# Signal to read this against (same three as phase 2, at a MATCHED STEP against phase 1
# where relevant): val_loss_ar / training-set free-run (did AR regress), within-video
# eval (ok to fail -- 45k videos may still lack syntax diversity), val_loss_fim (is FIM
# learning). The one NEW thing worth checking that phase 1/2 couldn't: whether the
# train/val gap on residual coding (coeff_token, level, total_zeros -- see phase 1's
# result doc) shrinks relative to phase 1's 1000-video baseline. That is the actual
# test of whether 45k videos bought real generalization, not just more training tokens.
#
# Run on a Vulcan login node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm"}

# --- Model arch: 27/1024/16 (~341M), sized by the memory-probe bisection above -----
export N_LAYER=${N_LAYER:-27}
export N_EMBD=${N_EMBD:-1024}
export N_HEAD=${N_HEAD:-16}
export DEVICES=${DEVICES:-4}                     # 4x a6000 FSDP; keep in sync with --gres
export NO_ENCODING=${NO_ENCODING:-1}             # AVC-LM-faithful; also required by window FIM
export NUM_WORKERS=${NUM_WORKERS:-4}             # 16 cpus / 4 tasks

export MAX_ROWS=${MAX_ROWS:-45000}               # 45,000 videos -- see derivation above
export STEPS=${STEPS:-1000000}                   # real LR horizon (cosine anneals over this)
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
# Per-GPU micro_batch=1 is the proven-safe value from the memory probes (27L peaked
# 39.5GB on 2 GPUs at micro=1; do not raise this without re-probing -- the 30L/32L
# probes showed the failure mode is a hard ceiling, not a graceful degradation).
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export VAL_FRACTION=${VAL_FRACTION:-0.01}
export EVAL_INTERVAL=${EVAL_INTERVAL:-250}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-1000}
export FREE_RUN_TEMP=${FREE_RUN_TEMP:-0.0}
export FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
# REAL frames (first_mb_in_slice == 0), not macroblocks -- see phase2_fim/submit.sh's
# note on the stale 288/144 macroblock-era numbers.
export FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-4}
export FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-2}

# --- FIM objective, held identical to phase 2's defaults ---------------------------
export P_FIM=${P_FIM:-0.5}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export USE_EOS=${USE_EOS:-1}
export FIM_MIN_GAP=${FIM_MIN_GAP:-64}
export FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
export SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}

export MODEL_TAG=${MODEL_TAG:-byte-phase3-45kv-341m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
mkdir -p "${OUT_DIR}/logs"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL
    --output="${OUT_DIR}/logs/%x-%j.out" --error="${OUT_DIR}/logs/%x-%j.err")
[[ -n "${EXCLUDE_NODES:-}" ]] && sbatch_args+=(--exclude="${EXCLUDE_NODES}")
# AFTER_JOBID chains this submission behind a prior one (see resubmit.sh) -- afterany,
# not afterok, since the prior job will most likely end by hitting vulcan-high's
# 1-12:00:00 wall-time cap rather than exiting 0, and afterok would never fire on that.
[[ -n "${AFTER_JOBID:-}" ]] && sbatch_args+=(--dependency=afterany:"${AFTER_JOBID}")

echo "[phase3/vulcan] ${MAX_ROWS} videos / ~341M (${N_LAYER}/${N_EMBD}/${N_HEAD}) / ${DEVICES}x a6000 / p_fim=${P_FIM} / steps=${STEPS}"
echo "  OUT_DIR=${OUT_DIR}"
[[ -n "${AFTER_JOBID:-}" ]] && echo "  chained after job ${AFTER_JOBID} (afterany)"
job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train.sbatch")
echo "Submitted phase3 job ${job_id}"
