#!/bin/bash
# Two ~1B-global-model ablations against the 7B patch-256 run
# (byte-jpeglm-7b-qwen3-patch256-fim-p100-changing-fullseq-spanw1-eosaux1-5ep, "A").
#
#   B  patch 32, one GOP per window       -> is a 256-byte patch (> a P frame) the bottleneck?
#   D  patch 32, consecutive GOPs packed  -> does earlier-GOP (reference) context help, esp. IDR?
#
# B changes only global size + patch vs A (global FLOPs/byte ~= A: 1.0B/32 vs 7B/256)
# and keeps A's exact window set (131 KB raw budget, window_unit=gop).
# D changes only the window unit vs B. Packed windows are ~4.7x a GOP (whole
# clips, ~5 GOPs), so its global batch is 16 windows instead of 64 to keep
# bytes per optimizer step close to B. Compare B and D per step (and per byte,
# from the logged raw_tokens).
#
# Both keep A's 1M-step cosine schedule, so the LR at step <= 25k matches A,
# and stop on the wall-clock cap (SBATCH_TIME). Permanent milestones every 5k
# steps give comparable checkpoints (final is not written on a timeout).
# FIM holes are drawn 50/50 IDR/P during training so per-frame-type span CE is
# logged (training/fim_{idr,p}_span_ce_local_100steps); validation holes stay
# uniform, so val_loss_fim is comparable with A.
#
# Usage (from the repo root on a Zaratan login node):
#   bash scripts/hpc/zaratan/jpeglm_pretrain/submit_patch_context_ablation.sh           # B and D
#   VARIANTS=B bash scripts/hpc/zaratan/jpeglm_pretrain/submit_patch_context_ablation.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}
VARIANTS=${VARIANTS:-"B D"}

# Shared: everything not listed per variant matches run A.
export STAGED_CORPUS
export MODEL_ARCHITECTURE=qwen3
export N_LAYER=${N_LAYER:-20}
export N_EMBD=${N_EMBD:-2048}
export N_HEAD=${N_HEAD:-16}
export BYTE_PATCH_SIZE=32
export P_FIM=1.0
export FIM_FORMAT=psm
export FIM_LOSS_SCOPE=full
export FIM_SPAN_LOSS_WEIGHT=1
export EOS_AUX_LOSS_WEIGHT=1
export FIM_MIN_GAP=64
export FIM_MAX_GAP=1400
export SLICE_HEADER_GUARD_BYTES=0
export FIM_IDR_SAMPLING_PROBABILITY=${FIM_IDR_SAMPLING_PROBABILITY:-0.5}
export LEARNING_RATE=3e-4
export MIN_LEARNING_RATE=3e-5
export STEPS=1000000
export WARMUP_STEPS=2000
export EVAL_INTERVAL=${EVAL_INTERVAL:-2500}
export EVAL_ITERS=20
export SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
export LATEST_SAVE_INTERVAL=${LATEST_SAVE_INTERVAL:-5000}
export SAVE_FINAL=1
export INLINE_FINAL_EVAL=0
export NUM_WORKERS=12
export TRAIN_CPUS_PER_TASK=16
export ENABLE_LENGTH_BUCKETING=1
export LENGTH_BUCKET_POOL_SIZE=8192
export ACTIVATION_CHECKPOINTING=1
export COMPILE=1
export SBATCH_TIME=${SBATCH_TIME:-12:00:00}

for variant in ${VARIANTS}; do
    case "${variant}" in
        B)
            window_unit=gop
            raw_context=131072   # same windows as A (no GOP dropped)
            gbs=64
            micro=8
            tag=byte-jpeglm-1b-qwen3-patch32-gop-fim-p100-idr50-ablB
            ;;
        D)
            window_unit=byte_budget
            raw_context=32768    # ~5 consecutive GOPs (a whole clip) per window
            gbs=16
            micro=4
            tag=byte-jpeglm-1b-qwen3-patch32-bb32k-fim-p100-idr50-ablD
            ;;
        *)
            echo "Unknown variant ${variant}; expected B or D" >&2
            exit 2
            ;;
    esac
    echo "== variant ${variant}: window_unit=${window_unit} raw_context=${raw_context}B" \
         "block=$((raw_context / BYTE_PATCH_SIZE)) gbs=${gbs} micro=${micro} time=${SBATCH_TIME}"
    WINDOW_UNIT="${window_unit}" \
    RAW_CONTEXT_BYTES="${raw_context}" \
    GLOBAL_BATCH_SIZE="${gbs}" \
    MICRO_BATCH_SIZE="${micro}" \
    MODEL_TAG="${tag}" \
    OUT_DIR="${STAGED_CORPUS}/runs/${tag}" \
    bash "${SCRIPT_DIR}/submit.sh"
done
