#!/bin/bash
# Two ~1B-global-model ablations against the 7B patch-256 run
# (byte-jpeglm-7b-qwen3-patch256-fim-p100-changing-fullseq-spanw1-eosaux1-5ep, "A").
#
#   B  patch 32, one GOP per window       -> is a 256-byte patch (> a P frame) the bottleneck?
#   D  patch 32, consecutive GOPs packed  -> does earlier-GOP (reference) context help, esp. IDR?
#   F  no MEGABYTE: plain 1B byte-level transformer (patch 1, every byte attends to
#      every earlier byte), 16 KB context (99.9% of GOPs whole), one GOP per window,
#      same data/objective/batch as B -> does the patch hierarchy lose content
#      information? ~23x B's FLOPs per byte; default cap 24 h. Compare with B at
#      matched bytes seen (logged raw_tokens), not matched steps.
#   G  paper-shaped MEGABYTE: patch 8, global 24 x 2048 (~1.16B; 256 global dims per
#      byte slot), local 15 x 1024 / 16 heads (~181M; local/global ~16% of params,
#      ~1.3x compute per byte), same windows as A and B (131 KB, <= 16,384 global
#      positions) -> MEGABYTE's best PG-19 shape. Does a correctly balanced
#      MEGABYTE close F's lead on P-frame content (setup), or not (fixed patches
#      inherently lose byte-exact copying)? Default cap 24 h; compare with F and B
#      at matched wall-clock and matched steps.
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
export N_EMBD=${N_EMBD:-2048}
export N_HEAD=${N_HEAD:-16}
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
SBATCH_TIME_OVERRIDE=${SBATCH_TIME:-}

for variant in ${VARIANTS}; do
    case "${variant}" in
        B)
            layers=20; local_layers=4; local_embd=512; local_heads=8
            patch=32
            time=12:00:00
            window_unit=gop
            raw_context=131072   # same windows as A (no GOP dropped)
            gbs=64
            micro=8
            tag=byte-jpeglm-1b-qwen3-patch32-gop-fim-p100-idr50-ablB
            ;;
        D)
            layers=20; local_layers=4; local_embd=512; local_heads=8
            patch=32
            time=12:00:00
            window_unit=byte_budget
            raw_context=32768    # ~5 consecutive GOPs (a whole clip) per window
            gbs=16
            micro=4
            tag=byte-jpeglm-1b-qwen3-patch32-bb32k-fim-p100-idr50-ablD
            ;;
        F)
            layers=20; local_layers=4; local_embd=512; local_heads=8   # local unused at patch 1
            patch=1              # plain transformer: no global/local split
            time=24:00:00
            window_unit=gop
            raw_context=16384    # 99.9% of GOPs fit whole
            gbs=64
            micro=4
            tag=byte-jpeglm-1b-qwen3-bytes-ctx16k-gop-fim-p100-idr50-ablF
            ;;
        G)
            layers=24; local_layers=15; local_embd=1024; local_heads=16
            patch=8
            time=24:00:00
            window_unit=gop
            raw_context=131072   # same windows as A and B
            gbs=64
            micro=4
            tag=byte-jpeglm-1b-qwen3-patch8-local180m-gop-fim-p100-idr50-ablG
            ;;
        *)
            echo "Unknown variant ${variant}; expected B, D, F or G" >&2
            exit 2
            ;;
    esac
    time=${SBATCH_TIME_OVERRIDE:-${time}}
    echo "== variant ${variant}: global ${layers}x${N_EMBD} local ${local_layers}x${local_embd}/${local_heads}h" \
         "patch=${patch} window_unit=${window_unit} raw_context=${raw_context}B" \
         "block=$((raw_context / patch)) gbs=${gbs} micro=${micro} time=${time}"
    N_LAYER="${layers}" \
    MEGABYTE_LOCAL_LAYERS="${local_layers}" \
    MEGABYTE_LOCAL_EMBD="${local_embd}" \
    MEGABYTE_LOCAL_HEADS="${local_heads}" \
    BYTE_PATCH_SIZE="${patch}" \
    SBATCH_TIME="${time}" \
    WINDOW_UNIT="${window_unit}" \
    RAW_CONTEXT_BYTES="${raw_context}" \
    GLOBAL_BATCH_SIZE="${gbs}" \
    MICRO_BATCH_SIZE="${micro}" \
    MODEL_TAG="${tag}" \
    OUT_DIR="${STAGED_CORPUS}/runs/${tag}" \
    bash "${SCRIPT_DIR}/submit.sh"
done
