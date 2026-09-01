#!/usr/bin/env bash
# Controlled 200-step rerun of the failed AVC-LM joint CE+GRPO experiment.
# The only objective change is parser-derived FIM stop/no-stop supervision.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STAGED_CORPUS=${STAGED_CORPUS:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm}
export NAL_INDEX=${NAL_INDEX:-${STAGED_CORPUS}/nal_index.sqlite}

export N_LAYER=27
export N_EMBD=1024
export N_HEAD=16
export BLOCK_SIZE=16384
export BYTE_PATCH_SIZE=8
export MEGABYTE_LOCAL_LAYERS=4
export MEGABYTE_LOCAL_EMBD=512
export MEGABYTE_LOCAL_HEADS=8
export DEVICES=4
export NUM_WORKERS=4
export NO_ENCODING=1

export MAX_ROWS=45000
export GLOBAL_BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export VAL_FRACTION=0.01
export SPLIT_BY_VIDEO=0
export WINDOW_MIN_FRAMES=2
export P_FIM=0.5
export FIXED_FIM_HOLES=0
export FIXED_FIM_HOLES_PER_WINDOW=0
export FIM_FORMAT=psm
export FIM_LOSS_SCOPE=full
export FIM_SPAN_LOSS_WEIGHT=0.0
export USE_EOS=1
export EOS_LOSS_WEIGHT=1.0
export EOS_AUX_LOSS_WEIGHT=0.0
export FIM_MIN_GAP=64
export FIM_MAX_GAP=1400
export SLICE_HEADER_GUARD_BYTES=64

export STEPS=200
export WARMUP_STEPS=0
export LEARNING_RATE=2e-6
export MIN_LEARNING_RATE=2e-6
export EVAL_INTERVAL=100
# Saving every ten steps includes the planned 20/50/100/200 timeline without
# modifying generic checkpoint scheduling. Keep only those four after analysis.
export SAVE_INTERVAL=10
export FREE_RUN_INTERVAL=0
export MRT_INTERVAL=0

export RESUME=0
export INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-${STAGED_CORPUS}/runs/byte-phase3-avclm-341m-megabyte-patch8-fim-p050-fullseq-eos-45kv/step-00057000}
export GRPO_REFERENCE_CHECKPOINT_DIR=${GRPO_REFERENCE_CHECKPOINT_DIR:-${INITIAL_CHECKPOINT_DIR}}
export GRPO_INTERVAL=1
export GRPO_START_STEP=0
export GRPO_GROUP_SIZE=4
export GRPO_CONTEXT_SAMPLING=online
export GRPO_CONTEXT_SEED=42
export GRPO_MAX_TARGET_BYTES=1400
export GRPO_TEMPERATURE=1.0
export GRPO_TOP_K=50
export GRPO_TOP_P=0.9
export GRPO_KL_COEFF=0.02
export GRPO_PSNR_CAP=40.0
export GRPO_DECODE_FAILURE_REWARD=-1.0
export GRPO_MU=1
export GRPO_CLIP_RANGE=0.2
export GRPO_LEARNED_EOS=1
export GRPO_GENERATION_BUDGET_MULTIPLIER=2.0
export GRPO_AR_PREFIX_FRAMES=4
export GRPO_AR_CONT_FRAMES=2
export GRPO_AR_SLICE_LAYOUT=macroblock
export GRPO_FIM_SLICE_LAYOUT=macroblock
export GRPO_STOP_NEGATIVE_SAMPLES=4
# One certified trajectory per rank is enough for the balanced auxiliary loss
# and bounds repeated suffix-parser work during this first controlled run.
export GRPO_STOP_MAX_POSITIVE_STATES=1
export GRPO_DECODE_WORKERS=2
export GRPO_TIMEOUT_SEC=30
export GRPO_FFMPEG_BINARY=ffmpeg

: "${GRPO_STOP_LOSS_WEIGHT:?Set GRPO_STOP_LOSS_WEIGHT after the one-update calibration check}"
export GRPO_STOP_LOSS_WEIGHT

export MODEL_TAG=${MODEL_TAG:-byte-phase3-avclm-341m-megabyte-patch8-grpo-stopaux200-v1}
export OUT_DIR=${OUT_DIR:-${STAGED_CORPUS}/runs/${MODEL_TAG}}

exec "${SCRIPT_DIR}/submit.sh"
