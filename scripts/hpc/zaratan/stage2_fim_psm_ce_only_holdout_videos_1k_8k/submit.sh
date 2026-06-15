#!/bin/bash
# Byte-CE-only Stage 2 mixed AR/FIM with held-out-video validation.
#
# Purpose: clean test of whether byte-CE alone is sufficient for the FIM
# repair task. Removes three confounds in prior runs:
#   1. No MRT (CE only).
#   2. Held-out video split (no within-video slice leakage).
#   3. Strict-mode decoding is the primary metric (concealment-on is logged
#      as a diagnostic only; the comparison story is built around strict).
#
# Reconstruction probes log a full comparison matrix per checkpoint, broken
# down by gap-size bucket (1k_2k, 2k_4k, 4k_8k):
#
#   reconstruction/fim/model_learned/strict/...       <- our method, no concealment (primary)
#   reconstruction/fim/model_learned/error_exploding/...  <- our method, with concealment (diagnostic)
#   reconstruction/fim/model_oracle/strict/...        <- our method, oracle length, no concealment
#   reconstruction/fim/model_oracle/error_exploding/...  <- our method, oracle length, with concealment
#   reconstruction/fim/deleted_gap/strict/...         <- empty-replacement, no concealment (lower bound)
#   reconstruction/fim/deleted_gap/error_exploding/...  <- standard decoder error concealment baseline
#   reconstruction/fim/random_bytes/strict/...        <- random-byte filling, no concealment
#   reconstruction/fim/random_bytes/error_exploding/...  <- random-byte filling, with concealment
#   reconstruction/fim/ground_truth/strict/...        <- upper bound (~inf PSNR)
#
# Same metrics repeat under .../bucket_1k_2k/, .../bucket_2k_4k/, .../bucket_4k_8k/.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STEPS=${STEPS:-1000}
export P_FIM=${P_FIM:-0.5}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export FIM_MIN_GAP=${FIM_MIN_GAP:-1024}
export FIM_MAX_GAP=${FIM_MAX_GAP:-8192}

# CE only. No MRT.
export MRT_INTERVAL=0
export USE_EOS=${USE_EOS:-0}
export EOS_LOSS_WEIGHT=${EOS_LOSS_WEIGHT:-1}
export EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-0}
export CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-1.0}
export CE_BYTE_ONLY=${CE_BYTE_ONLY:-0}

# Held-out-video validation split. val_fraction interpreted as fraction of
# source videos (not slices). With ~OpenVid manifest sizes a 5% video-level
# split gives a few hundred genuinely-held-out videos.
export VAL_FRACTION=${VAL_FRACTION:-0.05}
export SPLIT_BY_VIDEO=1

# Eval probes: strict primary, error_exploding as diagnostic. FIM baselines
# (ground_truth, deleted_gap, random_bytes) enabled so the comparison matrix
# is complete. Oracle-length probe added as a content-only diagnostic.
export RECONSTRUCTION_TASK=fim
export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-500}
export RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-50}
export RECONSTRUCTION_VISUALIZATION_SAMPLES=${RECONSTRUCTION_VISUALIZATION_SAMPLES:-3}
export RECONSTRUCTION_MAX_TARGET_BYTES=8192
export RECONSTRUCTION_ORACLE_LENGTH=1
export RECONSTRUCTION_LEARNED_EOS=0
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-ce-only-holdout-videos-1k-8k}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-ce-only-holdout-videos-1k-8k"}
# Train from scratch on Stage 2 mixed objective. No initial checkpoint so the
# CE optimization is unconfounded by prior MRT-trained weights.
export INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-""}

exec bash "${BASE_DIR}/submit.sh"
