#!/bin/bash
# Byte-CE-only Stage 2 mixed AR/FIM with within-video validation.
#
# This is the direct counterpart to the held-out-video CE experiment. The
# model, data objective, gap range, and reconstruction probes are unchanged;
# only the validation split differs. Train and validation contain distinct
# target-slice samples, but slices from the same source video may occur in both.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STEPS=${STEPS:-1000}
export P_FIM=${P_FIM:-0.5}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export FIM_MIN_GAP=${FIM_MIN_GAP:-1024}
export FIM_MAX_GAP=${FIM_MAX_GAP:-8192}

# CE only. Keep EOS and MRT disabled to isolate byte-CE learning.
export MRT_INTERVAL=0
export USE_EOS=${USE_EOS:-0}
export EOS_LOSS_WEIGHT=${EOS_LOSS_WEIGHT:-1}
export EOS_AUX_LOSS_WEIGHT=${EOS_AUX_LOSS_WEIGHT:-0}
export CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-1.0}
export CE_BYTE_ONLY=${CE_BYTE_ONLY:-0}

# Slice-level split: exact target samples remain disjoint, while source videos
# may appear in both train and validation. This tests within-video interpolation
# before asking the small model to generalize to entirely unseen videos.
export VAL_FRACTION=${VAL_FRACTION:-0.05}
export SPLIT_BY_VIDEO=0

# Use the same strict reconstruction matrix as the held-out-video experiment.
export RECONSTRUCTION_TASK=fim
export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-500}
export RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-50}
export RECONSTRUCTION_VISUALIZATION_SAMPLES=${RECONSTRUCTION_VISUALIZATION_SAMPLES:-3}
export RECONSTRUCTION_MAX_TARGET_BYTES=8192
export RECONSTRUCTION_ORACLE_LENGTH=1
export RECONSTRUCTION_LEARNED_EOS=0
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-ce-only-within-video-1k-8k}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-ce-only-within-video-1k-8k"}
# Train from scratch so the split comparison is not confounded by initialization.
export INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-""}

exec bash "${BASE_DIR}/submit.sh"
