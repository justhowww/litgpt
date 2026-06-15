#!/bin/bash
# MRT-primary learned stopping with a non-saturating (clipped_mse) reward.
#
# One variable changed vs. stage2_fim_psm_mrt_primary_learned_stop_1k_8k:
#   MRT_RISK_MODE: smooth_mse -> clipped_mse
# Adds an oracle-length reconstruction probe at eval time as a diagnostic
# (logged separately as model_oracle/*), without changing training-time policy.
# GT remains in the 8-candidate pool (intentional anchor for the sampled
# candidates; see exp design notes).

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STEPS=${STEPS:-1000}
export P_FIM=1
export FIM_FORMAT=psm
export FIM_MIN_GAP=1024
export FIM_MAX_GAP=8192

# EOS is not a dataset target. It is available only as an MRT generation
# action and receives reward through the decoded candidate it terminates.
export USE_EOS=0
export EOS_LOSS_WEIGHT=1
export EOS_AUX_LOSS_WEIGHT=0
export CE_LOSS_WEIGHT=${CE_LOSS_WEIGHT:-0.05}
export CE_BYTE_ONLY=1

export MRT_INTERVAL=1
export MRT_START_STEP=0
export MRT_NUM_CANDIDATES=${MRT_NUM_CANDIDATES:-8}
export MRT_CONTEXT_POOL_SIZE=${MRT_CONTEXT_POOL_SIZE:-64}
export MRT_MAX_TARGET_BYTES=8192
export MRT_ORACLE_LENGTH=0
export MRT_LEARNED_EOS=1
export MRT_WEIGHT=${MRT_WEIGHT:-1.0}

# Desaturated reward. scaled_mse maps mse -> mse_weight * mse with no clipping
# and no asymptote. The scale is chosen so typical sampled risk is order 1:
#
#   r_i = MSE_i / s,   s = median(MSE_sampled) from prior runs ~= 0.035
#
# i.e. MRT_MSE_WEIGHT = 1 / 0.035 = 28.5714. Under this scale:
#   MSE 0.01  -> risk 0.29
#   MSE 0.035 -> risk 1.0    (typical candidate)
#   MSE 0.10  -> risk 2.86
#
# Decode failure risk = 5.0 sits at the upper tail of typical successful
# decodes -- catastrophically broken bitstreams get a clear penalty without
# dominating the gradient.
#
# Scale fixed across runs so risk curves remain comparable. If the
# mrt_grad_norm / weighted_ce_grad_norm ratio is wrong after 100-500 steps
# (target ~2-10x for MRT-primary), adjust MRT_WEIGHT, NOT this scale.
export MRT_RISK_MODE=${MRT_RISK_MODE:-scaled_mse}
export MRT_MSE_WEIGHT=${MRT_MSE_WEIGHT:-28.5714}
export MRT_MAX_RISK=${MRT_MAX_RISK:-100.0}
export MRT_DECODE_FAILURE_WEIGHT=${MRT_DECODE_FAILURE_WEIGHT:-5.0}
export MRT_DECODE_WORKERS=${MRT_DECODE_WORKERS:-8}

export RECONSTRUCTION_TASK=fim
export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-250}
export RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-50}
export RECONSTRUCTION_MAX_TARGET_BYTES=8192
# Primary policy: learned EOS (training-time).
# Diagnostic probe: oracle length on the same checkpoint, logged under
# reconstruction/fim/model_oracle/*. Both flags are simultaneously enabled now
# that the spurious mutex check in scripts/byte/train.py has been removed.
export RECONSTRUCTION_ORACLE_LENGTH=1
export RECONSTRUCTION_LEARNED_EOS=1
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-mrt-primary-learned-stop-desat-1k-8k}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-mrt-primary-learned-stop-desat-1k-8k"}
export INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-balanced-eos-1k-8k/step-00001000"}

if [[ ! -r "${INITIAL_CHECKPOINT_DIR}/lit_model.pth" ]]; then
    echo "Initial checkpoint is not readable: ${INITIAL_CHECKPOINT_DIR}/lit_model.pth" >&2
    exit 1
fi

exec bash "${BASE_DIR}/submit.sh"
