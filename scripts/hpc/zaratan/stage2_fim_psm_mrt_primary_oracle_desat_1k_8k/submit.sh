#!/bin/bash
# MRT-primary oracle-length, desaturated reward (run alpha).
#
# Paired with stage2_fim_psm_mrt_primary_learned_stop_desat_1k_8k (run beta).
# Identical reward, CE weight, checkpoint init, compute, and eval probes;
# only the training-time MRT policy differs:
#   alpha (this run): MRT_ORACLE_LENGTH=1, MRT_LEARNED_EOS=0
#   beta              MRT_ORACLE_LENGTH=0, MRT_LEARNED_EOS=1
#
# Both checkpoints are evaluated under both oracle-length and learned-stop
# probes (model_oracle/* and model_learned/*), enabling the four-cell
# decomposition that tests whether oracle-length training contradicts
# visual-only supervision.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd -- "${SCRIPT_DIR}/../stage2_fim_psm" && pwd)
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}

export STEPS=${STEPS:-1000}
export P_FIM=1
export FIM_FORMAT=psm
export FIM_MIN_GAP=1024
export FIM_MAX_GAP=8192

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

# The training-time policy variable. Oracle length, EOS disabled.
export MRT_ORACLE_LENGTH=1
export MRT_LEARNED_EOS=0

export MRT_WEIGHT=${MRT_WEIGHT:-1.0}
export MRT_INCLUDE_GROUND_TRUTH=${MRT_INCLUDE_GROUND_TRUTH:-1}

# Same reward calibration as beta. Risk = MSE / 0.035 (typical sampled risk
# is order 1); decode failure = 5.0.
export MRT_RISK_MODE=${MRT_RISK_MODE:-scaled_mse}
export MRT_MSE_WEIGHT=${MRT_MSE_WEIGHT:-28.5714}
export MRT_MAX_RISK=${MRT_MAX_RISK:-100.0}
export MRT_DECODE_FAILURE_WEIGHT=${MRT_DECODE_FAILURE_WEIGHT:-5.0}
export MRT_DECODE_WORKERS=${MRT_DECODE_WORKERS:-8}

# Eval probes. Both learned-stop and oracle-length so each checkpoint emits
# both model_learned/* and model_oracle/* strict-mode PSNR/SSIM curves.
export RECONSTRUCTION_TASK=fim
export RECONSTRUCTION_EVAL_INTERVAL=${RECONSTRUCTION_EVAL_INTERVAL:-250}
export RECONSTRUCTION_EVAL_SAMPLES=${RECONSTRUCTION_EVAL_SAMPLES:-50}
export RECONSTRUCTION_MAX_TARGET_BYTES=8192
export RECONSTRUCTION_ORACLE_LENGTH=1
export RECONSTRUCTION_LEARNED_EOS=1
export RECONSTRUCTION_ERROR_EXPLODING=1
export RECONSTRUCTION_FIM_BASELINES=1

export MODEL_NAME=${MODEL_NAME:-byte-stage2-fim-psm-mrt-primary-oracle-desat-1k-8k}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-mrt-primary-oracle-desat-1k-8k"}
export INITIAL_CHECKPOINT_DIR=${INITIAL_CHECKPOINT_DIR:-"${STAGED_CORPUS}/runs/byte-stage2-fim-psm-balanced-eos-1k-8k/step-00001000"}

if [[ ! -r "${INITIAL_CHECKPOINT_DIR}/lit_model.pth" ]]; then
    echo "Initial checkpoint is not readable: ${INITIAL_CHECKPOINT_DIR}/lit_model.pth" >&2
    exit 1
fi

exec bash "${BASE_DIR}/submit.sh"
