#!/bin/bash
# Fixed-hole FIM overfit sanity test.
#
# Twenty train windows repeatedly see the exact same FIM prompt, missing span, and
# EOS target. This is intentionally different from normal Phase 2/3 training, which
# redraws holes on every access. The test asks only whether the training/evaluation
# path can memorize and replay FIM examples exactly.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm"}

export N_LAYER=${N_LAYER:-6}
export N_EMBD=${N_EMBD:-384}
export N_HEAD=${N_HEAD:-6}
export DEVICES=${DEVICES:-1}
export NO_ENCODING=${NO_ENCODING:-1}
export NUM_WORKERS=${NUM_WORKERS:-3}

# With the corpus's one-window-per-video layout, 21 rows and one held-out window
# produce exactly 20 fixed training holes.
export MAX_ROWS=${MAX_ROWS:-21}
export STEPS=${STEPS:-5000}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
export VAL_FRACTION=${VAL_FRACTION:-0.05}
export EVAL_INTERVAL=${EVAL_INTERVAL:-50}
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}

export P_FIM=${P_FIM:-1.0}
export FIXED_FIM_HOLES=${FIXED_FIM_HOLES:-1}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export USE_EOS=${USE_EOS:-1}
export FIM_MIN_GAP=${FIM_MIN_GAP:-64}
export FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
export SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}

export MODEL_TAG=${MODEL_TAG:-byte-fim-fixed-hole-overfit-10m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
mkdir -p "${OUT_DIR}/logs"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL
    --output="${OUT_DIR}/logs/%x-%j.out" --error="${OUT_DIR}/logs/%x-%j.err")
[[ -n "${EXCLUDE_NODES:-}" ]] && sbatch_args+=(--exclude="${EXCLUDE_NODES}")

echo "[fim-fixed-overfit] rows=${MAX_ROWS} model=~10M steps=${STEPS} p_fim=${P_FIM} fixed=${FIXED_FIM_HOLES}"
echo "  OUT_DIR=${OUT_DIR}"
job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train.sbatch")
echo "Submitted fixed-hole FIM overfit job ${job_id}"
