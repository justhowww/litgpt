#!/bin/bash
# JPEG-LM overfit job: train a plain byte transformer on one A100, then replay
# the exact training windows/holes and check byte-exact greedy generation.

#SBATCH --job-name=jpeglm-overfit
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
export DEVICES=1 NUM_NODES=1
EVAL_CHECKPOINT=${OVERFIT_EVAL_CHECKPOINT:-final}

if [[ "${OVERFIT_EVAL_ONLY:-0}" != "1" ]]; then
    source "${REPO_ROOT}/scripts/hpc/zaratan/jpeglm_pretrain/train.sh"
fi

if [[ ! -r "${OUT_DIR}/${EVAL_CHECKPOINT}/lit_model.pth" || ! -r "${OUT_DIR}/train_split.json" ]]; then
    echo "Missing ${OUT_DIR}/${EVAL_CHECKPOINT}/lit_model.pth or train_split.json" >&2
    exit 1
fi
cd "${REPO_ROOT}"

if [[ "${P_FIM}" == "0" || "${P_FIM}" == "0.0" ]]; then
    # O0: pure AR. Prompt = first frame of each training GOP; generate the rest + EOS.
    for split in train val; do
        python -u scripts/hpc/zaratan/jpeglm_overfit/eval_overfit_ar.py "${OUT_DIR}" \
            --checkpoint "${EVAL_CHECKPOINT}" \
            --split "${split}" \
            --nal-index-path "${NAL_INDEX}" \
            --window-min-frames "${WINDOW_MIN_FRAMES}"
    done
    exit 0
fi

# O1+: FIM. Three evaluations, each skipped if its summary already exists:
#   train          replays the cached training holes (fixed-hole runs only; hash-verified)
#   train_newholes 4 deterministic new holes per training window, disjoint from any
#                  cached training hole: known content, unseen cut
#   val            sampled holes in the held-out video (reference only)
fim_common=(
    "${MANIFEST}"
    --nal-index-path "${NAL_INDEX}"
    --checkpoint-dirs "${OUT_DIR}/${EVAL_CHECKPOINT}"
    --train-split-file "${OUT_DIR}/train_split.json"
    --max-manifest-rows "${MAX_ROWS}"
    --max-window-bytes "${RAW_CONTEXT_BYTES}"
    --window-min-frames "${WINDOW_MIN_FRAMES}"
    --window-unit gop
    --val-fraction "${VAL_FRACTION}"
    --split-by-video
    --seed 42
    --fim-format "${FIM_FORMAT}"
    --fim-loss-scope full
    --use-eos
    --fim-min-gap "${FIM_MIN_GAP}"
    --fim-max-gap "${FIM_MAX_GAP}"
    --slice-header-guard-bytes 0
    --stop-modes learned_eos
    --slice-layout frame
    --temperature 0
    --max-gen-bytes 4096
    --num-visualizations 4
    --ffmpeg-binary "${FFMPEG_BINARY}"
    --ffprobe-binary "${FFPROBE_BINARY}"
)
run_fim_eval() {
    local name=$1; shift
    local out="${OUT_DIR}/eval_overfit_fim/${EVAL_CHECKPOINT}/${name}"
    if [[ -e "${out}/summary.csv" ]]; then
        echo "[overfit] ${name}: ${out}/summary.csv exists; skipping"
        return
    fi
    python -u scripts/byte/eval/eval_fim_avclm.py "${fim_common[@]}" "$@" --out-dir "${out}"
}
if [[ "${FIXED_FIM_HOLES_PER_WINDOW}" != "0" ]]; then
    run_fim_eval train --eval-split train --hole-set trained --num-clips 1000
fi
run_fim_eval train_newholes --eval-split train --hole-set heldout \
    --heldout-holes-per-window 4 --num-clips 1000
run_fim_eval val --eval-split val --hole-set sampled --num-clips 20
