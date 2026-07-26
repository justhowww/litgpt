#!/usr/bin/env bash
# Phase 3 AR eval. By default this targets the 45k AVC-LM run; eval_bscv.sh supplies
# the BSCV corpus and the one-progressive-picture-per-slice layout. Compare at a
# MATCHED STEP against earlier checkpoints where relevant -- checkpoints at different
# steps sit at different learning rates and are not directly comparable.
#
# The one signal phase 1/2 could NOT give: whether the train/val gap on residual
# coding (coeff_token, level, total_zeros, run_before -- see
# 04 - projects/real-time-diffusion-video-decoder/260712 - phase 1 result.md) shrinks
# now that the corpus is ~45x larger. That gap, not success_rate alone, is
# the actual test of whether 45k videos bought real generalization. Read the
# per-element accuracy table this script's teacher-forced pass produces (via
# eval_ar_continuation.py's own output) against phase 1's numbers directly.
#
# Same train-split caveat as phase 1/2: --train-split-file gives a "did we train on
# this video" filter (within-video split), not a leakage-free held-out set.
#
# NOT covered here: FIM reconstruction (slice-only probe, not meaningful on this
# per-MB corpus -- see phase2_fim/train.sh's header). val_loss_fim from training is
# the FIM signal.
set -euo pipefail

DATA=${DATA:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm}
OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-phase3-45kv-341m}
NAL_INDEX=${NAL_INDEX:-${DATA}/nal_index.sqlite}
MAX_MANIFEST_ROWS=${MAX_MANIFEST_ROWS:-45000}
NUM_CLIPS=${NUM_CLIPS:-20}
PREFIX_FRAMES=${PREFIX_FRAMES:-4}
CONT_FRAMES=${CONT_FRAMES:-2}
MAX_WINDOW_BYTES=${MAX_WINDOW_BYTES:-16384}
MAX_GEN_MULTIPLE=${MAX_GEN_MULTIPLE:-2}
SLICE_LAYOUT=${SLICE_LAYOUT:-macroblock}
EVAL_INTRA=${EVAL_INTRA:-1}
CLIP_LIST=${CLIP_LIST:-}
EVAL_SET_NAME=${EVAL_SET_NAME:-}
EVAL_SPLIT=${EVAL_SPLIT:-train}

if [[ -n "${EVAL_SET_NAME}" && ! "${EVAL_SET_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EVAL_SET_NAME may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 1
fi
if [[ "${EVAL_SPLIT}" != "train" && "${EVAL_SPLIT}" != "all" ]]; then
    echo "EVAL_SPLIT must be train or all" >&2
    exit 1
fi

CKPT=${CKPT:-$(basename "$(ls -d "${OUT_DIR}"/step-* 2>/dev/null | sort | tail -1)")}
if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found under ${OUT_DIR}/step-*" >&2
    exit 1
fi
echo "[phase3 eval] checkpoint=${CKPT}"

EVAL=scripts/byte/eval/eval_ar_continuation.py
COMMON_TRAIN=(
    "${DATA}/manifest.jsonl"
    --nal-index-path "${NAL_INDEX}"
    --checkpoint-dirs "${OUT_DIR}/${CKPT}"
    --train-split-file "${OUT_DIR}/train_split.json"
    --max-manifest-rows "${MAX_MANIFEST_ROWS}"
    --num-clips "${NUM_CLIPS}"
    --seed 42
    --prefix-frames "${PREFIX_FRAMES}" --cont-frames "${CONT_FRAMES}"
    --max-window-bytes "${MAX_WINDOW_BYTES}"
    --max-gen-multiple "${MAX_GEN_MULTIPLE}"
    --slice-layout "${SLICE_LAYOUT}"
)
if [[ "${EVAL_INTRA}" == "0" ]]; then
    COMMON_TRAIN+=(--no-eval-intra)
fi
if [[ "${EVAL_SPLIT}" == "all" ]]; then
    COMMON_TRAIN+=(--no-train-split-filter)
fi
if [[ -n "${CLIP_LIST}" ]]; then
    COMMON_TRAIN+=(--clip-list "${CLIP_LIST}")
fi

run() {  # name  out_root  extra-sampling-args...
    local name="$1" out_root="$2"; shift 2
    local dir="${out_root}/${name}"
    # Start clean: metrics.jsonl is append-only while config.json/summary.csv are
    # overwritten, so re-running into a populated dir interleaves two runs' rows and
    # destroys the record of which config produced them.
    if [[ -d "${dir}" ]]; then
        echo "[clean] removing previous ${dir}"
        rm -rf "${dir}"
    fi
    echo "===================== [train/${name}] ====================="
    python "${EVAL}" "${COMMON_TRAIN[@]}" --out-dir "${dir}" "$@"
}

TRAIN_OUT="${OUT_DIR}/eval_decode/train/${CKPT}"
if [[ -n "${EVAL_SET_NAME}" ]]; then
    TRAIN_OUT="${TRAIN_OUT}/${EVAL_SET_NAME}"
fi
run greedy        "${TRAIN_OUT}" --temperature 0.0
# run greedy_residual_masked "${TRAIN_OUT}" \
#     --temperature 0.0 --mask-illegal-bytes --mask-residual-only --mask-debug
run greedy_full_masked "${TRAIN_OUT}" --temperature 0.0 --mask-illegal-bytes --mask-debug

echo
echo "===================== train sweep summary ====================="
python3 - "${TRAIN_OUT}" "${SLICE_LAYOUT}" greedy greedy_residual_masked greedy_full_masked <<'PY'
import json, sys
from pathlib import Path

root = sys.argv[1]
layout = sys.argv[2]
names = sys.argv[3:]

def rows(name):
    p = Path(root) / name / "metrics.jsonl"
    if not p.exists():
        return None, None
    tf = cont = None
    for line in p.open():
        r = json.loads(line)
        if r.get("mode") == "teacher_forced":
            tf = r
        elif r.get("mode") == "continuation":
            cont = r
    return tf, cont

hdr = f"{'sampling':<14} {'success':>10} {'frames(done/target)':>19} {'completed_bytes':>16} {'target_bytes':>14} {'psnr':>8}  desync_top"
print(hdr)
print("-" * len(hdr))
for name in names:
    tf, cont = rows(name)
    if cont is None:
        print(f"{name:<14} (missing metrics.jsonl)")
        continue
    completed_frames = cont.get('completed_frames_mean')
    frames_ref = f"{completed_frames:.1f}/{cont.get('target_frames', 0)}" if completed_frames is not None else f"-/{cont.get('target_frames', 0)}"
    gen_seconds = cont.get("generation_seconds_mean")
    gen_rate = cont.get("generation_bytes_per_second_mean")
    print(
        f"{name:<14} {cont.get('success_rate', 0):>10.3f} "
        f"{frames_ref:>19} "
        f"{cont.get('completed_bytes_mean', 0):>16.1f} "
        f"{cont.get('target_bytes_mean', 0):>14.1f} "
        f"{cont.get('cont_psnr_mean') or 0:>8.2f}  {cont.get('desync_region_top')}"
    )
    if gen_seconds is not None:
        rate_text = f"{gen_rate:.2f}" if gen_rate is not None else "-"
        print(
            f"{'':<14} generation={gen_seconds:.2f}s/clip "
            f"throughput={rate_text} bytes/s"
        )
    membership_total = cont.get("train_split_membership_total", 0)
    if membership_total:
        print(
            f"{'':<14} train_membership="
            f"{cont.get('train_split_membership_count', 0)}/{membership_total}"
        )
print()
for name in names:
    tf, _ = rows(name)
    if tf is not None:
        elements = tf.get('element_correct', {})
        print(
            f"{name:<14} tf_byte_acc={tf.get('tf_byte_acc_mean'):.5f} "
            f"coeff_token={elements.get('luma.coeff_token')} "
            f"cbp={elements.get('coded_block_pattern')}"
        )
print()
if layout == "macroblock":
    print("Historical phase 1 @ step-00031000 (the old residual-only mask):")
    print("  greedy                   full_cont=0.750  tf_byte_acc=0.99978")
    print("  greedy_residual_masked   full_cont=1.000  tf_byte_acc=0.99978")
    print("Read greedy_full_masked as the current constrained-decoding result.")
    print()
    print("Phase 1's val/train gap on residual coding (260712 - phase 1 result.md), the")
    print("thing 45k videos is meant to fix -- compare element_correct for coeff_token/CBP")
    print("above against these on a held-out (val) pass, not just this train pass:")
    print("  coeff_token: train 0.9916 -> val 0.7609")
    print("  CBP:         train 0.9896 -> val 0.9185")
else:
    print("BSCV frame-slice evaluation: no AVC-LM historical baseline is printed.")
    print("Compare greedy and greedy_full_masked at the same checkpoint and clip set.")
PY

# Per-NAL termination diagnostics. Runs AFTER the summary and cannot abort it: under
# `set -e` a non-zero exit here would otherwise kill the script before the headline
# ever printed. A diagnostic must never suppress the result it explains.
NAL_TERM=scripts/byte/eval/analyze_nal_termination.py
for name in greedy greedy_residual_masked greedy_full_masked; do
    echo
    echo "===================== [nal-termination/${name}] ====================="
    python "${NAL_TERM}" "${TRAIN_OUT}/${name}" --slice-layout "${SLICE_LAYOUT}" \
        || echo "[warn] nal_termination failed for ${name} (summary above is unaffected)"
done
