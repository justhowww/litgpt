#!/usr/bin/env bash
# Phase 3 eval: the same AR probes as phase2_fim/eval.sh, pointed at the 45k-video/
# 341M run. Compare at a MATCHED STEP against phase 1/phase 2 checkpoints where
# relevant -- all three anneal over STEPS=1M, so checkpoints at different steps sit at
# different LR and are not directly comparable.
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

DATA=/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm
OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-phase3-45kv-341m}

CKPT=${CKPT:-$(basename "$(ls -d "${OUT_DIR}"/step-* 2>/dev/null | sort | tail -1)")}
if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found under ${OUT_DIR}/step-*" >&2
    exit 1
fi
echo "[phase3 eval] checkpoint=${CKPT}"

EVAL=scripts/byte/eval/eval_ar_continuation.py
COMMON_TRAIN=(
    "${DATA}/manifest.jsonl"
    --nal-index-path "${DATA}/nal_index.sqlite"
    --checkpoint-dirs "${OUT_DIR}/${CKPT}"
    --train-split-file "${OUT_DIR}/train_split.json"
    --max-manifest-rows 45000
    --num-clips 20
    --seed 42
    --prefix-frames 4 --cont-frames 2
    --max-window-bytes 16384
)

run() {  # name  out_root  extra-sampling-args...
    local name="$1" out_root="$2"; shift 2
    local dir="${out_root}/${name}"
    if [[ -d "${dir}" ]]; then
        echo "[clean] removing previous ${dir}"
        rm -rf "${dir}"
    fi
    echo "===================== [train/${name}] ====================="
    python "${EVAL}" "${COMMON_TRAIN[@]}" --out-dir "${dir}" "$@"
}

TRAIN_OUT="${OUT_DIR}/eval_decode/train/${CKPT}"
run greedy        "${TRAIN_OUT}" --temperature 0.0
run greedy_masked "${TRAIN_OUT}" --temperature 0.0 --mask-illegal-bytes --mask-debug

echo
echo "===================== train sweep summary ====================="
python3 - "${TRAIN_OUT}" greedy greedy_masked <<'PY'
import json, sys
from pathlib import Path

root = sys.argv[1]
names = sys.argv[2:]

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
    print(
        f"{name:<14} {cont.get('success_rate', 0):>10.3f} "
        f"{frames_ref:>19} "
        f"{cont.get('completed_bytes_mean', 0):>16.1f} "
        f"{cont.get('target_bytes_mean', 0):>14.1f} "
        f"{cont.get('cont_psnr_mean') or 0:>8.2f}  {cont.get('desync_region_top')}"
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
print("Phase 1 @ step-00031000 for reference (260716 - Eval with syntax validation mask):")
print("  greedy         full_cont=0.750  tf_byte_acc=0.99978")
print("  greedy_masked  full_cont=1.000  tf_byte_acc=0.99978")
print()
print("Phase 1's val/train gap on residual coding (260712 - phase 1 result.md), the")
print("thing 45k videos is meant to fix -- compare element_correct for coeff_token/CBP")
print("above against these on a held-out (val) pass, not just this train pass:")
print("  coeff_token: train 0.9916 -> val 0.7609")
print("  CBP:         train 0.9896 -> val 0.9185")
PY

NAL_TERM=scripts/byte/eval/analyze_nal_termination.py
for name in greedy greedy_masked; do
    echo
    echo "===================== [nal-termination/${name}] ====================="
    python "${NAL_TERM}" "${TRAIN_OUT}/${name}" --slice-max-mbs 1 \
        || echo "[warn] nal_termination failed for ${name} (summary above is unaffected)"
done
