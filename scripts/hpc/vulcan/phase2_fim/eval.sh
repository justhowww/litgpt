#!/usr/bin/env bash
# Phase 2 eval: the AR probes, run with phase 1's EXACT flags (same clips, seed,
# prefix/cont frames, window budget, sampling) against the phase-2 checkpoint. The
# whole point of phase 2 is the delta against phase 1, so any flag drift between this
# file and phase1_repro/eval.sh silently destroys the comparison -- keep them in step.
#
# Compare at a MATCHED STEP, not at "latest": both runs anneal over STEPS=1M, so a
# phase-2 checkpoint at a different step is at a different LR and is not comparable.
#   CKPT=step-00031000 ./eval.sh          # the step phase 1 was last read at
#
# The same train-split caveat as phase 1 applies verbatim (within-video split =>
# train_split.json's video list is ~all 1000 videos, so --train-split-file gives a
# "did we train on these videos" filter, NOT a leakage-free held-out set).
#
# NOT covered here: FIM reconstruction. The probe is slice-only (reconstruction.py
# reads sample.target_index), and on this per-MB corpus a repair number would not be a
# task result anyway -- val_loss_fim from training is the FIM signal. See train.sh.
set -euo pipefail

DATA=/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm
OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-phase2-fim-1000v-85m}

CKPT=${CKPT:-$(basename "$(ls -d "${OUT_DIR}"/step-* 2>/dev/null | sort | tail -1)")}
if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found under ${OUT_DIR}/step-*" >&2
    exit 1
fi
echo "[phase2 eval] checkpoint=${CKPT}"

EVAL=scripts/byte/eval/eval_ar_continuation.py
COMMON_TRAIN=(
    "${DATA}/manifest.jsonl"
    --nal-index-path "${DATA}/nal_index.sqlite"
    --checkpoint-dirs "${OUT_DIR}/${CKPT}"
    --train-split-file "${OUT_DIR}/train_split.json"
    --max-manifest-rows 1000
    --num-clips 20
    --seed 42
    --prefix-frames 4 --cont-frames 2
    --max-window-bytes 16384
)

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

hdr = f"{'sampling':<14} {'decode_rate':>11} {'full_cont':>10} {'survival_mean':>14} {'psnr':>8}  desync_top"
print(hdr)
print("-" * len(hdr))
for name in names:
    tf, cont = rows(name)
    if cont is None:
        print(f"{name:<14} (missing metrics.jsonl)")
        continue
    print(
        f"{name:<14} {cont.get('decode_rate', 0):>11.3f} "
        f"{cont.get('full_continuation_rate', 0):>10.3f} "
        f"{cont.get('survival_bytes_mean', 0):>14.1f} "
        f"{cont.get('cont_psnr_mean') or 0:>8.2f}  {cont.get('desync_region_top')}"
    )
print()
for name in names:
    tf, _ = rows(name)
    if tf is not None:
        print(
            f"{name:<14} tf_byte_acc={tf.get('tf_byte_acc_mean'):.5f} "
            f"correct_coeff_token={tf.get('correct_coeff_token')} "
            f"correct_cbp={tf.get('correct_cbp')}"
        )
print()
print("Phase 1 @ step-00031000 for reference (260716 - Eval with syntax validation mask):")
print("  greedy         full_cont=0.750  tf_byte_acc=0.99978")
print("  greedy_masked  full_cont=1.000  tf_byte_acc=0.99978")
print("A phase-2 full_cont at or near these means FIM did not cost AR.")
PY

# Per-NAL termination diagnostics. Runs AFTER the summary and cannot abort it: under
# `set -e` a non-zero exit here would otherwise kill the script before the headline
# ever printed. A diagnostic must never suppress the result it explains.
NAL_TERM=scripts/byte/eval/analyze_nal_termination.py
for name in greedy greedy_masked; do
    echo
    echo "===================== [nal-termination/${name}] ====================="
    python "${NAL_TERM}" "${TRAIN_OUT}/${name}" --slice-max-mbs 1 \
        || echo "[warn] nal_termination failed for ${name} (summary above is unaffected)"
done
