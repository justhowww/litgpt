#!/usr/bin/env bash
#
# AVC-LM eval sweep: teacher-forced accuracy + free-run desync-syntax test across the
# three sampling regimes, on the per-MB AVC-LM corpus (slice-max-mbs=1).
#
#   greedy       temperature 0.0
#   temp1        temperature 1.0                       (raw, samples the tail)
#   avclm_topk   temperature 1.0 top-k 50 top-p 0.9    (AVC-LM / JPEG-LM protocol)
#
# Reuses eval_ar_continuation.py. --survival-only bypasses the ffmpeg frame gate,
# which is invalid here: under slice-max-mbs=1 a VCL NAL is one macroblock, not a frame,
# so decode-based frame counting is meaningless. TF accuracy is sampling-independent, so
# it is identical across the three runs (kept in each for a self-contained record).
#
# Usage:
#   scripts/byte/eval/eval_avclm.sh <checkpoint-step-dir> [out-root]
# Env overrides:
#   AVCLM_DATA   dir holding manifest.jsonl + nal_index.sqlite
#                (default: $HOME/scratch.metzler-prj/OpenVid-1M_Data/data-avclm)
#
set -euo pipefail

CKPT="${1:?usage: eval_avclm.sh <checkpoint step dir> [out_root]}"
AVCLM_DATA="${AVCLM_DATA:-$HOME/scratch.metzler-prj/OpenVid-1M_Data/data-avclm}"
OUT_ROOT="${2:-$(dirname "$CKPT")/eval/avclm_sweep_$(basename "$CKPT")}"

MANIFEST="$AVCLM_DATA/manifest.jsonl"
NAL_INDEX="$AVCLM_DATA/nal_index.sqlite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="/../eval_ar_continuation.py"

# Match training exactly (stage0_ar_multiframe_xl/train.sh): dataset-mode=window,
# BLOCK_SIZE=16384 -> --max-window-bytes 16384 (= the model's trained context / KV-cache
# size). Clip selection is already IDR-anchored + SLICE_BOS + per-NAL offset reset, like
# ByteStreamWindowDataset, so an IDR-anchored eval window is in-distribution.
#
# Under slice-max-mbs=1 a VCL NAL is one MB, so at 256x144 (144 MBs/frame) prefix/cont
# "frames" count MB-slices: 288 = 2 frames of context, 144 = generate 1 frame. 288+144=432
# slices sit inside one 16384-byte window -- i.e. "given 2 frames of a training window,
# generate the 3rd", which is exactly the full-window next-byte objective the model saw.
BLOCK_SIZE="${BLOCK_SIZE:-16384}"
COMMON=(
  "$MANIFEST"
  --nal-index-path "$NAL_INDEX"
  --checkpoint-dirs "$CKPT"
  --survival-only --no-eval-intra
  --num-clips 20 --prefix-frames 288 --cont-frames 144 --max-window-bytes "$BLOCK_SIZE"
  --seed 42
)

run() {  # name  extra-sampling-args...
  local name="$1"; shift
  echo "===================== [$name] ====================="
  python "$EVAL" "${COMMON[@]}" --out-dir "$OUT_ROOT/$name" "$@"
}

run greedy      --temperature 0.0
run temp1       --temperature 1.0
run avclm_topk  --temperature 1.0 --top-k 50 --top-p 0.9

echo
echo "===================== consolidated ====================="
python3 - "$OUT_ROOT" greedy temp1 avclm_topk <<'PY'
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

# Teacher-forced value metrics are identical across configs -- print once.
tf0, _ = rows(names[0])
if tf0:
    print("\n[teacher-forced value metrics | sampling-independent]")
    print("  argmax-decodes-to: CORRECT value (exact) | LEGAL value (would not desync)")
    cor = tf0.get("element_correct", {})
    leg = tf0.get("element_legal", {})
    for k in sorted(cor, key=lambda k: cor[k]["acc"]):
        legs = f"{leg[k]['acc']:.3f}" if k in leg else "  -  "
        print(f"  {k:24s} correct={cor[k]['acc']:.3f}  legal={legs}  n={cor[k]['n']}")

print("\n[free-run desync by sampling regime]")
hdr = f"  {'config':12s} {'surv_med':>8s} {'surv_mean':>9s} {'top desync region':>22s}"
print(hdr)
for name in names:
    _, c = rows(name)
    if not c:
        print(f"  {name:12s}  (missing)")
        continue
    med = c.get("survival_bytes_median")
    mean = c.get("survival_bytes_mean")
    hist = c.get("desync_region_hist", {}) or {}
    top = max(hist, key=hist.get) if hist else "-"
    mean_s = f"{mean:.1f}" if isinstance(mean, (int, float)) else "-"
    print(f"  {name:12s} {str(med):>8s} {mean_s:>9s} {top:>22s}")
    print(f"               desync_region_hist={hist}")
PY

echo
echo "Sweep complete. Per-config detail: $OUT_ROOT/{greedy,temp1,avclm_topk}/"
