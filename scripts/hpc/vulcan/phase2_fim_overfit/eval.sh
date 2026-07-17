#!/usr/bin/env bash
# Phase 2 overfit eval: the same AR probes as phase2_fim/eval.sh, pointed at the
# 10-video run. This is a sanity gate -- greedy full_cont should climb toward 1.0
# fast (it's an overfit set) and val_loss_fim (from training's own logs, not this
# script) should visibly drop. Neither number here is phase 2's actual result; at
# 10 videos there isn't enough syntax/content diversity to read interference from,
# only "can this objective be learned at all."
set -euo pipefail

DATA=/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm
OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-phase2-fim-overfit-10v-10m}

CKPT=${CKPT:-$(basename "$(ls -d "${OUT_DIR}"/step-* 2>/dev/null | sort | tail -1)")}
if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found under ${OUT_DIR}/step-*" >&2
    exit 1
fi
echo "[phase2-overfit eval] checkpoint=${CKPT}"

EVAL=scripts/byte/eval/eval_stream_continuation.py
COMMON_TRAIN=(
    "${DATA}/manifest.jsonl"
    --nal-index-path "${DATA}/nal_index.sqlite"
    --checkpoint-dirs "${OUT_DIR}/${CKPT}"
    --train-split-file "${OUT_DIR}/train_split.json"
    --max-manifest-rows 10
    --num-clips 10
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
print("Sanity gate, not phase-2's result: full_cont should climb toward 1.0 on this")
print("10-video overfit set. If it does not, debug HERE (10 videos, cheap iteration)")
print("before trusting anything from the full 1000-video run.")
PY
