#!/usr/bin/env bash
# Phase 1 eval: training-set sampling sweep (memorization check, focus of this phase)
# + a single within-video "val" sanity run.
#
# CAVEAT on the val run: phase1 uses split_by_video=0 (within-video split -- see
# train.sh SPLIT_BY_VIDEO), so the held-out set is individual WINDOWS, not videos.
# train_split.json's "videos" list is therefore ~all 1000 videos regardless of
# split (a video only drops out if none of its sampled windows landed in train,
# unlikely at val_fraction=0.01) -- --train-split-file's video-level filter gives
# a real "did we train on these videos" restriction but NOT a leakage-free held-out
# set. The val run below just omits --train-split-file and uses a different --seed
# so it's unlikely (not guaranteed) to hit the exact trained windows. A rigorous
# held-out-window eval would need eval_stream_continuation.py to consume
# train_split.json's "windows" list directly (not yet wired up) -- until then, treat
# "val" here as indicative, not a leakage-free number.
set -euo pipefail

DATA=/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm
OUT_DIR=${DATA}/runs/byte-phase1-repro-1000v-85m

# Latest checkpoint by default; override with CKPT=step-XXXXXX ./eval.sh
CKPT=${CKPT:-$(basename "$(ls -d "${OUT_DIR}"/step-* 2>/dev/null | sort | tail -1)")}
if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found under ${OUT_DIR}/step-*" >&2
    exit 1
fi
echo "[phase1 eval] checkpoint=${CKPT}"

EVAL=scripts/byte/eval/eval_stream_continuation.py
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
COMMON_VAL=(
    "${DATA}/manifest.jsonl"
    --nal-index-path "${DATA}/nal_index.sqlite"
    --checkpoint-dirs "${OUT_DIR}/${CKPT}"
    --max-manifest-rows 1000
    --num-clips 20
    --seed 1337
    --prefix-frames 4 --cont-frames 2
    --max-window-bytes 16384
)

run() {  # split(train|val)  name  out_root  extra-sampling-args...
    local split="$1" name="$2" out_root="$3"; shift 3
    local -n common_ref="COMMON_${split^^}"
    echo "===================== [${split}/${name}] ====================="
    python "${EVAL}" "${common_ref[@]}" --out-dir "${out_root}/${name}" "$@"
}

# --- (1) Training-set sampling sweep: this phase's focus (see 260712 - phase 1
# result.md -- TF is near-perfect on train, greedy free-run success is 0.45;
# comparing sampling regimes here isolates how much of that gap is decoding-strategy
# vs. genuine per-decision compounding). --------------------------------------
TRAIN_OUT="${OUT_DIR}/eval_decode/train/${CKPT}"
run train greedy      "${TRAIN_OUT}" --temperature 0.0
run train temp1       "${TRAIN_OUT}" --temperature 1.0
run train avclm_topk  "${TRAIN_OUT}" --temperature 1.0 --top-k 50 --top-p 0.9

# --- (2) Within-video "val" sanity run (greedy only; see caveat above) --------
VAL_OUT="${OUT_DIR}/eval_decode/val/${CKPT}"
run val greedy "${VAL_OUT}" --temperature 0.0

# --- Consolidate the train sweep -----------------------------------------------
echo
echo "===================== train sweep summary ====================="
python3 - "${TRAIN_OUT}" greedy temp1 avclm_topk <<'PY'
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

print(f"{'sampling':<12} {'decode_rate':>11} {'full_cont':>10} {'survival_mean':>14} {'psnr':>8}  desync_top")
for name in names:
    tf, cont = rows(name)
    if cont is None:
        print(f"{name:<12} (missing metrics.jsonl)")
        continue
    print(
        f"{name:<12} {cont.get('decode_rate', 0):>11.3f} "
        f"{cont.get('full_continuation_rate', 0):>10.3f} "
        f"{cont.get('survival_bytes_mean', 0):>14.1f} "
        f"{cont.get('cont_psnr_mean') or 0:>8.2f}  {cont.get('desync_region_top')}"
    )
PY
