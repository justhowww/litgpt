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
    local dir="${out_root}/${name}"
    # Start from a clean dir. metrics.jsonl is opened APPEND-only while config.json and
    # summary.csv are overwritten, so re-running into a populated dir interleaves two
    # runs' rows and destroys the only record of which config produced them -- the
    # summary below then silently reports the last rows against the wrong config. (The
    # previous greedy/ shipped exactly this: two continuation rows, full_cont 0.05 and
    # 0.75, indistinguishable without the intra PSNR as a tie-break.)
    if [[ -d "${dir}" ]]; then
        echo "[clean] removing previous ${dir}"
        rm -rf "${dir}"
    fi
    echo "===================== [${split}/${name}] ====================="
    python "${EVAL}" "${common_ref[@]}" --out-dir "${dir}" "$@"
}

# --- (1) Training-set sampling sweep: this phase's focus (see 260712 - phase 1
# result.md -- TF is near-perfect on train, greedy free-run success is 0.45;
# comparing sampling regimes here isolates how much of that gap is decoding-strategy
# vs. genuine per-decision compounding). --------------------------------------
TRAIN_OUT="${OUT_DIR}/eval_decode/train/${CKPT}"
# greedy vs greedy_masked is the A/B that isolates constrained decoding: same
# checkpoint, clips and seed, differing only in --mask-illegal-bytes. They must write
# to DIFFERENT --out-dirs -- metrics.jsonl is opened append-only while config.json and
# summary.csv are overwritten, so sharing a name silently mixes the two runs' rows and
# destroys the record of which config produced them.
run train greedy        "${TRAIN_OUT}" --temperature 0.0
# --mask-debug keeps the permissive-fallback path visible: with v1's headers
# unconstrained, an illegal mb_type/cbp flips the automaton to "unknown" and quietly
# drops strict masking for the rest of that NAL. Without this you cannot tell "the mask
# worked" from "the mask switched itself off".
run train greedy_masked "${TRAIN_OUT}" --temperature 0.0 --mask-illegal-bytes --mask-debug
# run train temp1       "${TRAIN_OUT}" --temperature 1.0
# run train avclm_topk  "${TRAIN_OUT}" --temperature 1.0 --top-k 50 --top-p 0.9

# # --- (2) Within-video "val" sanity run (greedy only; see caveat above) --------
# VAL_OUT="${OUT_DIR}/eval_decode/val/${CKPT}"
# run val greedy "${VAL_OUT}" --temperature 0.0

# --- Consolidate the train sweep -----------------------------------------------
# Headline: full_cont for greedy vs greedy_masked. The mask's claim is that it turns
# decodability from a probabilistic AND-chain into an invariant, so full_cont should
# rise and the BitReaderError desyncs (running off the end of a mis-parsed field)
# should collapse -- desync_reasons is printed to check that second half.
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
    _, cont = rows(name)
    if cont is not None:
        print(f"{name:<14} desync_reasons: {cont.get('desync_reason_hist')}")
PY

# --- Per-NAL termination diagnostics -------------------------------------------
# desync_reason=BitReaderError only says the parser wanted more bits than the NAL held;
# it does not say WHY the NAL ended. This replays the persisted streams (no GPU, no
# model) and answers the decisive question: at each emitted boundary, was the automaton
# already in `done`? stage != done => the model learned how many macroblock NALs to
# emit but not when one is syntactically complete. stop_reason (recorded by
# free_run_rollout, whose six exits share one return) separately rules max_gen in/out.
#
# Runs AFTER the summary and cannot abort it: under `set -e` a non-zero exit here (e.g.
# results predating stream persistence) would otherwise kill the script before the
# headline A/B ever printed. A diagnostic must never suppress the result it explains.
NAL_TERM=scripts/byte/eval/nal_termination.py
for name in greedy greedy_masked; do
    echo
    echo "===================== [nal-termination/${name}] ====================="
    python "${NAL_TERM}" "${TRAIN_OUT}/${name}" --slice-max-mbs 1 \
        || echo "[warn] nal_termination failed for ${name} (summary above is unaffected)"
done
