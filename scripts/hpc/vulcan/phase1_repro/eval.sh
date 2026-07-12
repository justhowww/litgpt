# Phase 1 eval: training-set eval (memorization check) + within-video "val" eval.
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

DATA=/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm
OUT_DIR=${DATA}/runs/byte-phase1-repro-1000v-85m

# Latest checkpoint by default; override with CKPT=step-XXXXXX ./eval.sh
CKPT=${CKPT:-$(basename "$(ls -d "${OUT_DIR}"/step-* 2>/dev/null | sort | tail -1)")}
if [[ -z "${CKPT}" ]]; then
    echo "No checkpoint found under ${OUT_DIR}/step-*" >&2
    exit 1
fi
echo "[phase1 eval] checkpoint=${CKPT}"

# --- (1) Training-set eval: did it memorize what it trained on? ---------------
python scripts/byte/eval/eval_stream_continuation.py \
    "${DATA}/manifest.jsonl" \
    --nal-index-path "${DATA}/nal_index.sqlite" \
    --checkpoint-dirs "${OUT_DIR}/${CKPT}" \
    --out-dir "${OUT_DIR}/eval_decode/train/${CKPT}" \
    --train-split-file "${OUT_DIR}/train_split.json" \
    --max-manifest-rows 1000 \
    --num-clips 20 \
    --seed 42 \
    --temperature 0 \
    --prefix-frames 4 --cont-frames 2 \
    --max-window-bytes 16384

# --- (2) Within-video "val" eval: different seed, no train-split filter -------
# See caveat above -- not guaranteed disjoint from the trained windows.
python scripts/byte/eval/eval_stream_continuation.py \
    "${DATA}/manifest.jsonl" \
    --nal-index-path "${DATA}/nal_index.sqlite" \
    --checkpoint-dirs "${OUT_DIR}/${CKPT}" \
    --out-dir "${OUT_DIR}/eval_decode/val/${CKPT}" \
    --max-manifest-rows 1000 \
    --num-clips 20 \
    --seed 1337 \
    --temperature 0 \
    --prefix-frames 4 --cont-frames 2 \
    --max-window-bytes 16384
