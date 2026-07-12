DATA=/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm
OUT_DIR=${DATA}/runs/byte-phase0-overfit-10v-10m

python scripts/byte/eval/eval_stream_continuation.py \
    "${DATA}/manifest.jsonl" \
    --nal-index-path "${DATA}/nal_index.sqlite" \
    --checkpoint-dirs "${OUT_DIR}/step-00006500" \
    --out-dir "${OUT_DIR}/eval_decode" \
    --train-split-file "${OUT_DIR}/train_split.json" \
    --max-manifest-rows 10 \
    --num-clips 10 \
    --temperature 0 \
    --prefix-frames 4 --cont-frames 2 \
    --max-window-bytes 16384