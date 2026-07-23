#!/usr/bin/env bash
# Replay the exact fixed FIM holes recorded by the fixed-overfit training run.
# This primary sanity check is deliberately unmasked: the model itself must memorize
# both the missing bytes and EOS. The evaluator aborts if any replayed hole differs
# from the hole recorded in train_split.json.
set -euo pipefail

DATA=${DATA:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm}
OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-fim-fixed-hole-overfit-10m}
CKPT=${CKPT:-$(basename "$(ls -d "${OUT_DIR}"/step-* 2>/dev/null | sort | tail -1)")}
NUM_CLIPS=${NUM_CLIPS:-20}
SEED=${SEED:-42}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -z "${CKPT}" || ! -d "${OUT_DIR}/${CKPT}" ]]; then
    echo "No checkpoint found under ${OUT_DIR}/step-*" >&2
    exit 1
fi

EVAL_ROOT="${OUT_DIR}/eval_fim_fixed/train/${CKPT}/greedy"
if [[ -d "${EVAL_ROOT}" ]]; then
    echo "[clean] removing previous ${EVAL_ROOT}"
    rm -rf "${EVAL_ROOT}"
fi

echo "[fixed-hole FIM eval] checkpoint=${CKPT} clips=${NUM_CLIPS}"
"${PYTHON_BIN}" scripts/byte/eval/eval_fim_avclm.py \
    "${DATA}/manifest.jsonl" \
    --nal-index-path "${DATA}/nal_index.sqlite" \
    --checkpoint-dirs "${OUT_DIR}/${CKPT}" \
    --train-split-file "${OUT_DIR}/train_split.json" \
    --eval-split train \
    --out-dir "${EVAL_ROOT}" \
    --max-manifest-rows 21 \
    --num-clips "${NUM_CLIPS}" \
    --num-visualizations 8 \
    --seed "${SEED}" \
    --max-window-bytes 16384 \
    --window-min-frames 2 \
    --fim-format psm \
    --use-eos \
    --fim-min-gap 64 \
    --fim-max-gap 1400 \
    --slice-header-guard-bytes 64 \
    --slice-max-mbs 1 \
    --stop-modes learned_eos \
    --temperature 0 \
    --top-k 0 \
    --top-p 1.0 \
    --max-gen-bytes 4096 \
    --decode \
    --timeout-sec 60 \
    --save-streams

"${PYTHON_BIN}" - "${EVAL_ROOT}/metrics.jsonl" "${NUM_CLIPS}" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
expected = int(sys.argv[2])
for row in rows:
    verified = int(row.get("exact_training_holes_verified", 0))
    available = int(row.get("fixed_training_holes_available", 0))
    if verified != expected or available != expected:
        raise SystemExit(
            f"expected exactly {expected} fixed training holes, "
            f"but verified={verified} available={available}"
        )
    print(
        "fixed-hole result:"
        f" verified={verified}/{available}"
        f" tf_byte_acc={row.get('tf_byte_acc_mean')}"
        f" tf_eos_acc={row.get('tf_eos_acc_mean')}"
        f" ce={row.get('tf_ce_nats_mean')}"
        f" termination={row.get('termination_success_rate')}"
        f" end_to_end={row.get('end_to_end_success_rate')}"
    )
PY
