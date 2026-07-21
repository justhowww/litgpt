#!/usr/bin/env bash
# Phase 2 paired FIM evaluation on the exact window-FIM training layout.
#
# Runs the same deterministic train-window holes twice:
#   greedy              learned EOS + oracle length, no syntax mask
#   greedy_full_masked  learned EOS + oracle length, full H.264 syntax mask
#
# The learned-EOS masked run permits EOS only when the generated middle legally
# reconnects to the fixed orphan/suffix.  Oracle length is a diagnostic: if it is
# good while learned EOS is bad, termination rather than byte reconstruction is the
# problem.
#
# Usage:
#   ./scripts/hpc/vulcan/phase2_fim/eval_fim.sh
#   CKPT=step-00006000 NUM_CLIPS=100 ./scripts/hpc/vulcan/phase2_fim/eval_fim.sh
set -euo pipefail

DATA=${DATA:-/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm}
OUT_DIR=${OUT_DIR:-${DATA}/runs/byte-phase2-fim-1000v-85m}
CKPT=${CKPT:-step-00006000}
NUM_CLIPS=${NUM_CLIPS:-100}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
PYTHON_BIN=${PYTHON_BIN:-python}

CHECKPOINT_DIR="${OUT_DIR}/${CKPT}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

echo "[phase2 FIM eval] checkpoint=${CKPT} clips=${NUM_CLIPS} seed=${SEED}"

EVAL=scripts/byte/eval/eval_fim_avclm.py
COMMON=(
    "${DATA}/manifest.jsonl"
    --nal-index-path "${DATA}/nal_index.sqlite"
    --checkpoint-dirs "${CHECKPOINT_DIR}"
    --train-split-file "${OUT_DIR}/train_split.json"
    --eval-split train
    --device "${DEVICE}"
    --max-manifest-rows 1000
    --num-clips "${NUM_CLIPS}"
    --seed "${SEED}"
    --max-window-bytes 16384
    --window-min-frames 2
    --fim-format psm
    --use-eos
    --fim-min-gap 64
    --fim-max-gap 1400
    --slice-header-guard-bytes 64
    --slice-max-mbs 1
    --stop-modes learned_eos oracle_len
    --temperature 0
    --top-k 0
    --top-p 1.0
    --max-gen-multiple 2.0
    --max-gen-extra 16
    --decode
    --timeout-sec 60
    --save-streams
)

EVAL_ROOT="${OUT_DIR}/eval_fim/train/${CKPT}"

run() {
    local name="$1"
    shift
    local dir="${EVAL_ROOT}/${name}"
    # metrics.jsonl and sample_details.jsonl are append-only.  Mixing reruns in one
    # directory would make the summary disagree with the saved configuration.
    if [[ -d "${dir}" ]]; then
        echo "[clean] removing previous ${dir}"
        rm -rf "${dir}"
    fi

    echo "===================== [FIM/${name}] ====================="
    "${PYTHON_BIN}" "${EVAL}" "${COMMON[@]}" --out-dir "${dir}" "$@"
}

run greedy
run greedy_full_masked --mask-illegal-bytes

echo
echo "===================== FIM paired summary ====================="
"${PYTHON_BIN}" - "${EVAL_ROOT}" greedy greedy_full_masked <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = sys.argv[2:]

header = (
    f"{'variant':<20} {'stop':<12} {'n':>4} {'exact':>8} {'byte_acc':>10} "
    f"{'len_match':>10} {'eos_stop':>9} {'parse_ok':>9} {'decode_ok':>10} "
    f"{'eos_blocked':>12}"
)
print(header)
print("-" * len(header))

for name in names:
    path = root / name / "metrics.jsonl"
    if not path.exists():
        print(f"{name:<20} (missing metrics.jsonl)")
        continue
    for line in path.open():
        row = json.loads(line)
        blocked = row.get("mask_eos_blocked_total")
        blocked_text = "-" if blocked is None else str(blocked)
        decode = row.get("decode_ok_rate")
        decode_text = "-" if decode is None else f"{decode:.3f}"
        print(
            f"{name:<20} {row.get('stop_mode', '-'):<12} {row.get('n', 0):>4} "
            f"{row.get('exact_match_rate', 0):>8.3f} "
            f"{row.get('byte_acc_mean', 0):>10.4f} "
            f"{row.get('length_match_rate', 0):>10.3f} "
            f"{row.get('eos_stop_rate', 0):>9.3f} "
            f"{row.get('parse_ok_rate', 0):>9.3f} "
            f"{decode_text:>10} {blocked_text:>12}"
        )

print()
print("Read learned_eos as the deployable result; oracle_len isolates byte reconstruction.")
print("Outputs:", root)
PY
