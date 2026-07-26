#!/bin/bash
# Phase 2 -- AR + masked-span FIM training body. Byte-identical to
# phase1_repro/train.sh except that P_FIM is a knob (phase 1 pins --p-fim 0), so the
# objective is the ONLY thing that differs between the two runs. See
# scripts/byte/train.py for the full arg surface.
#
# Ladder: phase 0 = tiny overfit sanity (memorize + greedy free-run reproduce);
#         phase 1 = small AR reproduction; phase 2 = the same window, AR + FIM.
#
# AR and FIM share the multi-frame window and differ only in masking, so phase 2 is
# phase 1 with p_fim turned up -- that is what makes the AR comparison clean (0616.md).
#
# WHAT THIS RUN CAN AND CANNOT ANSWER. The FIM hole here follows BSCV's operator
# (interior byte excision, uniformly random frame, one hole per frame) but this corpus
# is slice-max-mbs=1, so a 64-1400 B hole spans ~100+ single-macroblock NALs instead of
# sitting inside one slice payload. Survivors still parse (first_mb_in_slice just jumps)
# and the decoder conceals the gap, so BSCV's "present but desynced slice" cannot occur
# -- per-MB slicing is a desync firewall every ~10 bytes. So:
#   ANSWERS:     does adding FIM damage AR? (road-list phase-2 signals 1 and 2)
#   DOES NOT:    how well FIM repairs a corrupted bitstream. That number is not a task
#                result on this corpus; it needs a one-slice-per-frame corpus.
# See 04 - projects/real-time-diffusion-video-decoder/corruption-vs-bscv.md.

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/vulcan/env.sh"

: "${MANIFEST:?Set MANIFEST to the scratch-resident corpus manifest.jsonl}"

# Model arch (head_dim 64 by convention: N_HEAD = N_EMBD/64).
N_LAYER=${N_LAYER:-12}
N_EMBD=${N_EMBD:-768}
N_HEAD=${N_HEAD:-12}
MODEL_TAG=${MODEL_TAG:-byte-phase}

OUT_DIR=${OUT_DIR:?Set OUT_DIR to the run directory}
NAL_INDEX=${NAL_INDEX:-"$(dirname "${MANIFEST}")/nal_index.sqlite"}
BLOCK_SIZE=${BLOCK_SIZE:-16384}
BYTE_PATCH_SIZE=${BYTE_PATCH_SIZE:-1}
STEPS=${STEPS:-100000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICES=${DEVICES:-1}
NUM_NODES=${NUM_NODES:-1}
WINDOW_MIN_FRAMES=${WINDOW_MIN_FRAMES:-2}
LOGGER_NAME=${LOGGER_NAME:-tensorboard}
WARMUP_STEPS=${WARMUP_STEPS:-$((STEPS / 50))}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
MIN_LEARNING_RATE=${MIN_LEARNING_RATE:-3e-5}
VAL_FRACTION=${VAL_FRACTION:-0.01}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
# MAX_ROWS = number of videos (0 = full corpus). SPLIT_BY_VIDEO=1 => held-out-video split
# (default is within-video, window-level). FREE_RUN_INTERVAL>0 turns on the in-training
# free-run (H0) probe -- set FREE_RUN_TEMP=0 for the greedy overfit-reproduction signal.
MAX_ROWS=${MAX_ROWS:-0}
SPLIT_BY_VIDEO=${SPLIT_BY_VIDEO:-0}
# --- FIM knobs (the only objective difference from phase 1) -----------------------
# P_FIM is the per-sample probability of an infill sample; the rest stay AR, so the
# run trains both modes over the same windows. P_FIM=0 reproduces phase 1 exactly.
P_FIM=${P_FIM:-0.5}
FIXED_FIM_HOLES=${FIXED_FIM_HOLES:-0}
FIM_FORMAT=${FIM_FORMAT:-psm}
# On by default: train a terminator (SEQ_EOS after f_middle) rather than relying on
# an oracle-supplied span length, which is a training-time convenience a real
# deployment wouldn't have. Lifts the vocab (vocab_size_for_fim_format); phase 1
# already uses a different fim_format/vocab, so there is no warm-start path to break.
USE_EOS=${USE_EOS:-1}
# Hole size, in bytes. BSCV excises 1024-2048 B; 64-1400 keeps the operator's shape
# while staying inside one frame of this corpus (~1.8 KB at 256x144/QP37 per-MB).
FIM_MIN_GAP=${FIM_MIN_GAP:-64}
FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
# Untouchable head of the damaged frame. Mirrors BSCV keeping the start code + NAL
# header, and keeps the frame's first NAL -- which carries the first_mb_in_slice == 0
# that MAKES it a frame boundary -- out of the hole.
SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}
FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}
FREE_RUN_TEMP=${FREE_RUN_TEMP:-1.0}
FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-288}
FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-144}
FREE_RUN_SLICE_LAYOUT=${FREE_RUN_SLICE_LAYOUT:-macroblock}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable on the compute node: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    echo "Build nal_index.sqlite (scripts/byte build_byte_nal_index) before training." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cmd=(
    python -u scripts/byte/train.py
    "${MANIFEST}"
    --nal-index-path "${NAL_INDEX}"
    --out-dir "${OUT_DIR}"
    --model-name "${MODEL_TAG}"
    --n-layer "${N_LAYER}"
    --n-embd "${N_EMBD}"
    --n-head "${N_HEAD}"
    --dataset-mode window
    --p-fim "${P_FIM}"
    --fim-format "${FIM_FORMAT}"
    --fim-min-gap "${FIM_MIN_GAP}"
    --fim-max-gap "${FIM_MAX_GAP}"
    --slice-header-guard-bytes "${SLICE_HEADER_GUARD_BYTES}"
    --window-min-frames "${WINDOW_MIN_FRAMES}"
    --block-size "${BLOCK_SIZE}"
    --byte-patch-size "${BYTE_PATCH_SIZE}"
    --steps "${STEPS}"
    --warmup-steps "${WARMUP_STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --devices "${DEVICES}"
    --num-nodes "${NUM_NODES}"
    --learning-rate "${LEARNING_RATE}"
    --min-learning-rate "${MIN_LEARNING_RATE}"
    --val-fraction "${VAL_FRACTION}"
    --precision bf16-mixed
    --logger-name "${LOGGER_NAME}"
    --num-workers "${NUM_WORKERS}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters 20
    --save-interval "${SAVE_INTERVAL}"
    --resume
)

if [[ "${MAX_ROWS}" != "0" ]]; then
    cmd+=(--max-manifest-rows "${MAX_ROWS}")
fi
if [[ "${SPLIT_BY_VIDEO}" == "1" ]]; then
    cmd+=(--split-by-video)
fi
if [[ "${USE_EOS}" == "1" ]]; then
    cmd+=(--use-eos)
fi
if [[ "${FIXED_FIM_HOLES}" == "1" ]]; then
    cmd+=(--fixed-fim-holes)
fi
# NO_ENCODING is not optional here: train.py rejects window FIM with offset ids on,
# because a hole spanning ~100 NALs has no single within-NAL offset to encode.
# Phase 1 runs NO_ENCODING=1 anyway (AVC-LM-faithful), so this also keeps the arms
# matched -- but fail here with the reason rather than at argparse.
if [[ "${NO_ENCODING:-0}" == "1" ]]; then
    cmd+=(--no-region-id --no-offset-id)
elif [[ "${P_FIM}" != "0" ]]; then
    echo "NO_ENCODING=0 with P_FIM=${P_FIM}: window FIM requires --no-offset-id" >&2
    echo "(see scripts/byte/train.py). Phase 1 also ran NO_ENCODING=1." >&2
    exit 1
fi
if [[ "${FREE_RUN_INTERVAL}" != "0" ]]; then
    cmd+=(
        --free-run-eval-interval "${FREE_RUN_INTERVAL}"
        --free-run-temperature "${FREE_RUN_TEMP}"
        --free-run-eval-clips "${FREE_RUN_CLIPS}"
        --free-run-prefix-frames "${FREE_RUN_PREFIX_FRAMES}"
        --free-run-cont-frames "${FREE_RUN_CONT_FRAMES}"
        --free-run-slice-layout "${FREE_RUN_SLICE_LAYOUT}"
    )
fi

COMPILE=${COMPILE:-1}
if [[ "${COMPILE}" == "1" ]]; then
    cmd+=(--compile)
fi

echo "[phase2-fim] model=${MODEL_TAG} n_layer=${N_LAYER} n_embd=${N_EMBD} n_head=${N_HEAD} block=${BLOCK_SIZE} patch=${BYTE_PATCH_SIZE} raw_byte_capacity~=$((BLOCK_SIZE * BYTE_PATCH_SIZE)) steps=${STEPS} warmup=${WARMUP_STEPS} gbs=${GLOBAL_BATCH_SIZE} max_rows=${MAX_ROWS} split_by_video=${SPLIT_BY_VIDEO} no_encoding=${NO_ENCODING:-0} free_run_interval=${FREE_RUN_INTERVAL} free_run_temp=${FREE_RUN_TEMP} free_run_slice_layout=${FREE_RUN_SLICE_LAYOUT}"
echo "[phase2-fim] p_fim=${P_FIM} fixed_fim_holes=${FIXED_FIM_HOLES} fim_format=${FIM_FORMAT} use_eos=${USE_EOS} gap=[${FIM_MIN_GAP},${FIM_MAX_GAP}] frame_guard=${SLICE_HEADER_GUARD_BYTES}"

flock -n "${OUT_DIR}/.training.lock" srun --unbuffered "${cmd[@]}" || {
    echo "Another training job is already using OUT_DIR=${OUT_DIR}" >&2
    exit 1
}
