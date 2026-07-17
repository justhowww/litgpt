#!/bin/bash
# Phase 2 -- AR + masked-span FIM on UMD Vulcan (2x a6000, FSDP). Same corpus, arch,
# split, batch and LR horizon as phase 1 (submit.sh there); the ONLY difference is
# P_FIM > 0. That is deliberate: phase 2's question is whether adding the FIM
# objective damages AR, and any second change would confound the comparison.
#
# Read it against phase 1 at the same step count:
#   signal 1  val_loss_ar + the training-set free-run eval  -- did AR regress?
#   signal 2  within-video eval                             -- ok to fail (1000 videos
#                                                              may lack syntax diversity)
#   signal 3  val_loss_fim                                  -- is FIM learning at all?
#
# Signal 3 is CE only. The reconstruction probe (PSNR/SSIM on a re-decoded stream) is
# slice-only -- it reads sample.target_index, which WindowSample does not have -- so
# train.py rejects it in window mode. That is not much of a loss here: on this per-MB
# corpus a FIM repair number is not a bitstream-repair result anyway (see train.sh's
# header and corruption-vs-bscv.md). Run on a Vulcan login node.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

STAGED_CORPUS=${STAGED_CORPUS:-"/fs/nexus-projects/time-control-videogen/OpenVid-1M_Data/data-avclm"}

# --- Held identical to phase 1 -- do not drift these without re-baselining ---------
export N_LAYER=${N_LAYER:-12}
export N_EMBD=${N_EMBD:-768}
export N_HEAD=${N_HEAD:-12}
export DEVICES=${DEVICES:-2}                     # 2x a6000 FSDP; keep in sync with --gres
export NO_ENCODING=${NO_ENCODING:-1}             # AVC-LM-faithful; also required by window FIM
export NUM_WORKERS=${NUM_WORKERS:-3}
export MAX_ROWS=${MAX_ROWS:-1000}                # 1000 videos
export STEPS=${STEPS:-1000000}                   # real LR horizon (cosine anneals over this)
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}   # per-GPU; 2 leaves margin for eval/save spikes
export VAL_FRACTION=${VAL_FRACTION:-0.01}
export EVAL_INTERVAL=${EVAL_INTERVAL:-250}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-1000}
export FREE_RUN_TEMP=${FREE_RUN_TEMP:-0.0}
export FREE_RUN_CLIPS=${FREE_RUN_CLIPS:-8}
# REAL frames, counted by first_mb_in_slice == 0 -- NOT macroblocks. A 16384-byte
# window on this corpus holds ~12 real frames (144 MB/frame at ~10 B/MB), so
# prepare_free_run_samples needs prefix+cont < ~12 or it skips EVERY window and the
# probe silently reports nothing. phase1_repro/submit.sh still carries 288/144, which
# are macroblock-era numbers the frame-counting fix reinterpreted as frames: they
# require 289 real frames and yield zero samples. 4/2 matches the standalone eval in
# eval.sh and in 260712 - phase 1 result.md, so the two agree.
export FREE_RUN_PREFIX_FRAMES=${FREE_RUN_PREFIX_FRAMES:-4}
export FREE_RUN_CONT_FRAMES=${FREE_RUN_CONT_FRAMES:-2}

# --- The objective change ---------------------------------------------------------
# 0.5 mirrors the stage2_fim_psm default: half the samples AR, half infill, so AR does
# not go unrehearsed and signal 1 stays readable. Lower it (e.g. 0.25) if AR regresses
# and you want to find the interference threshold rather than just observe it.
export P_FIM=${P_FIM:-0.5}
export FIM_FORMAT=${FIM_FORMAT:-psm}
export USE_EOS=${USE_EOS:-0}                     # oracle length; USE_EOS=1 changes the vocab
export FIM_MIN_GAP=${FIM_MIN_GAP:-64}
export FIM_MAX_GAP=${FIM_MAX_GAP:-1400}
export SLICE_HEADER_GUARD_BYTES=${SLICE_HEADER_GUARD_BYTES:-64}

export MODEL_TAG=${MODEL_TAG:-byte-phase2-fim-1000v-85m}
export OUT_DIR=${OUT_DIR:-"${STAGED_CORPUS}/runs/${MODEL_TAG}"}
MANIFEST="${STAGED_CORPUS}/manifest.jsonl"
mkdir -p "${OUT_DIR}/logs"
export REPO_ROOT MANIFEST OUT_DIR STAGED_CORPUS

sbatch_args=(--parsable --export=ALL
    --output="${OUT_DIR}/logs/%x-%j.out" --error="${OUT_DIR}/logs/%x-%j.err")
[[ -n "${EXCLUDE_NODES:-}" ]] && sbatch_args+=(--exclude="${EXCLUDE_NODES}")

echo "[phase2-fim/vulcan] ${MAX_ROWS} videos / ~85M (${N_LAYER}/${N_EMBD}/${N_HEAD}) / ${DEVICES}x a6000 / p_fim=${P_FIM} / steps=${STEPS}"
echo "  OUT_DIR=${OUT_DIR}"
job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/train.sbatch")
echo "Submitted phase2-fim job ${job_id}"
