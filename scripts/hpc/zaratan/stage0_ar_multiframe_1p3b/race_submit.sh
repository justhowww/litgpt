#!/bin/bash
# Race several GPU (type x count) variants of the 1.3 B Stage 0 run against each
# other; when one enters RUNNING, cancel the still-pending others.
#
# Racing across GPU TYPES is the main use: e.g. "2 H100 vs 4 A100", take whichever
# the scheduler frees up first. Counts can differ per type.
#
# Correctness rests on submit.sh's `flock -n ${OUT_DIR}/.training.lock`: every
# variant targets the SAME canonical OUT_DIR, so only one ever trains -- a sibling
# that starts in the same poll window fails its flock and exits fast, never
# touching the checkpoint. All variants RESUME the canonical checkpoint, so the
# winner continues progress regardless of which (type,count) wins.
#
# Etiquette: this parks N jobs for one experiment and spends fair-share priority.
# Keep it to 2-3 variants, only when queue wait (not compute) is the bottleneck.
#
# Config (env):
#   SPECS="h100:2 a100:4"   space-separated <type>:<count> variants to race
#                           (type = h100|a100, count <= GPUs-per-node = 4)
#   OUT_DIR=...             canonical run dir (default: the 1.3B run)
#   POLL=30                 seconds between squeue polls
#
# Example:
#   SPECS="h100:2 a100:4" bash scripts/hpc/zaratan/stage0_ar_multiframe_1p3b/race_submit.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
read -r -a SPECS <<< "${SPECS:-h100:4 a100:4}"
POLL=${POLL:-30}
export OUT_DIR=${OUT_DIR:-"$HOME/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage0-ar-multiframe-1p3b"}
export FREE_RUN_INTERVAL=${FREE_RUN_INTERVAL:-0}   # avoid the step-0 free-run hang

echo "Racing variants [${SPECS[*]}] into OUT_DIR=${OUT_DIR}"

declare -A label_of
jobids=()
cancel_all() { [[ ${#jobids[@]} -gt 0 ]] && scancel "${jobids[@]}" 2>/dev/null || true; }

for spec in "${SPECS[@]}"; do
    gpu_type=${spec%%:*}
    ngpu=${spec##*:}
    case "$gpu_type" in h100|a100) ;; *) echo "bad type in '$spec' (h100|a100)" >&2; cancel_all; exit 1 ;; esac
    if ! [[ "$ngpu" =~ ^[1-9][0-9]*$ ]]; then echo "bad count in '$spec'" >&2; cancel_all; exit 1; fi

    out=$(GPU_TYPE="$gpu_type" NGPU="$ngpu" bash "$SCRIPT_DIR/submit_multigpu.sh")
    jid=$(printf '%s\n' "$out" | grep -oE 'training job [0-9]+' | grep -oE '[0-9]+' | tail -1)
    if [[ -z "$jid" ]]; then
        echo "ERROR: could not parse job id for ${spec}. Output:" >&2
        printf '%s\n' "$out" >&2
        cancel_all; exit 1
    fi
    label_of["$jid"]="${gpu_type}x${ngpu}"
    jobids+=("$jid")
    echo "  ${gpu_type} x${ngpu} -> job ${jid}"
done

echo "Watching (first RUNNING wins; flock backstops any same-window double-start)..."
winner=""
while [[ -z "$winner" ]]; do
    alive=0
    for jid in "${jobids[@]}"; do
        state=$(squeue -h -j "$jid" -o "%T" 2>/dev/null || true)
        [[ -n "$state" ]] && alive=1
        if [[ "$state" == "RUNNING" ]]; then winner="$jid"; break; fi
    done
    [[ -n "$winner" ]] && break
    if [[ "$alive" == 0 ]]; then
        echo "No race jobs left in queue and none reached RUNNING (all failed/cancelled?)." >&2
        exit 1
    fi
    echo "$(date '+%F %T') still pending:"
    squeue -j "$(IFS=,; echo "${jobids[*]}")" -o "%.10i %.26j %.8T %.12M %R" || true
    sleep "$POLL"
done

echo
echo "Winner: job ${winner} (${label_of[$winner]}). Cancelling losers..."
for jid in "${jobids[@]}"; do
    if [[ "$jid" != "$winner" ]]; then
        echo "  scancel ${jid} (${label_of[$jid]})"
        scancel "$jid" || true
    fi
done
echo
squeue -j "$winner" -o "%.10i %.26j %.8T %.12M %R" || true
