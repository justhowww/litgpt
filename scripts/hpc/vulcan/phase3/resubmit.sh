#!/usr/bin/env bash
# Chain the next phase-3 training segment to start once <AFTER_JOBID> ends -- needed
# because vulcan-high's QoS caps a single job at 1-12:00:00 wall time, so a run
# targeting STEPS=1000000 will not fit in one job and must be split across chained
# submissions, each resuming from OUT_DIR's latest checkpoint (train.sh already
# passes --resume, so this is automatic as long as OUT_DIR matches the prior job's).
#
# Uses --dependency=afterany (via submit.sh's AFTER_JOBID support), not afterok: the
# prior job will almost always end by hitting the wall-time limit rather than exiting
# 0, and afterok would never fire in that case, silently stalling the chain forever.
#
# Usage:
#   scripts/hpc/vulcan/phase3/resubmit.sh <AFTER_JOBID>
#
# IMPORTANT: run with the SAME env var overrides (OUT_DIR, MODEL_TAG,
# BYTE_PATCH_SIZE, arch, etc.) as
# the original submission, or defaults must match it -- otherwise this resolves to a
# DIFFERENT OUT_DIR and starts a fresh run instead of continuing. Simplest: don't
# override MODEL_TAG/OUT_DIR at all; submit.sh's defaults then match automatically.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <AFTER_JOBID>" >&2
    exit 1
fi

export AFTER_JOBID="$1"
echo "[phase3-resubmit] chaining after job ${AFTER_JOBID}"
bash "${SCRIPT_DIR}/submit.sh"
