#!/bin/bash
# Resume from the newest checkpoint in OUT_DIR. Optionally chain after a job:
#   bash resubmit.sh [JOB_ID]

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if (( $# > 1 )); then
    echo "Usage: $0 [JOB_ID]" >&2
    exit 2
fi
if (( $# == 1 )); then
    export AFTER_JOBID=$1
    # A wall-time-limited training allocation commonly exits non-zero. Resume
    # after it ends regardless; the newest complete checkpoint remains safe.
    export DEPENDENCY_TYPE=afterany
fi
exec bash "${SCRIPT_DIR}/submit.sh"
