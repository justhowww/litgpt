#!/bin/bash
# One shard of the whole-corpus syntax-mask scan (Slurm array task = shard).
# Submit through submit_scan.sh, which sets MANIFEST, RUN_DIR, NUM_SHARDS, ...
# Every task runs with --resume, so resubmitting a timed-out shard continues it.

#SBATCH --job-name=syntax-mask-scan
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
: "${MANIFEST:?Set MANIFEST to the scratch-resident manifest.jsonl}"
: "${RUN_DIR:?Set RUN_DIR}"
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
LAYOUT=${LAYOUT:-frame}
WORKERS=${WORKERS:-"${SLURM_CPUS_PER_TASK}"}
REFERENCE_RATE=${REFERENCE_RATE:-0.001}
PROBE_RATE=${PROBE_RATE:-0.0005}
PROBE_LEN=${PROBE_LEN:-64}
LIMIT=${LIMIT:-0}
DUMP_DIR=${DUMP_DIR:-}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-"/home/${USER}/scratch.metzler-prj"}
CONDA_ROOT=${CONDA_ROOT:-"${PROJECT_SCRATCH}/miniforge3"}
CONDA_ENV=${CONDA_ENV:-"litpt"}
PYTHON_BIN=${PYTHON_BIN:-"${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Shard ${SHARD}/${NUM_SHARDS} on $(/usr/bin/hostname) at $(/usr/bin/date --iso-8601=seconds)"
echo "Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
echo "Manifest: ${MANIFEST}  layout=${LAYOUT}  workers=${WORKERS}"

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}"

cmd=(
    "${PYTHON_BIN}" -m syntax_mask.scan
    --manifest "${MANIFEST}"
    --layout "${LAYOUT}"
    --num-shards "${NUM_SHARDS}" --shard-index "${SHARD}"
    --workers "${WORKERS}"
    --reference-rate "${REFERENCE_RATE}"
    --probe-rate "${PROBE_RATE}" --probe-len "${PROBE_LEN}"
    --seed "${SHARD}"
    --out-jsonl "${RUN_DIR}/shard_$(printf %04d "${SHARD}").jsonl"
    --resume
)
[[ "${LIMIT}" != 0 ]] && cmd+=(--limit "${LIMIT}")
[[ -n "${DUMP_DIR}" ]] && cmd+=(--dump-dir "${DUMP_DIR}")

echo "${cmd[*]}"
# A non-zero exit only means "some file failed a check"; the shard's results
# are complete either way and merge.sbatch reports them.
"${cmd[@]}" || echo "scan reported failures (exit $?); see merged summary"
echo "Shard ${SHARD} finished at $(/usr/bin/date --iso-8601=seconds)"
