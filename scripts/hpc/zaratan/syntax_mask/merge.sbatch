#!/bin/bash
# Merge all shard_*.jsonl of a syntax-mask scan into RUN_DIR/summary.json.

#SBATCH --job-name=syntax-mask-merge
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
: "${RUN_DIR:?Set RUN_DIR}"
LAYOUT=${LAYOUT:-frame}
PROJECT_SCRATCH=${PROJECT_SCRATCH:-"/home/${USER}/scratch.metzler-prj"}
CONDA_ROOT=${CONDA_ROOT:-"${PROJECT_SCRATCH}/miniforge3"}
CONDA_ENV=${CONDA_ENV:-"litpt"}
PYTHON_BIN=${PYTHON_BIN:-"${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"}

cd "${REPO_ROOT}"
shopt -s nullglob
shards=("${RUN_DIR}"/shard_*.jsonl)
echo "Merging ${#shards[@]} shard files from ${RUN_DIR}"
"${PYTHON_BIN}" -m syntax_mask.merge "${shards[@]}" --layout "${LAYOUT}" \
    --json "${RUN_DIR}/summary.json" | tee "${RUN_DIR}/summary.txt" \
    || echo "merged summary reports failures; see ${RUN_DIR}/summary.json"
