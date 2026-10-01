#!/bin/bash
# AdamW-state probe for one MEGABYTE checkpoint (CPU only; reads the full training state).
#
#   RUN_DIR=/path/to/runs/<run> CHECKPOINT=latest COMPARE=step-00200000 \
#     sbatch scripts/hpc/zaratan/jpeglm_pretrain/optimizer_probe_cpu.sbatch
#
# Writes RUN_DIR/eval_global_ablation/optimizer_probe-<CHECKPOINT>.json (never overwrites).

#SBATCH --job-name=megabyte-optimizer-probe
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

: "${RUN_DIR:?Set RUN_DIR to the training run directory}"
CHECKPOINT=${CHECKPOINT:-latest}
COMPARE=${COMPARE:-}
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

cmd=(python -u scripts/byte/eval/megabyte_optimizer_probe.py "${RUN_DIR}/${CHECKPOINT}"
     --out "${RUN_DIR}/eval_global_ablation/optimizer_probe-${CHECKPOINT}.json")
[[ -n "${COMPARE}" ]] && cmd+=(--compare "${RUN_DIR}/${COMPARE}")
cd "${REPO_ROOT}"
"${cmd[@]}"
