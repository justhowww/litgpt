#!/bin/bash
#SBATCH --job-name=byte-conditioning-summary
#SBATCH --account=metzler-prj-cmsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

: "${ABLATION_ROOT:?Set ABLATION_ROOT to the matched-run output directory}"

python scripts/summarize_conditioning_ablations.py "${ABLATION_ROOT}"
