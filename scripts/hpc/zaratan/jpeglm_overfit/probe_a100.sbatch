#!/bin/bash
# Parser-state linear probes: extract hidden states for each run (plus an untrained
# control with the first run's architecture), then train and report the probes.
#
#   PROBE_RUNS="G2=<run dir> G4=<run dir> G4L=<run dir>" sbatch probe_a100.sbatch
#
# Optional: PROBE_CHECKPOINT (best), PROBE_NUM_CLIPS (300), STAGED_CORPUS,
# PROBE_OUT (default: <first run dir>/probe).

#SBATCH --job-name=jpeglm-probe
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail
: "${PROBE_RUNS:?Set PROBE_RUNS=\"name=run_dir ...\"}"
REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"
STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm"}
CKPT=${PROBE_CHECKPOINT:-best}
first_dir=$(echo "${PROBE_RUNS}" | awk '{print $1}' | cut -d= -f2)
OUT=${PROBE_OUT:-"${first_dir}/probe"}
mkdir -p "${OUT}"

extract() {  # name run_dir [--random-init]
    local name=$1 dir=$2; shift 2
    python -u scripts/byte/eval/probe_parser_state.py "${dir}" \
        --checkpoint "${CKPT}" \
        --manifest "${STAGED_CORPUS}/manifest.jsonl" \
        --nal-index-path "${STAGED_CORPUS}/nal_index.sqlite" \
        --num-clips "${PROBE_NUM_CLIPS:-300}" \
        --out "${OUT}/${name}.npz" "$@"
}

npz_args=()
for spec in ${PROBE_RUNS}; do
    name=${spec%%=*}; dir=${spec#*=}
    [[ -e "${OUT}/${name}.npz" ]] || extract "${name}" "${dir}"
    npz_args+=("${name}=${OUT}/${name}.npz")
done
[[ -e "${OUT}/random.npz" ]] || extract random "${first_dir}" --random-init
npz_args+=("random=${OUT}/random.npz")

python -u scripts/byte/eval/probe_train.py --npz "${npz_args[@]}" --out "${OUT}/probe_report.json"
