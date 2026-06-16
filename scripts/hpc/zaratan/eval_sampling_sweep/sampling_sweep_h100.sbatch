#!/bin/bash
# Offline sampling-diagnostic sweep for byte FIM checkpoints.

#SBATCH --job-name=byte-eval-sweep
#SBATCH --account=metzler-prj-cmsc
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
source "${REPO_ROOT}/scripts/hpc/zaratan/env.sh"

STAGED_CORPUS=${STAGED_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"}
RUN_NAME=${RUN_NAME:-"byte-stage2-fim-psm-ce-only-within-video-1k-8k"}
RUN_DIR=${RUN_DIR:-"${STAGED_CORPUS}/runs/${RUN_NAME}"}
MANIFEST=${MANIFEST:-"${STAGED_CORPUS}/manifest.jsonl"}
NAL_INDEX=${NAL_INDEX:-"${STAGED_CORPUS}/nal_index.sqlite"}
FFMPEG_BINARY=${FFMPEG_BINARY:-"${CONDA_ROOT}/bin/ffmpeg"}
OUT_ROOT=${OUT_ROOT:-"${RUN_DIR}/offline_sampling_sweep"}

# Space-separated optimizer steps. The wrapper defaults to 500..4500.
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"500 1000 1500 2000 2500 3000 3500 4000 4500"}
NUM_SAMPLES=${NUM_SAMPLES:-20}
NUM_VISUALIZATIONS=${NUM_VISUALIZATIONS:-4}
BEST_OF_N=${BEST_OF_N:-64}
FIM_MIN_GAP=${FIM_MIN_GAP:-1024}
FIM_MAX_GAP=${FIM_MAX_GAP:-8192}
VAL_FRACTION=${VAL_FRACTION:-0.05}
SPLIT_BY_VIDEO=${SPLIT_BY_VIDEO:-0}
RESUME_SKIP=${RESUME_SKIP:-1}

if [[ ! -r "${MANIFEST}" ]]; then
    echo "Manifest is not readable: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -r "${NAL_INDEX}" ]]; then
    echo "NAL index is not readable: ${NAL_INDEX}" >&2
    exit 1
fi
if [[ ! -x "${FFMPEG_BINARY}" ]]; then
    echo "FFmpeg executable is not available: ${FFMPEG_BINARY}" >&2
    exit 1
fi

checkpoint_dirs=()
for step in ${CHECKPOINT_STEPS}; do
    printf -v step_dir "step-%08d" "${step}"
    checkpoint_dir="${RUN_DIR}/${step_dir}"
    if [[ ! -r "${checkpoint_dir}/lit_model.pth" ]]; then
        echo "Checkpoint is missing: ${checkpoint_dir}/lit_model.pth" >&2
        exit 1
    fi
    checkpoint_dirs+=("${checkpoint_dir}")
done

mkdir -p "${OUT_ROOT}"
COMBINED_SUMMARY="${OUT_ROOT}/sampling_sweep_summary.csv"
SUMMARY_HEADER_WRITTEN=0
echo "Repository: ${REPO_ROOT}"
echo "Run dir: ${RUN_DIR}"
echo "Output root: ${OUT_ROOT}"
echo "Checkpoints: ${checkpoint_dirs[*]}"
echo "NUM_SAMPLES=${NUM_SAMPLES} BEST_OF_N=${BEST_OF_N}"

run_setting() {
    local name=$1
    local temperature=$2
    local top_k=$3
    local top_p=$4
    local out_dir="${OUT_ROOT}/${name}"
    if [[ "${RESUME_SKIP}" == "1" && -s "${out_dir}/summary.csv" ]]; then
        echo "[skip] ${name}: ${out_dir}/summary.csv exists"
        append_summary "${name}" "${temperature}" "${top_k}" "${top_p}" "${out_dir}/summary.csv"
        return 0
    fi
    echo "[run] ${name}: temperature=${temperature} top_k=${top_k} top_p=${top_p}"
    cmd=(
        python scripts/byte/eval/eval_checkpoints.py
        "${MANIFEST}"
        --nal-index-path "${NAL_INDEX}"
        --checkpoint-dirs "${checkpoint_dirs[@]}"
        --out-dir "${out_dir}"
        --num-samples "${NUM_SAMPLES}"
        --num-visualizations "${NUM_VISUALIZATIONS}"
        --best-of-n "${BEST_OF_N}"
        --temperature "${temperature}"
        --top-k "${top_k}"
        --top-p "${top_p}"
        --fim-min-gap "${FIM_MIN_GAP}"
        --fim-max-gap "${FIM_MAX_GAP}"
        --val-fraction "${VAL_FRACTION}"
        --ffmpeg-binary "${FFMPEG_BINARY}"
    )
    if [[ "${SPLIT_BY_VIDEO}" == "1" ]]; then
        cmd+=(--split-by-video)
    fi
    srun "${cmd[@]}"
    append_summary "${name}" "${temperature}" "${top_k}" "${top_p}" "${out_dir}/summary.csv"
}

append_summary() {
    local setting_name=$1
    local temperature=$2
    local top_k=$3
    local top_p=$4
    local summary_csv=$5
    if [[ ! -s "${summary_csv}" ]]; then
        echo "Missing summary for ${setting_name}: ${summary_csv}" >&2
        return 1
    fi
    if [[ ${SUMMARY_HEADER_WRITTEN} -eq 0 ]]; then
        head -n 1 "${summary_csv}" | awk 'BEGIN{FS=OFS=","} {print "setting","temperature","top_k","top_p",$0}' > "${COMBINED_SUMMARY}"
        SUMMARY_HEADER_WRITTEN=1
    fi
    tail -n +2 "${summary_csv}" | awk -v setting="${setting_name}" -v temp="${temperature}" -v topk="${top_k}" -v topp="${top_p}" 'BEGIN{FS=OFS=","} {print setting,temp,topk,topp,$0}' >> "${COMBINED_SUMMARY}"
}

# One-dimensional diagnostics. These are not a full cross product.
for temperature in 0.1 0.3 0.5 0.7 1.0; do
    setting_name="temperature_${temperature//./p}"
    run_setting "${setting_name}" "${temperature}" 0 1.0
done

for top_k in 1 2 5 10; do
    run_setting "topk_${top_k}" 1.0 "${top_k}" 1.0
done

for top_p in 0.7 0.8 0.9 0.95; do
    setting_name="topp_${top_p//./p}"
    run_setting "${setting_name}" 1.0 0 "${top_p}"
done

echo "Sampling sweep complete: ${OUT_ROOT}"
