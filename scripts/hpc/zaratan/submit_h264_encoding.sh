#!/bin/bash
# Encode OpenVid-1M source videos to H.264 on CPU nodes. Both source videos
# and the output corpus live on project scratch, so no staging or copy-back
# rsync is needed -- compute nodes read and write scratch directly.
#
# Run this script on a Zaratan login node, preferably inside tmux; it submits
# one sbatch job per source directory and waits for each to finish.

set -euo pipefail

# Resolve repository-relative helper scripts regardless of the caller's cwd.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-"$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"}

# Scratch layout: source videos in, encoded H.264 corpus out.
SOURCE_VIDEO_ROOT=${SOURCE_VIDEO_ROOT:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/video"}
SCRATCH_CORPUS=${SCRATCH_CORPUS:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data-bscv"}

# Encoder and scheduler configuration. Every part uses the same config so the
# resulting directory shards belong to one reproducible corpus.
CONFIG=${CONFIG:-"${REPO_ROOT}/preprocessing/h264_preprocess_config.json"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
SBATCH_PARTITION=${SBATCH_PARTITION:-""}

# Number of source directories to encode concurrently.
MAX_ACTIVE_DIRS=${MAX_ACTIVE_DIRS:-1}

# Fail before submitting jobs if required roots or settings are invalid.
if [[ ! -d "${SOURCE_VIDEO_ROOT}" ]]; then
    echo "Source video root is missing: ${SOURCE_VIDEO_ROOT}" >&2
    exit 1
fi
if [[ ! -r "${CONFIG}" ]]; then
    echo "Preprocessing config is not readable: ${CONFIG}" >&2
    exit 1
fi

# Match process-level parallelism to the threads requested by each x264
# process. One-thread configs retain 16 concurrent encoders. The BSCV config
# requests 22 threads, so it gets 22 CPUs and one encoder per job.
FFMPEG_THREADS=$(python -c \
    'import json, sys; value = json.load(open(sys.argv[1]))["ffmpeg"].get("threads"); print(0 if value is None else int(value))' \
    "${CONFIG}")
if (( FFMPEG_THREADS < 0 )); then
    echo "ffmpeg.threads must be null or a positive integer in ${CONFIG}" >&2
    exit 1
fi

if (( FFMPEG_THREADS == 0 )); then
    # x264 auto-threading consumes the allocation by itself.
    ENCODE_WORKERS=${ENCODE_WORKERS:-1}
    ENCODE_CPUS_PER_TASK=${ENCODE_CPUS_PER_TASK:-16}
else
    if [[ -n "${ENCODE_WORKERS:-}" ]]; then
        ENCODE_CPUS_PER_TASK=${ENCODE_CPUS_PER_TASK:-$((ENCODE_WORKERS * FFMPEG_THREADS))}
    else
        ENCODE_CPUS_PER_TASK=${ENCODE_CPUS_PER_TASK:-16}
        if (( ENCODE_CPUS_PER_TASK < FFMPEG_THREADS )); then
            ENCODE_CPUS_PER_TASK=${FFMPEG_THREADS}
        fi
        ENCODE_WORKERS=$((ENCODE_CPUS_PER_TASK / FFMPEG_THREADS))
    fi
fi

if (( ENCODE_WORKERS < 1 || ENCODE_CPUS_PER_TASK < 1 )); then
    echo "ENCODE_WORKERS and ENCODE_CPUS_PER_TASK must be positive" >&2
    exit 1
fi
if (( FFMPEG_THREADS > 0 && ENCODE_WORKERS * FFMPEG_THREADS > ENCODE_CPUS_PER_TASK )); then
    echo "Refusing CPU oversubscription: ${ENCODE_WORKERS} encoders x ${FFMPEG_THREADS} threads > ${ENCODE_CPUS_PER_TASK} allocated CPUs" >&2
    exit 1
fi
if (( MAX_ACTIVE_DIRS < 1 )); then
    echo "MAX_ACTIVE_DIRS must be at least 1" >&2
    exit 1
fi

mkdir -p "${SCRATCH_CORPUS}/h264"

# Each Slurm job writes an independent manifest shard. Serialize corpus-level
# merges so concurrent directory jobs cannot overwrite one another.
merge_manifest() {
    local shard=$1
    flock "${SCRATCH_CORPUS}/.manifest.lock" \
        python "${REPO_ROOT}/preprocessing/merge_h264_manifests.py" \
            "${SCRATCH_CORPUS}/manifest.jsonl" \
            "${shard}"
}

process_directory() {
    local source_dir=$1
    local part_name
    local part_manifest
    local job_id

    part_name=$(basename "${source_dir}")
    part_manifest="${SCRATCH_CORPUS}/.manifest-${part_name}.jsonl"

    # Submit the encoder job against the source directory in place and wait
    # for it to finish. --wait is intentional: the manifest check and merge
    # below must not run early.
    sbatch_args=(
        --parsable
        --wait
        --account="${SBATCH_ACCOUNT}"
        --cpus-per-task="${ENCODE_CPUS_PER_TASK}"
        --export="ALL,REPO_ROOT=${REPO_ROOT},STAGED_SOURCE_ROOT=${SOURCE_VIDEO_ROOT},SCRATCH_CORPUS=${SCRATCH_CORPUS},PART_NAME=${part_name},CONFIG=${CONFIG},ENCODE_WORKERS=${ENCODE_WORKERS}"
    )
    if [[ -n "${SBATCH_PARTITION}" ]]; then
        sbatch_args+=(--partition="${SBATCH_PARTITION}")
    fi

    echo "[submit] ${part_name}: cpus=${ENCODE_CPUS_PER_TASK} workers=${ENCODE_WORKERS} ffmpeg_threads=${FFMPEG_THREADS:-auto}"
    job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/encode_h264_dir.sbatch")
    echo "[encoded] ${part_name} (job ${job_id})"

    # Slurm success alone is insufficient. Require a non-empty per-directory
    # manifest before treating the encoded outputs as complete.
    if [[ ! -s "${part_manifest}" ]]; then
        echo "Missing manifest after successful job: ${part_manifest}" >&2
        return 1
    fi

    # Publish the shard into the corpus manifest that training reads.
    merge_manifest "${part_manifest}"

    echo "[done] ${part_name}; H.264 retained at ${SCRATCH_CORPUS}/h264/${part_name}"
}

# Command-line arguments may be either absolute source directories or names
# such as "part1" resolved under SOURCE_VIDEO_ROOT. With no arguments, process
# every immediate source directory in deterministic sorted order.
source_dirs=()
if (( $# > 0 )); then
    for name in "$@"; do
        if [[ -d "${name}" ]]; then
            source_dirs+=("$(realpath "${name}")")
        elif [[ -d "${SOURCE_VIDEO_ROOT}/${name}" ]]; then
            source_dirs+=("$(realpath "${SOURCE_VIDEO_ROOT}/${name}")")
        else
            echo "Source directory not found: ${name}" >&2
            exit 1
        fi
    done
else
    while IFS= read -r source_dir; do
        source_dirs+=("${source_dir}")
    done < <(find "${SOURCE_VIDEO_ROOT}" -mindepth 1 -maxdepth 1 -type d -print | sort)
fi

if (( ${#source_dirs[@]} == 0 )); then
    echo "No source directories found under ${SOURCE_VIDEO_ROOT}" >&2
    exit 1
fi

# Run directory workflows in the background while bounding how many jobs are
# active at once. A failed workflow does not abort the others; failures are
# summarized after all active jobs finish.
failures=0
for source_dir in "${source_dirs[@]}"; do
    process_directory "${source_dir}" &

    while (( $(jobs -pr | wc -l) >= MAX_ACTIVE_DIRS )); do
        if ! wait -n; then
            failures=$((failures + 1))
        fi
    done
done

while (( $(jobs -pr | wc -l) > 0 )); do
    if ! wait -n; then
        failures=$((failures + 1))
    fi
done

if (( failures > 0 )); then
    echo "${failures} directory job(s) failed." >&2
    exit 1
fi

echo "All requested directories encoded successfully."
