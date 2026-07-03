#!/bin/bash
# Stage source directories, encode them on CPU nodes, and retain both copies.
#
# Run this script on a Zaratan login node, preferably inside tmux. Compute nodes
# cannot access SHELL.metzler-prj, so this login process performs both rsync
# operations and waits for each Slurm job.
#
# Per-directory workflow:
#
#   SHELL source videos
#          |
#          | 1. rsync on login node
#          v
#   temporary scratch source
#          |
#          | 2. sbatch CPU encoding job
#          v
#   persistent scratch H.264 corpus  <-- training reads this copy
#          |
#          | 3. rsync on login node
#          v
#   SHELL H.264 backup
#          |
#          | 4. merge manifests, then remove temporary scratch source
#
# Safety invariant: temporary source videos are deleted only after Slurm
# succeeds, the output manifest exists, and H.264 copy-back succeeds.

set -euo pipefail

# Resolve repository-relative helper scripts regardless of the caller's cwd.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-"$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"}

# Persistent input and backup storage, visible only from the login node.
SOURCE_VIDEO_ROOT=${SOURCE_VIDEO_ROOT:-"/home/${USER}/SHELL.metzler-prj/OpenVid-1M/video"}
SHELL_CORPUS=${SHELL_CORPUS:-"/home/${USER}/SHELL.metzler-prj/OpenVid-1M/h264"}

# Scratch layout:
# - encode-staging/: temporary source videos, deleted after successful copy-back
# - data/h264/: persistent encoded corpus retained for model training
SCRATCH_DATA_ROOT=${SCRATCH_DATA_ROOT:-"/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data"}
SCRATCH_STAGE_ROOT=${SCRATCH_STAGE_ROOT:-"${SCRATCH_DATA_ROOT}/encode-staging"}
SCRATCH_CORPUS=${SCRATCH_CORPUS:-"${SCRATCH_DATA_ROOT}/data"}

# Encoder and scheduler configuration. Every part uses the same config so the
# resulting directory shards belong to one reproducible corpus.
CONFIG=${CONFIG:-"${REPO_ROOT}/preprocessing/h264_preprocess_config.json"}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"metzler-prj-cmsc"}
SBATCH_PARTITION=${SBATCH_PARTITION:-""}

# One directory at a time bounds temporary source storage. Increase only when
# scratch capacity can hold several staged source directories simultaneously.
MAX_ACTIVE_DIRS=${MAX_ACTIVE_DIRS:-1}

# Back up encoded H.264 + manifest to SHELL after each part. Set BACKUP_TO_SHELL=0
# to skip the SHELL copy-back entirely (scratch corpus is still fully populated;
# training reads scratch, not SHELL).
BACKUP_TO_SHELL=${BACKUP_TO_SHELL:-1}

# Fail before transferring data if required roots or settings are invalid.
if [[ ! -d "${SOURCE_VIDEO_ROOT}" ]]; then
    echo "Source video root is missing: ${SOURCE_VIDEO_ROOT}" >&2
    exit 1
fi
if [[ ! -r "${CONFIG}" ]]; then
    echo "Preprocessing config is not readable: ${CONFIG}" >&2
    exit 1
fi
if (( MAX_ACTIVE_DIRS < 1 )); then
    echo "MAX_ACTIVE_DIRS must be at least 1" >&2
    exit 1
fi

mkdir -p \
    "${SCRATCH_STAGE_ROOT}" \
    "${SCRATCH_CORPUS}/h264"
if [[ "${BACKUP_TO_SHELL}" == "1" ]]; then
    mkdir -p \
        "${SHELL_CORPUS}/h264" \
        "${SHELL_CORPUS}/manifest-parts"
fi

# Each Slurm job writes an independent manifest shard. Serialize corpus-level
# merges so concurrent directory jobs cannot overwrite one another.
merge_manifest() {
    local corpus_root=$1
    local shard=$2
    flock "${corpus_root}/.manifest.lock" \
        python "${REPO_ROOT}/preprocessing/merge_h264_manifests.py" \
            "${corpus_root}/manifest.jsonl" \
            "${shard}"
}

process_directory() {
    local source_dir=$1
    local part_name
    local work_root
    local staged_source_root
    local scratch_part_manifest
    local shell_part_manifest
    local job_id

    part_name=$(basename "${source_dir}")
    work_root="${SCRATCH_STAGE_ROOT}/${part_name}"
    staged_source_root="${work_root}/source"
    scratch_part_manifest="${SCRATCH_CORPUS}/.manifest-${part_name}.jsonl"
    shell_part_manifest="${SHELL_CORPUS}/manifest-parts/${part_name}.jsonl"

    # Phase 1: stage one source directory onto compute-visible scratch.
    # --partial preserves an interrupted transfer for the next retry.
    echo "[stage] ${source_dir} -> ${staged_source_root}/${part_name}"
    mkdir -p "${staged_source_root}/${part_name}"
    rsync -rltp --no-g --no-o --partial --info=progress2 \
        "${source_dir}/" \
        "${staged_source_root}/${part_name}/"

    # Phase 2: submit the scratch-local encoder and wait for it to finish.
    # --wait is intentional: copy-back and cleanup must not run early.
    # The sbatch script cannot access SHELL storage.
    sbatch_args=(
        --parsable
        --wait
        --account="${SBATCH_ACCOUNT}"
        --export="ALL,REPO_ROOT=${REPO_ROOT},STAGED_SOURCE_ROOT=${staged_source_root},SCRATCH_CORPUS=${SCRATCH_CORPUS},PART_NAME=${part_name},CONFIG=${CONFIG}"
    )
    if [[ -n "${SBATCH_PARTITION}" ]]; then
        sbatch_args+=(--partition="${SBATCH_PARTITION}")
    fi

    echo "[submit] ${part_name}"
    job_id=$(sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/encode_h264_dir.sbatch")
    echo "[encoded] ${part_name} (job ${job_id})"

    # Slurm success alone is insufficient. Require a non-empty per-directory
    # manifest before treating the encoded outputs as complete.
    if [[ ! -s "${scratch_part_manifest}" ]]; then
        echo "Missing manifest after successful job: ${scratch_part_manifest}" >&2
        return 1
    fi

    # Publish the shard into the scratch corpus manifest that training reads.
    merge_manifest "${SCRATCH_CORPUS}" "${scratch_part_manifest}"

    # Phase 3: optionally back up completed H.264 files to SHELL. A failed rsync
    # exits this function before cleanup, leaving staged source and scratch H.264
    # intact. Skipped entirely when BACKUP_TO_SHELL=0.
    if [[ "${BACKUP_TO_SHELL}" == "1" ]]; then
        echo "[copy-back] ${part_name}"
        mkdir -p "${SHELL_CORPUS}/h264/${part_name}"
        rsync -rltp --no-g --no-o --partial --info=progress2 \
            "${SCRATCH_CORPUS}/h264/${part_name}/" \
            "${SHELL_CORPUS}/h264/${part_name}/"
        cp "${scratch_part_manifest}" "${shell_part_manifest}"
        # Mirror the shard into the SHELL backup manifest.
        merge_manifest "${SHELL_CORPUS}" "${shell_part_manifest}"
        cp "${SCRATCH_CORPUS}/corpus.json" "${SHELL_CORPUS}/corpus.json"
    fi

    # Phase 4: delete only temporary source data. realpath guards against an
    # empty or malformed variable turning rm into an out-of-tree deletion.
    # The encoded H.264 directory under SCRATCH_CORPUS is intentionally kept.
    case "$(realpath -m "${work_root}")" in
        "$(realpath -m "${SCRATCH_STAGE_ROOT}")"/*)
            rm -rf "${work_root}"
            ;;
        *)
            echo "Refusing to delete path outside SCRATCH_STAGE_ROOT: ${work_root}" >&2
            return 1
            ;;
    esac

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

# Run directory workflows in the background while bounding how many complete
# source directories can occupy scratch at once. A failed workflow does not
# abort the others; failures are summarized after all active jobs finish.
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
    echo "${failures} directory job(s) failed; staged inputs were retained for retry." >&2
    exit 1
fi

echo "All requested directories encoded successfully."
