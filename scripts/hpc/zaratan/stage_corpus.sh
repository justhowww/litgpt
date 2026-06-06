#!/bin/bash
# Shared login-node helper for staging the H.264 corpus to compute-visible storage.

stage_corpus() {
    local source_corpus=$1
    local staged_corpus=$2
    local staged_group=${STAGED_GROUP:-zt-metzler-prj}

    if [[ ! -r "${source_corpus}/manifest.jsonl" || ! -d "${source_corpus}/h264" ]]; then
        echo "Expected manifest.jsonl and h264/ under SOURCE_CORPUS=${source_corpus}" >&2
        return 1
    fi

    mkdir -p "${staged_corpus}"
    chgrp "${staged_group}" "${staged_corpus}"
    chmod g+s "${staged_corpus}"

    # Repair a partial prior copy. Every directory must be setgid so temporary
    # rsync files inherit the destination project group at creation time.
    if [[ -d "${staged_corpus}/h264" ]]; then
        chgrp -R "${staged_group}" "${staged_corpus}/h264"
        find "${staged_corpus}/h264" -type d -exec chmod g+s {} +
    fi

    echo "Staging H.264 files to compute-node-visible storage..."
    # Run rsync with the destination project as its effective group. --chown
    # alone is too late because rsync creates temporary files before chowning.
    local source_h264_q staged_h264_q
    printf -v source_h264_q "%q" "${source_corpus}/h264/"
    printf -v staged_h264_q "%q" "${staged_corpus}/h264/"
    sg "${staged_group}" -c \
        "rsync -a --no-group --chmod=Dg+s --info=progress2 ${source_h264_q} ${staged_h264_q}"

    if [[ -r "${source_corpus}/corpus.json" ]]; then
        local source_corpus_json_q staged_corpus_json_q
        printf -v source_corpus_json_q "%q" "${source_corpus}/corpus.json"
        printf -v staged_corpus_json_q "%q" "${staged_corpus}/corpus.json"
        sg "${staged_group}" -c \
            "rsync -a --no-group ${source_corpus_json_q} ${staged_corpus_json_q}"
    fi

    # Publish the manifest last so it describes the most complete staged snapshot.
    local source_manifest_q staged_manifest_q
    printf -v source_manifest_q "%q" "${source_corpus}/manifest.jsonl"
    printf -v staged_manifest_q "%q" "${staged_corpus}/manifest.jsonl"
    sg "${staged_group}" -c \
        "rsync -a --no-group ${source_manifest_q} ${staged_manifest_q}"
}

# Allow both:
#   source stage_corpus.sh; stage_corpus SOURCE DEST
#   bash stage_corpus.sh SOURCE DEST
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    if [[ $# -ne 2 ]]; then
        echo "Usage: bash $0 SOURCE_CORPUS STAGED_CORPUS" >&2
        exit 2
    fi
    stage_corpus "$1" "$2"
fi
