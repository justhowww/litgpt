#!/bin/bash
# Shared login-node helper for staging the H.264 corpus to compute-visible storage.

stage_corpus() {
    local source_corpus=$1
    local staged_corpus=$2

    if [[ ! -r "${source_corpus}/manifest.jsonl" || ! -d "${source_corpus}/h264" ]]; then
        echo "Expected manifest.jsonl and h264/ under SOURCE_CORPUS=${source_corpus}" >&2
        return 1
    fi

    mkdir -p "${staged_corpus}"

    echo "Staging H.264 files to compute-node-visible storage..."
    rsync -a --info=progress2 "${source_corpus}/h264/" "${staged_corpus}/h264/"

    if [[ -r "${source_corpus}/corpus.json" ]]; then
        rsync -a "${source_corpus}/corpus.json" "${staged_corpus}/corpus.json"
    fi

    # Publish the manifest last so it describes the most complete staged snapshot.
    rsync -a "${source_corpus}/manifest.jsonl" "${staged_corpus}/manifest.jsonl"
}

