#!/bin/bash
#SBATCH --job-name=byte-data-cleanup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${STAGED_CORPUS:?STAGED_CORPUS must be set by submit_stage1.sh}"

expected_root="/home/${USER}/scratch.metzler-prj/OpenVid-1M_Data/data"
staged_real=$(realpath -m "${STAGED_CORPUS}")
expected_real=$(realpath -m "${expected_root}")

case "${staged_real}" in
    "${expected_real}"/*) ;;
    *)
        echo "Refusing to delete path outside ${expected_real}: ${staged_real}" >&2
        exit 1
        ;;
esac

if [[ "${staged_real}" == "${expected_real}" ]]; then
    echo "Refusing to delete the data root itself: ${staged_real}" >&2
    exit 1
fi

echo "Removing staged training corpus: ${staged_real}"
rm -rf -- "${staged_real}"

