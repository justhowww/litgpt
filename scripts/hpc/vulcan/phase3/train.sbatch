#!/usr/bin/env bash
# UMD Vulcan allocation for phase 3's AR+FIM training body, on the vulcan-high QOS:
# 4x a6000, 16 cpus, 128G mem, 1-12:00:00 wall (that QOS's per-job caps). Sources
# phase2_fim/train.sh directly (not a copy) -- the training body (objective, dataset
# mode, FIM knobs) is identical between phase 2 and phase 3; only scale (arch, video
# count, device count) differs, and that's all set via env vars from submit.sh. A
# duplicated copy here would silently drift out of sync with phase 2's bug fixes.
# submit.sh may override --output/--error and --gres/--exclude.
#
# NOTE: qos=vulcan-high is this script's best guess at the QOS name matching the
# vulcan-high partition (by analogy with phase2_fim's vulcan-ampere/vulcan-medium
# pairing, which does NOT share a name with its partition) -- verify with
# `sacctmgr show qos` before the first real submission and correct if wrong.

#SBATCH --job-name=byte-phase3
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-high
#SBATCH --account=vulcan-metzler
#SBATCH --time=1-12:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-"${SLURM_SUBMIT_DIR}"}
export REPO_ROOT
source "${REPO_ROOT}/scripts/hpc/vulcan/phase2_fim/train.sh"
