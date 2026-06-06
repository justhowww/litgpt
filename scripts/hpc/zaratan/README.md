# Zaratan Training Jobs

These wrappers keep Zaratan/Slurm configuration separate from the portable
`scripts/train_byte_stage1.py` launcher.

The manifest and its sibling `h264/` directory must be available from compute
nodes, preferably under project scratch rather than `SHELL`.

## Stage and submit

The source corpus currently lives under storage that compute nodes cannot see.
Run this command on a login node to copy it into `scratch.metzler-prj`, submit
training, and delete only the staged corpus after successful training:

```bash
bash scripts/hpc/zaratan/submit_stage1.sh
```

Defaults:

```text
source: /home/$USER/SHELL.metzler-prj/OpenVid-1M/h264
stage:  /home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data
output: /home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage1
```

The manifest is copied last, after the encoded files. If training fails, staged
data is retained so the job can be resumed. Disable automatic cleanup with:

```bash
CLEANUP_AFTER_SUCCESS=0 bash scripts/hpc/zaratan/submit_stage1.sh
```

Paths and training settings can be overridden at submission:

```bash
SOURCE_CORPUS=/path/in/SHELL \
STAGED_CORPUS=/path/in/scratch/data \
OUT_DIR=/path/in/scratch/runs/byte-stage1 \
BLOCK_SIZE=16384 \
STEPS=10000 \
bash scripts/hpc/zaratan/submit_stage1.sh
```

## Smoke job

Run the login-node wrapper so the corpus is copied before the GPU smoke job:

```bash
bash scripts/hpc/zaratan/submit_smoke.sh
```

The staged corpus is retained after smoke testing for the subsequent training
run. `rsync` makes repeated staging incremental. The smoke job indexes only
100 manifest clips by default; override with `MAX_MANIFEST_ROWS`.

Staging forces copied files into group `zt-metzler-prj`; otherwise `rsync -a`
can preserve the source project's group and charge the wrong project quota.
Override with `STAGED_GROUP` if needed.

## Training job

```bash
MANIFEST=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/manifest.jsonl \
OUT_DIR=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage1 \
BLOCK_SIZE=16384 \
STEPS=10000 \
sbatch scripts/hpc/zaratan/train_stage1.sbatch
```

The scripts default to one A100. Override Slurm resources at submission time,
for example:

```bash
sbatch --gpus=h100:1 --time=02:00:00 \
  scripts/hpc/zaratan/smoke_stage1.sbatch
```

Shared environment defaults are in `env.sh`. Override them without editing
tracked files:

```bash
REPO_ROOT=$PWD \
CONDA_ROOT=/home/$USER/scratch.metzler-prj/miniforge3 \
CONDA_ENV=litpt \
MANIFEST=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/manifest.jsonl \
sbatch scripts/hpc/zaratan/smoke_stage1.sbatch
```

The submission wrappers default to Slurm account `metzler-prj-cmsc`. This is
distinct from the filesystem group `zt-metzler-prj`. Override it with
`SBATCH_ACCOUNT` if the allocation changes.
