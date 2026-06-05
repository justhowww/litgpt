# Zaratan Training Jobs

These wrappers keep Zaratan/Slurm configuration separate from the portable
`scripts/train_byte_stage1.py` launcher.

The manifest and its sibling `h264/` directory must be available from compute
nodes, preferably under project scratch rather than `SHELL`.

## Smoke job

```bash
MANIFEST=/scratch/zt1/project/metzler-prj/user/$USER/data/OpenVid-1M/manifest.jsonl \
sbatch --account=metzler-prj \
  scripts/hpc/zaratan/smoke_stage1.sbatch
```

## Training job

```bash
MANIFEST=/scratch/zt1/project/metzler-prj/user/$USER/data/OpenVid-1M/manifest.jsonl \
OUT_DIR=/scratch/zt1/project/metzler-prj/user/$USER/runs/byte-stage1 \
BLOCK_SIZE=8192 \
STEPS=10000 \
sbatch --account=metzler-prj \
  scripts/hpc/zaratan/train_stage1.sbatch
```

The scripts default to one A100. Override Slurm resources at submission time,
for example:

```bash
sbatch --account=metzler-prj --gpus=h100:1 --time=02:00:00 \
  scripts/hpc/zaratan/smoke_stage1.sbatch
```

Shared environment defaults are in `env.sh`. Override them without editing
tracked files:

```bash
CONDA_ROOT=/scratch/path/to/miniforge3 \
CONDA_ENV=litpt \
MANIFEST=/scratch/path/to/manifest.jsonl \
sbatch --account=metzler-prj scripts/hpc/zaratan/smoke_stage1.sbatch
```
