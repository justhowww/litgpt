# Zaratan Training Jobs

These wrappers keep Zaratan/Slurm configuration separate from the portable
`scripts/train_byte_stage1.py` launcher.

The manifest and its sibling `h264/` directory must be available from compute
nodes, preferably under project scratch rather than `SHELL`.

## Build the NAL index

Build the persistent NAL-offset cache once on a CPU node before submitting
training:

```bash
bash scripts/hpc/zaratan/submit_byte_nal_index.sh
```

The default output is
`/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/nal_index.sqlite`.
The build commits every 100 files, so rerunning the same command resumes after
an interruption. Set `REBUILD_INDEX=1` only when a full rebuild is required.

Training validates the index against the manifest and exits before allocating
model memory if the cache is missing, incomplete, or stale. A limited
`--max-manifest-rows` debug run reads only those files' rows from SQLite.

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
NAL_INDEX=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/nal_index.sqlite \
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

## Conditioning viability experiments

Evaluate an existing checkpoint under correct, removed, zeroed, and shuffled
reference conditioning. The same job also fits and evaluates a train-target
byte-unigram baseline:

```bash
MANIFEST=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/manifest.jsonl \
CHECKPOINT_DIR=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage1/final \
sbatch scripts/hpc/zaratan/evaluate_conditioning.sbatch
```

The report is written to `CHECKPOINT_DIR/conditioning_eval.json`.

Submit matched 1,000-step training runs for correct, removed, and shuffled
references, followed by automatic evaluation of each final checkpoint:

```bash
STEPS=1000 bash scripts/hpc/zaratan/submit_conditioning_ablations.sh
```

Outputs live under
`scratch.metzler-prj/OpenVid-1M_Data/data/runs/conditioning-ablations/`.
After all evaluations finish, `summary.json` contains the checkpoint-by-condition
loss matrix and verifies that every run used the same validation targets.

## Reconstruction validation

The full Stage 1 job runs a sparse decoder-level probe every 1,000 optimizer
steps. By default it greedily reconstructs five fixed held-out slices of at
most 2,048 bytes, reinserts each generated NAL into the original Annex-B
stream, and logs decode rate, PSNR, and SSIM through the configured logger.
The exact sample IDs are saved to `OUT_DIR/reconstruction_samples.json`.

Override the probe cost at submission time:

```bash
RECONSTRUCTION_EVAL_INTERVAL=500 \
RECONSTRUCTION_EVAL_SAMPLES=10 \
RECONSTRUCTION_MAX_TARGET_BYTES=4096 \
bash scripts/hpc/zaratan/submit_stage1.sh
```

Set `RECONSTRUCTION_EVAL_INTERVAL=0` to disable the probe. Decode failures,
timeouts, invalid generated tokens, and unexpected probe errors are logged and
do not terminate training.
