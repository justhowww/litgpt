# JPEG-LM full-corpus pretraining on Zaratan

This launcher trains the planned mixed AR/FIM MEGABYTE model on the JPEG-LM
corpus: 32 layers, width 4096, 64 heads (~7B global parameters), patch size 8,
16,384 global positions (~131K raw-byte capacity), full-sequence PSM FIM,
learned EOS, and a held-out-video validation split.

The job uses 4 H100s with FSDP and Transformer-block activation checkpointing.
Run the exact-shape pilot before the full job:

```bash
cd /nfshomes/huangyh/litgpt

STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
bash scripts/hpc/zaratan/jpeglm_pretrain/submit_pilot.sh
```

To queue the pilot behind a corpus/index job and run it only if that job
succeeds:

```bash
STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
AFTER_JOBID=INDEX_JOB_ID \
bash scripts/hpc/zaratan/jpeglm_pretrain/submit_pilot.sh
```

The preflight is deferred to the compute job in this case, after the dependency
has produced the final manifest and index.

The pilot intentionally disables `torch.compile` so an OOM can be attributed to
the model rather than compilation. It warns rather than stops on low FIM
eligibility because it is only a systems test. If it passes, repeat once with
`COMPILE=1`.

The full launcher requires at least 900,000 manifest rows and refuses to submit
if fewer than 50% of non-IDR slices can host the configured FIM hole. The latter
guard matters because JPEG-LM P frames can be much smaller than the previous
BSCV-style slices. Inspect the report and set `FIM_MIN_GAP` and
`SLICE_HEADER_GUARD_BYTES` deliberately; do not override the check merely to get
the job into the queue.

```bash
cd /nfshomes/huangyh/litgpt

STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
FIM_MIN_GAP=64 \
FIM_MAX_GAP=1400 \
SLICE_HEADER_GUARD_BYTES=64 \
bash scripts/hpc/zaratan/jpeglm_pretrain/submit.sh
```

The exact settings above are the old experimental defaults, not yet validated
for JPEG-LM. The preflight is expected to reject them if they select mainly IDR
frames. Change the values only after inspecting the finished corpus statistics.

To resume immediately after a wall-time stop:

```bash
STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
bash scripts/hpc/zaratan/jpeglm_pretrain/resubmit.sh
```

To queue the next allocation before the current one finishes:

```bash
STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
bash scripts/hpc/zaratan/jpeglm_pretrain/resubmit.sh CURRENT_JOB_ID
```
