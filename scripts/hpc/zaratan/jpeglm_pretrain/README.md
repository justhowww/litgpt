# JPEG-LM full-corpus pretraining on Zaratan

This launcher trains the planned mixed AR/FIM MEGABYTE model on the JPEG-LM
corpus: 32 layers, width 4096, 64 heads (~7B global parameters), patch size 8,
16,384 global positions (~131K raw-byte capacity), full-sequence PSM FIM,
learned EOS, one IDR-anchored GOP per sample, and a held-out-video validation
split. With the JPEG-LM encoder this is normally 16 frames; a final partial GOP
may be shorter.

The job uses 4 H100s with FSDP and Transformer-block activation checkpointing.
Each full training-state checkpoint is approximately 65 GB. The full launcher
therefore keeps one rolling `latest` checkpoint updated every 1,000 optimizer
steps and saves permanent `step-*` milestones only every 100,000 steps. When
both intervals coincide, `latest` points to the milestone instead of duplicating
it. Automatic resume prefers `latest`. Override these independently with
`LATEST_SAVE_INTERVAL` and `SAVE_INTERVAL`.

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
eligibility because it is only a systems test. It saves only the final
checkpoint, avoiding a duplicate 65 GB step checkpoint. If it passes, repeat
once with `COMPILE=1`.

After the compiled baseline passes, Speed Pilot 1 measures whether activation
recomputation can be removed. It changes only
`ACTIVATION_CHECKPOINTING=0`; the model, 256-video subset, global/micro batch,
seed, compilation, and 20-step budget remain matched. Because this is a
disposable speed/OOM test, it saves no 65 GB training-state checkpoint.

```bash
cd /nfshomes/huangyh/litgpt

STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
bash scripts/hpc/zaratan/jpeglm_pretrain/submit_speed_pilot_no_activation_checkpointing.sh
```

Compare steady-state `iter time` and peak memory with the compiled baseline. A
pass on this small subset is only the first gate; disabling activation
checkpointing for the full run still requires a near-maximum-length GOP stress
test.

Speed Pilot 2 keeps activation checkpointing enabled and changes only the
per-GPU microbatch from 1 to 2. The global batch remains 64, so gradient
accumulation falls from 16 to 8. It also skips all 65 GB checkpoint writes.

```bash
cd /nfshomes/huangyh/litgpt

STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
bash scripts/hpc/zaratan/jpeglm_pretrain/submit_speed_pilot_microbatch2.sh
```

If microbatch 2 passes, Speed Pilot 3 tests microbatch 4 under the same setup.
The global batch remains 64 and gradient accumulation falls to 4.

```bash
cd /nfshomes/huangyh/litgpt

STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
bash scripts/hpc/zaratan/jpeglm_pretrain/submit_speed_pilot_microbatch4.sh
```

Compare optimizer-step time, not only the printed micro-iteration time: multiply
the baseline micro-iteration median by 16 and the Pilot 2 median by 8. Because
two variable-length GOPs share a batch, also check whether padding and peak
memory erase the expected utilization gain.

The full launcher requires at least 900,000 manifest rows and refuses to submit
if fewer than 50% of non-IDR slices can host the configured FIM hole. JPEG-LM
uses no protected prefix by default (`SLICE_HEADER_GUARD_BYTES=0`), so FIM may
reconstruct the frame's start code and headers as well as its payload. With a
64-byte minimum hole, a frame needs at least 65 bytes to be eligible.

```bash
cd /nfshomes/huangyh/litgpt

STAGED_CORPUS=/home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm \
FIM_MIN_GAP=64 \
FIM_MAX_GAP=1400 \
SLICE_HEADER_GUARD_BYTES=0 \
WINDOW_UNIT=gop \
bash scripts/hpc/zaratan/jpeglm_pretrain/submit.sh
```

This intentionally makes header-crossing and whole-frame-prefix holes part of
the training distribution. The deployment corruption placement remains an
evaluation choice and need not use the same guard.

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
