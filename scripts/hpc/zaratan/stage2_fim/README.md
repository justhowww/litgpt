# Stage 2: Mixed AR/FIM Pretraining

## What this tests

Can a single model learn fill-in-the-middle (FIM) recovery — regenerating a
missing span `B_miss` from the surrounding prefix/suffix bytes — while keeping
its AR ability? This is the packet-loss recovery case: a slice arrives with a
hole and the model must bridge it. The 50/50 mixture checks that adding FIM
does not degrade AR, and the PSM ablation asks which FIM input representation
the model learns from more easily.

## What's changed vs the shared script

Runs the same `scripts/train_byte_stage1.py` as Stage 1, but enables the FIM
path:

- `--p-fim 0.5` — each sample is FIM with prob 0.5, otherwise AR.
- `--fim-format bridge` (default) — single `SPAN_BOS` marker layout.
- `--fim-min-gap 64`, `--fim-max-gap 1400` — missing-span length range
  (~FU-A payload scale).
- `--slice-header-guard-bytes 64` — keep holes away from the slice header so
  early experiments focus on payload recovery.
- `--reconstruction-task both` — run AR and FIM decoder probes.
- `STEPS=100000` by default; trained from scratch (does not resume Stage 1).

A separate output directory keeps it from accidentally resuming Stage 1
checkpoints.

## How to run

```bash
bash scripts/hpc/zaratan/stage2_fim/submit_smoke.sh
P_FIM=0.5 STEPS=100000 \
  bash scripts/hpc/zaratan/stage2_fim/submit_h100.sh
```

The default output directory is
`scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage2-fim/`.

## Outputs

Every 1,000 steps separate AR and FIM decoder probes run on fixed validation
samples. TensorBoard records them under `reconstruction/ar/*` and
`reconstruction/fim/*`.

## PSM representation ablation

The primary run uses the original `bridge` layout. The explicit
Prefix-Suffix-Middle representation (`FIM_BEGIN` / `FIM_HOLE` / `FIM_END`
markers) is a separate from-scratch experiment testing whether the more
structured input is easier to learn:

```bash
bash scripts/hpc/zaratan/stage2_fim/submit_psm_smoke.sh
P_FIM=0.5 STEPS=100000 \
  bash scripts/hpc/zaratan/stage2_fim/submit_psm_h100.sh
```

PSM writes to `runs/byte-stage2-fim-psm/` and uses vocabulary size `262`. The
bridge run stays at vocabulary size `259` so its existing checkpoints continue
to resume unchanged.

## Options

Current runs use **oracle length**: the AR target / FIM span length is supplied
externally, so the model is never trained to terminate on its own.

`--use-eos` (shared script, **default off**) switches this off by appending a
learned `SEQ_EOS` terminator after each AR target and FIM span, so the model
can stop on its own instead of relying on an oracle length; it lifts the vocab
to `263` for both formats. It is not wired into these launchers, and enabling it
produces checkpoints incompatible with the default-vocab oracle-length runs.
