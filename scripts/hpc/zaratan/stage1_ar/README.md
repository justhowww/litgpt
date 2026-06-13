# Stage 1: AR Pretraining

## What this tests

Can the model autoregressively reconstruct a full target VCL NAL slice
(`B_t`) byte-for-byte, conditioned on metadata (`B_meta`) and previous-slice
reference bytes (`B_ref`)? This is the baseline capability every later stage
builds on, and the reference-mode ablations (`normal` / `no_ref` / `zero_ref` /
`shuffled_ref`) measure how much the reference actually contributes.

## What's changed vs the shared script

All stages run `scripts/byte/train.py`. Stage 1 simply leaves the FIM
knobs at their defaults:

- `--p-fim` **not passed** -> `0.0`, so every sample is AR (no FIM spans).
- `--reconstruction-task` defaults to `ar` (AR decoder probe only).
- Vocabulary size `259` (bytes + PAD + SLICE_BOS + SPAN_BOS).
- `STEPS=10000` by default.

## How to run

```bash
bash scripts/hpc/zaratan/stage1_ar/submit_smoke.sh
bash scripts/hpc/zaratan/stage1_ar/submit_h100.sh
```

Override any sbatch variable from the environment, e.g.
`REFERENCE_MODE=no_ref STEPS=20000 bash .../submit_h100.sh`.

The default output directory is
`scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage1/`.

## Outputs

Every `RECONSTRUCTION_EVAL_INTERVAL` steps an AR decoder probe runs on fixed
validation slices and logs `reconstruction/*` (decode rate, PSNR, SSIM) to
TensorBoard.

## Options

Current runs use **oracle length**: generation length is supplied externally,
so the model is never trained to terminate on its own.

`--use-eos` (shared script, **default off**) switches this off by appending a
learned `SEQ_EOS` terminator after each AR target and lifting the vocab to
`263`. It is not wired into these launchers; enabling it produces checkpoints
incompatible with the default `259`-wide oracle-length runs.
