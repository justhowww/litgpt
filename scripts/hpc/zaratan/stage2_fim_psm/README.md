# Stage 2 PSM + EOS

## What this tests

A single from-scratch run that changes **two** things at once relative to the
bridge Stage 2 baseline:

1. **PSM representation** — FIM samples use explicit Prefix-Suffix-Middle
   markers (`FIM_BEGIN` / `FIM_HOLE` / `FIM_END`) instead of the single
   `SPAN_BOS` bridge layout. This only affects the **FIM** path.
2. **Learned EOS** (`--use-eos`) — a `SEQ_EOS` terminator is appended to every
   AR target *and* every FIM span, so the model learns to stop on its own
   instead of relying on an externally supplied oracle length. This affects
   **both** AR and FIM.

Running them together is deliberate (saves a run); the two metrics axes below
keep the result interpretable even though no PSM-without-EOS baseline exists.

## What's changed vs the shared script

Same `scripts/train_byte_stage1.py` as the other stages, with:

- `--fim-format psm` (vs `bridge`)
- `--use-eos` (vocab lifts to `263` for both formats)
- `--p-fim 0.5`, `--reconstruction-task both`, `STEPS=100000` (as Stage 2)

`USE_EOS=1` is the default here. Set `USE_EOS=0` for a PSM-only baseline; the
launchers automatically drop the `-eos` tag from `MODEL_NAME`/`OUT_DIR` so the
two never share a checkpoint directory (their vocab sizes differ, 263 vs 262).

## How to run

```bash
bash scripts/hpc/zaratan/stage2_fim_psm/submit_smoke.sh
P_FIM=0.5 STEPS=100000 \
  bash scripts/hpc/zaratan/stage2_fim_psm/submit_h100.sh
```

Default output directory:
`scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage2-fim-psm-eos/`.

## Reading the result (de-confounding without a PSM-only run)

EOS quality and PSM quality answer different questions, and the reconstruction
probe logs them per task to TensorBoard:

- **EOS, AR channel** — `reconstruction/ar/stop_exact_rate`,
  `reconstruction/ar/gen_len_abs_err_mean`. PSM never touches AR, so this is a
  clean EOS canary. If AR stopping or PSNR degrades vs the Stage 1 / bridge AR
  baseline, EOS is the problem.
- **EOS, FIM channel** — `reconstruction/fim/stop_exact_rate`,
  `reconstruction/fim/gen_len_abs_err_mean`. Tells you whether the model stops
  in the right place *inside* FIM spans.
- **PSM quality** — `reconstruction/fim/psnr_mean_valid` /
  `ssim_mean_valid`.

Decision tree:

1. AR stop/PSNR degrades -> **EOS** is the culprit (PSM can't cause this).
2. AR fine, FIM `stop_exact_rate` high, FIM PSNR low -> EOS works; **PSM isn't
   helping** recovery.
3. AR fine, FIM `stop_exact_rate` low -> **EOS x FIM** stopping problem, not
   PSM — the case that would otherwise hide inside FIM byte-loss.

Note: do **not** judge EOS from the loss curves. EOS is ~1 token per
64-1400-byte span, so a broken terminator barely moves average byte
cross-entropy; the `stop_*` metrics are the real signal. The decoder probe caps
generation at the oracle length (FIM span ≤ 1 packet; AR full frame), so it
captures early stops; over-runs past the true length are not measured.
