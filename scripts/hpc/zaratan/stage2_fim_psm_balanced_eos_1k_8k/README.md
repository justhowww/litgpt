# Stage 2.5: Balanced EOS, 1-8 KiB Gaps

This run replaces `10x` positive EOS weighting with:

```text
ordinary byte CE + 1.0 * balanced EOS auxiliary loss
```

The auxiliary loss averages EOS-positive and non-EOS losses separately, so
missed endpoints and premature endpoints have equal class weight. It also
changes the FIM corruption distribution to `1024-8192` actual bitstream bytes.
For slices shorter than 8 KiB, the upper bound is the largest feasible span;
slices that cannot fit 1 KiB remain AR examples rather than silently using a
smaller FIM gap.

This is a new experiment. Do not point `OUT_DIR` at the completed
`byte-stage2-fim-psm-eos-weight10` run because the objective and data
distribution are different.

Smoke test:

```bash
bash scripts/hpc/zaratan/stage2_fim_psm_balanced_eos_1k_8k/submit_smoke.sh
```

Launch the 10,000-step H100 gate:

```bash
bash scripts/hpc/zaratan/stage2_fim_psm_balanced_eos_1k_8k/submit_h100.sh
```

The default output is:

```text
runs/byte-stage2-fim-psm-balanced-eos-1k-8k/
```
