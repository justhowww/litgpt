# Stage 2.5: Online MRT, 1-8 KiB Gaps

This run initializes model weights from the 1K-step supervised balanced-EOS
checkpoint. It retains supervised byte CE and balanced EOS loss on every
optimizer step, then adds one online decoder-scored MRT context every eighth
step.

The default candidate set contains the ground-truth span and 15 stochastic
current-policy samples. Candidates are decoded with FFmpeg concealment disabled
and strict error detection. Valid candidates receive `1000 * RGB_MSE`, capped
at `2.0`; strict decode failures receive risk `2.0`. The MRT gradient has weight
`4.0` and is not divided by the supervised gradient-accumulation factor.

The initial online gate limits MRT contexts to gaps of at most 2048 bytes to
control generation and decode cost. Supervised training still uses the full
1024-8192 byte gap distribution.

Launch:

```bash
bash scripts/hpc/zaratan/stage2_fim_psm_mrt_1k_8k/submit_h100.sh
```

Override `INITIAL_CHECKPOINT_DIR` when the 1K checkpoint is stored elsewhere.
Run `submit_smoke.sh` first to exercise one MRT update with two candidates.
