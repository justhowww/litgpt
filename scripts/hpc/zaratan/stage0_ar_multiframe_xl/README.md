# Stage 0 — multi-frame AR, SCALED (H0-XL)

A **single-variable scale comparison** against the 33.9 M H0 run
(`scripts/hpc/zaratan/stage0_ar_multiframe`). Same objective (window AR,
`p_fim=0`), same within-video split, same block size / batch / LR schedule —
**only the model arch grows.**

## Why this run

The H0 run converged and **plateaued** at val ppl ~120 / **6.90 bits/byte**
(monotone but flat; ~90 % of the gain by 30 B tokens; small train/val gap over
20–40 epochs). A plateau has two explanations with opposite consequences:

- **Capacity floor** — 33.9 M can't represent more structure → scale is the lever
  (H5 alive). The CE-generation premise holds.
- **Data/objective floor** — the CAVLC byte stream is near-incompressible past
  ~6.9 bits/byte → scale won't help; pivots toward the H1 / visual-supervision
  debate.

Only a bigger model on the *same* data/objective disambiguates these. See
`0616.md`.

## Model

Default **~200 M**: `n_layer 12 / n_embd 1024 / n_head 16` (head_dim 64 — same
head_dim as the 8 / 512 / 8 = 33.9 M H0 run). ~6× H0. Override via env:

- cheaper ~114 M: `N_EMBD=768 N_HEAD=12`
- deeper: `N_LAYER=24`

Everything else (block 16384, global batch 64, micro 1, LR 3e-4 → 3e-5,
within-video split) is held identical to H0 for comparability.

## Run

```bash
STAGED_CORPUS=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data \
bash scripts/hpc/zaratan/stage0_ar_multiframe_xl/submit_h100.sh
```

Overridable: `N_LAYER`, `N_EMBD`, `N_HEAD`, `STEPS`, `BLOCK_SIZE`,
`WINDOW_MIN_FRAMES`, `GLOBAL_BATCH_SIZE`, `OUT_DIR`. Out dir defaults to
`${STAGED_CORPUS}/runs/byte-stage0-ar-multiframe-xl`; TensorBoard under it.

## Reading it (pre-registered, vs the H0 curve)

Overlay `val_loss_ar` / bits-per-byte on H0. **Judge by where THIS curve
flattens, not H0's 190 B-token mark** — a bigger model converges slower in
tokens, so reading it at H0's token count would falsely call it "flat."

| XL vs H0 floor | Reading |
|---|---|
| breaks well below 6.90 bits/byte, still falling where H0 flattened | **Capacity-bound. H5 is the answer.** Keep scaling; H0 plateau was a small-model artifact. |
| lands at ~same 6.90 floor | **Data/objective floor.** Scale won't save CE. Stream near-incompressible → CE-on-bytes is the wrong surface, or need more/diverse video. |
| modestly lower, then re-plateaus | Partial — capacity helps with diminishing returns; scale-limited but real. |

## Notes / watch

- **Compute:** ~6× FLOPs/token and needs more tokens to converge; `STEPS`
  defaulted up to 300k. Auto-resumes across the 1-day job wall — expect several
  resubmissions.
- **Memory:** 200 M at 16 K context, micro batch 1, bf16 on an 80 GB H100 should
  fit; if OOM, drop to `N_EMBD=768` or lower `BLOCK_SIZE`.
- **Stability:** LR held at H0's 3e-4 for a clean single-variable comparison. If
  the larger model is unstable early, lower the peak LR (and note it — it breaks
  strict single-variable, but a diverged run tells you nothing).
- **Eval:** continuation probe is post-hoc and identical to H0 — reuse the H0
  dir's `submit_eval_h100.sh` with `RUN_DIR` pointed at this run's out dir.
