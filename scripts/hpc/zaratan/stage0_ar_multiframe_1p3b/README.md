# Stage 0 — multi-frame AR, 1.3 B (H0-1p3b)

Third rung of the H0 scaling curve: **33.9 M → 200 M (xl) → 1.3 B**. Same objective
(window AR, `p_fim=0`), same within-video split, same block/batch/LR schedule —
**only the model arch grows**, with `head_dim` held at 64 for a clean
single-variable comparison.

## Model

`n_layer 24 / n_embd 2048 / n_head 32` (head_dim 64) ≈ **1.3 B**. ~6.5× the 200 M xl.
Override via `N_LAYER`/`N_EMBD`/`N_HEAD`, but any arch change breaks strict
single-variable scaling.

## Two questions this run answers

1. **Does scale buy free-run validity?** Read the **free-run survival-length**
   metric (logged every `FREE_RUN_INTERVAL` steps as `val_freerun/*`), not val CE.
   We established val CE ↓ (33 M→xl) did **not** move validity — the binding
   quantity is the per-symbol illegal-token mass ε, and free-run survival ≈ 1/ε.
   If survival-length climbs 33 M→xl→1.3 B, you're on the AVC-LM scaling curve.
2. **Is the current corpus big enough for 1.3 B?** 1.3 B is ~6.5× the size the
   data is ~matched to (xl ≈ Chinchilla on this corpus). **Watch the train/val
   gap.** Gap opens early and val min ≈ xl's floor ⇒ **data-limited** (grow
   OpenVid before scaling further). Gap stays small and val dips below xl ⇒
   **scale still helping**. This is explicitly an overfit-probe; early-stop at the
   val minimum and read it via *val-min-vs-xl + gap*, not "did it converge."

## Memory

1.3 B at 16384 context is heavy. On an 80 GB H100, micro-batch 1 / bf16 is
expected to fit but is close (the sbatch requests 96 G host RAM). **Do a short
OOM-shakeout** (`STEPS=50 COMPILE=0`) before the real launch. If it OOMs:

- drop `BLOCK_SIZE` (e.g. `8192`) — note this changes how many frames a window
  holds and weakens comparability, so record it; or
- lower `N_EMBD`/`N_LAYER` for a smaller point on the curve.

## Run

```bash
STAGED_CORPUS=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data \
bash scripts/hpc/zaratan/stage0_ar_multiframe_1p3b/submit_h100.sh
```

Overridable: `N_LAYER`, `N_EMBD`, `N_HEAD`, `STEPS`, `BLOCK_SIZE`,
`GLOBAL_BATCH_SIZE`, `MICRO_BATCH_SIZE`, `OUT_DIR`, and the `FREE_RUN_*` knobs.
Out dir defaults to `${STAGED_CORPUS}/runs/byte-stage0-ar-multiframe-1p3b`.

## Reading it (pre-registered, vs the xl curve)

Overlay `val_loss_ar` AND `val_freerun/survival_bytes` on the xl curves.

| vs xl | Reading |
|---|---|
| val_loss_ar below xl, small gap, survival-length rising | **scale helping, data still sufficient** — continue the curve |
| gap opens fast, val min ≈ xl floor, survival flat | **data-limited at 1.3 B** — grow the corpus before more scale |
| val_loss_ar below xl but survival flat at 0 | scale lowers CE but not validity → exposure-bias / objective lever, not scale |

## Eval

Post-hoc continuation eval is identical to H0/xl — reuse
`scripts/hpc/zaratan/stage0_ar_multiframe/submit_eval_h100.sh` with `RUN_DIR`
pointed at this run's out dir. The in-loop `val_freerun/*` metric is the leading
indicator; the post-hoc eval (ffmpeg pixels + Fréchet) is the full read.
