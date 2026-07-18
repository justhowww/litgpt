# Stage 0 — multi-frame AR (H0)

Trains the byte-LM from scratch on the **multi-frame stream-window** AR objective
(`ByteStreamWindowDataset`, `dataset_mode=window`, `p_fim=0`) to verify the
AVC-LM/JPEG-LM generation legacy on our setup. See `0616.md` (H0).

- **Objective:** next-byte loss across contiguous windows beginning at a GOP
  boundary (parameter sets + IDR + P-frames); previous frames are the causal
  context, offsets reset per NAL.
- **Split:** window-level random split (within-video). Held-out-video is deferred
  to scaling.
- **Model:** defaults (n_layer 8, n_embd 512, n_head 8 = 33.9 M), same as Stage 1.
- **Block size:** 16384 (reused).
- **Reconstruction probe:** off (slice/FIM-only). Continuation quality is measured
  post-hoc.

## Run

```bash
STAGED_CORPUS=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data \
bash scripts/hpc/zaratan/stage0_ar_multiframe/submit_h100.sh
```

Overridable: `STEPS`, `BLOCK_SIZE`, `WINDOW_MIN_FRAMES`, `GLOBAL_BATCH_SIZE`,
`OUT_DIR`.

## Evaluate (continuation probe, post-hoc on checkpoints)

Launcher (Zaratan, H100). `CHECKPOINT_STEPS` selects which `step-XXXXXXXX`
snapshots under the run to evaluate; the rest default to the continuation-probe
defaults (8-frame clean prefix, 4-frame continuation, 20 clips, intra mode on):

```bash
STAGED_CORPUS=/home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data \
CHECKPOINT_STEPS="2000 4000 6000" \
bash scripts/hpc/zaratan/stage0_ar_multiframe/submit_eval_h100.sh
```

Overridable: `RUN_NAME`/`RUN_DIR`, `OUT_DIR`, `NUM_CLIPS`, `NUM_VISUALIZATIONS`,
`PREFIX_FRAMES`, `CONT_FRAMES`, `TEMPERATURE`, `EVAL_INTRA` (`0` to skip intra).
Output lands in `${RUN_DIR}/continuation_eval` by default.

Raw command (no Slurm):

```bash
python scripts/byte/eval/eval_ar_continuation.py \
    /home/$USER/scratch.metzler-prj/OpenVid-1M_Data/data/manifest.jsonl \
    --checkpoint-dirs RUN/step-XXXXXXXX [...] \
    --out-dir RUN/continuation_eval \
    --prefix-frames 8 --cont-frames 4 --num-clips 20
```

Pass = decode-valid + plausible continuation (videos in `continuation_eval/frames/`),
val CE plateauing. PSNR/SSIM vs GT is secondary (generation diverges from GT).
