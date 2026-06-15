# Byte-CE-only Stage 2, within-video validation

This experiment asks whether the small model can learn useful canonical H.264
completion before requiring held-out-video generalization.

It matches `stage2_fim_psm_ce_only_holdout_videos_1k_8k` except for the split:

- `SPLIT_BY_VIDEO=0`: validation uses unseen target-slice samples, but source
  videos can also contribute training slices.
- `VAL_FRACTION=0.05`: 5% of slice samples are selected deterministically for
  validation.
- CE only, PSM FIM, 1024-8192-byte gaps, strict reconstruction, and all
  baselines remain unchanged.
- Three fixed validation samples are rendered in TensorBoard at each probe:
  ground truth, deleted-gap strict, deleted-gap with FFmpeg's default
  concealment, and model reconstruction under strict decoding.

Interpret the paired experiments as follows:

| Within-video | Held-out-video | Interpretation |
|---|---|---|
| Fails | Fails | The current formulation or optimization is not learning the task. |
| Works | Fails | The task is learnable, but generalization likely needs more scale or diversity. |
| Works | Works | Strong evidence that byte CE is sufficient for this stage. |

Run on H100:

```bash
CLEANUP=0 \
STEPS=100000 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_ce_only_within_video_1k_8k/submit_h100.sh
```
