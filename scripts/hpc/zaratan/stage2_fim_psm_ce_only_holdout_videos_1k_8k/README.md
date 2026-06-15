# Byte-CE-only Stage 2, held-out-video validation

Clean test of whether byte-level cross-entropy training alone suffices for
the FIM repair task. Removes the three confounds present in all prior
measurements:

1. **CE only.** `MRT_INTERVAL=0`. No visual reward in the loss.
2. **Held-out video split.** `SPLIT_BY_VIDEO=1`, `VAL_FRACTION=0.05`. None of
   the videos used for training contribute slices to validation.
3. **Strict-mode decoding is the primary metric.** Concealment-on PSNR is
   logged as diagnostic only; experiment decisions read strict-mode numbers.

## Comparison matrix

Per checkpoint, the reconstruction probe logs all of:

| Method | Strict (concealment off) | Concealment on |
|---|---|---|
| Our model, learned-EOS | `model_learned/strict` | `model_learned/error_exploding` |
| Our model, oracle length | `model_oracle/strict` | `model_oracle/error_exploding` |
| Deleted-gap (lower bound / standard concealment baseline) | `deleted_gap/strict` (= 18.35 dB floor) | `deleted_gap/error_exploding` (= standard decoder concealment) |
| Random-byte fill | `random_bytes/strict` | `random_bytes/error_exploding` |
| Ground truth (upper bound) | `ground_truth/strict` | `ground_truth/error_exploding` |

The headline question is whether
`reconstruction/fim/model_oracle/strict/psnr_mean_valid`
sustainably exceeds `reconstruction/fim/deleted_gap/error_exploding/psnr_mean_valid`
(the standard decoder concealment baseline) across training. If yes, byte-CE is
a viable bitstream repair path. If not, the visual-supervision direction is
empirically justified.

## Gap-size buckets

The same comparison matrix repeats under three FIM gap-size buckets:

- `bucket_1k_2k/` — 1024 ≤ gap < 2048 bytes
- `bucket_2k_4k/` — 2048 ≤ gap < 4096 bytes
- `bucket_4k_8k/` — 4096 ≤ gap ≤ 8192 bytes

This separates "small-gap success" from "large-gap collapse," which a single
aggregate PSNR can hide.

## Decision criteria

| Strict PSNR (held-out video) | Reading |
|---|---|
| Sustains > deleted_gap/error_exploding across training | Byte-CE is sufficient. Ship CE-only and scale. MRT direction is unnecessary. |
| Peaks early then degrades, ending below deleted_gap/error_exploding | Misalignment is real on the clean test. MRT direction (or alternatives) is justified. |
| Never beats deleted_gap/strict (= 18.35) | Small-model byte-CE doesn't reach useful quality on held-out videos. Need scaling, not MRT. |
| Small-gap bucket works, large-gap collapses | Misalignment is gap-size-dependent. Suggests bit-weighted CE or longer-context training. |

```bash
CLEANUP=0 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_ce_only_holdout_videos_1k_8k/submit_h100.sh
```
