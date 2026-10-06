# JPEG-LM overfit check

The question: on the current JPEG-LM pipeline, can the simplest possible model memorize a
handful of videos well enough that **greedy, unmasked free-run generation reproduces the
ground-truth bytes exactly**? If not, nothing downstream (scaling, MEGABYTE, syntax masks)
is interpretable.

Setup (fixed by `submit.py`, not configurable): plain Qwen3 byte transformer
(`byte_patch_size: 1`, no MEGABYTE), one GOP per window, 16 KB context, PSM FIM with
full-sequence loss, plain next-byte CE (no span/EOS weighting), one A100.

| Config | Objective | Free-run check (train split) |
|---|---|---|
| `o0_ar.yaml` | pure AR, `p_fim: 0` | prompt = BOS + the GOP's first frame; generate the rest + EOS |
| `o1_fim_k1.yaml` | pure FIM, one fixed hole per window | replay each trained hole (hash-verified); generate the middle + EOS |
| `o2_fim_changing.yaml` | pure FIM, hole redrawn every access | 4 new deterministic holes per training window |

Every FIM run also gets `train_newholes`: 4 deterministic holes per training window, disjoint
from any cached training hole (known content, unseen cut). Rerunning O1 with
`OVERFIT_EVAL_ONLY=1` adds it without redoing the existing evaluations, which gives a
fixed-hole vs changing-hole comparison on nearly the same cuts.

Both train on the first 9 manifest videos (8 train, 1 held out as a reference).

Pass: teacher-forced accuracy ≈ 100% and `byte_exact_rate` ≥ 0.95 on the train split.
Failures report where the first wrong byte falls (`first_divergence_kind`,
`first_divergence_syntax`).

```bash
python scripts/hpc/zaratan/jpeglm_overfit/submit.py scripts/hpc/zaratan/jpeglm_overfit/o0_ar.yaml --dry-run
python scripts/hpc/zaratan/jpeglm_overfit/submit.py scripts/hpc/zaratan/jpeglm_overfit/o0_ar.yaml
python scripts/hpc/zaratan/jpeglm_overfit/submit.py scripts/hpc/zaratan/jpeglm_overfit/o1_fim_k1.yaml
```

Outputs:
- O0: `RUN/eval_overfit_ar/final/{train,val}/{summary.json,windows.jsonl}`
- FIM runs: `RUN/eval_overfit_fim/final/{train,train_newholes,val}/` (`eval_fim_avclm.py`; see `byte_exact_*` and
  `first_divergence_*` in the summary and per-sample rows)

To re-evaluate a saved checkpoint without training, resubmit the same YAML with
`OVERFIT_EVAL_ONLY=1` (and optionally `OVERFIT_EVAL_CHECKPOINT=step-00010000`).
Before trusting a number, check that the training log's `[window-gop] usable=N` line shows
the expected number of windows, and that exposures per window (≈ 16 × steps / N) are in
the ~10k+ range. If they aren't, raise `training.steps`.
