# MRT-primary oracle-length, desaturated reward (run alpha)

Paired with `stage2_fim_psm_mrt_primary_learned_stop_desat_1k_8k` (run beta).
Tests whether oracle-length training-time policy contradicts visual-only
supervision, by comparing the deployable (learned-stop) eval PSNR of both
checkpoints at matched compute.

## What differs

| Knob | alpha (this) | beta |
|---|---|---|
| `MRT_ORACLE_LENGTH` | 1 | 0 |
| `MRT_LEARNED_EOS` | 0 | 1 |

Everything else identical: reward function and scale, CE weight, MRT weight,
candidate count, GT-in-pool, initial checkpoint, compute, eval probes.

## What both runs evaluate

Both checkpoints log strict-mode PSNR/SSIM under both probes:

- `reconstruction/fim/model_learned/strict/psnr_mean_valid` — learned-stop eval.
- `reconstruction/fim/model_oracle/strict/psnr_mean_valid` — oracle-length eval.

## Four-cell decision table

Let `P_train_eval` = strict PSNR of the `<train>`-trained checkpoint under
the `<eval>` probe at matched compute.

| Observation | Reading |
|---|---|
| `P_β_learned ≥ P_α_learned` and `P_β_oracle ≈ P_α_oracle` | Oracle training adds no value; gap on deployable task says it didn't transfer. Hypothesis confirmed (weak form). |
| `P_α_learned << P_α_oracle` (large gap, e.g. > 3 dB) and `stop_no_stop_rate ≈ 1` on alpha's learned-stop probe | Alpha can fill content with given length but cannot stop. Hypothesis confirmed (strong form: contradictory). |
| `P_β_learned < P_α_learned` | Oracle pretraining transfers better than learned-stop training. Hypothesis falsified. |
| All four PSNRs similar and low | Both confounded by some orthogonal failure (e.g. reward still effectively saturated). Check MRT diagnostics first. |

## Contradictory-fingerprint diagnostic

Specifically on alpha's learned-stop probe:

- `reconstruction/fim/stop_no_stop_rate` near 1.
- `reconstruction/fim/model_learned/strict/decode_rate` low (cap-hit
  bitstreams don't decode strictly).
- `reconstruction/fim/model_oracle/strict/psnr_mean_valid` high on the same
  checkpoint.

That triple is the mechanical demonstration that the policy alpha learned is
incompatible with the deployable task, not just slightly worse.

```bash
CLEANUP=0 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_mrt_primary_oracle_desat_1k_8k/submit_h100.sh
```
