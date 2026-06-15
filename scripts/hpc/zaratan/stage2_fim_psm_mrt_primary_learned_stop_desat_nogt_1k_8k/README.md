# MRT-primary learned-stop, desaturated reward, GT dropped from pool

One variable changed vs. `stage2_fim_psm_mrt_primary_learned_stop_desat_1k_8k`:
`MRT_INCLUDE_GROUND_TRUTH=0`. All 8 candidates are model-sampled; the
ground-truth replacement is not prepended to the pool.

Tests whether GT-in-pool under the new `scaled_mse` reward is:

- **anchoring** (the policy gradient benefits from an upper-bound reference,
  especially when sampled candidates are all bad) -> PSNR rises *less* than
  the desat-with-GT run, confirming GT-keep was the right call;
- **hijacking** (q_GT still attracts probability mass under the new reward,
  producing CE-like reranking even without saturation) -> PSNR rises *more*
  than the desat-with-GT run, falsifying the GT-keep instinct.

Same reward calibration as the desat primary so risk curves are directly
comparable: `r_i = MSE_i / 0.035`, `MRT_MSE_WEIGHT=28.5714`,
`MRT_DECODE_FAILURE_WEIGHT=5.0`.

GT-absent diagnostics:
- `mrt/score_gt`, `mrt/score_margin_gt_vs_sampled` -> not emitted (no GT).
- `mrt/ground_truth_probability` -> NaN.
- `mrt/sampled_q_sum` -> identically 1.0.
- `mrt/sampled_conditional_expected_risk` -> equals `mrt/expected_risk`.
- `mrt/sampled_risk_*` -> now spans the full 8-candidate pool.

The interesting cross-run comparison vs. the desat-with-GT primary is at the
same training step:
- `mrt/sampled_risk_min` and `mrt/sampled_risk_mean` -> does the sampled
  distribution itself improve without GT to anchor it?
- `mrt/expected_risk` trajectory -> does it fall, and if so, is the fall
  driven entirely by sampled-candidate improvement (as it must be without GT)?
- `reconstruction/fim/model_learned/strict/psnr_mean_valid` -> the answer to
  whether GT was net-positive or net-negative.

```bash
CLEANUP=0 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_mrt_primary_learned_stop_desat_nogt_1k_8k/submit_h100.sh
```
