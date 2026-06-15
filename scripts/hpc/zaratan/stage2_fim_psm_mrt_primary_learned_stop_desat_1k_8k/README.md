# MRT-primary learned-stop, desaturated reward

One variable changed vs. `stage2_fim_psm_mrt_primary_learned_stop_1k_8k`:
the MRT reward `smooth_mse = mse / (mse + tau)` is replaced with the new
`scaled_mse = mse_weight * mse` (no clipping, no asymptote).

The scale is calibrated to the observed sampled-MSE median from prior
saturated-reward runs (~0.035), so typical sampled risk is order 1:

```
r_i = MSE_i / 0.035   ->   MRT_MSE_WEIGHT = 1/0.035 = 28.5714

  MSE 0.01   ->  risk 0.29
  MSE 0.035  ->  risk 1.0    (typical candidate)
  MSE 0.10   ->  risk 2.86
  decode fail -> risk 5.0    (upper tail of typical decodes)
```

The scale does not change candidate ordering; it only sets MRT gradient
magnitude relative to CE. Keep the scale fixed across runs so risk curves
remain comparable. If after 100-500 steps the ratio
`optimization/mrt_grad_norm / optimization/weighted_ce_grad_norm` is outside
~2-10x (the target for an MRT-primary experiment), adjust `MRT_WEIGHT`, NOT
`MRT_MSE_WEIGHT`.

GT is kept in the 8-candidate pool intentionally — when sampled candidates
are all bad, GT provides an upper anchor for the policy gradient.

The eval probe now runs both `learned-EOS` (the primary training policy)
and `oracle-length` (diagnostic) on each checkpoint. Metrics:

- `reconstruction/fim/model_learned/strict/psnr_mean_valid` — real objective.
- `reconstruction/fim/model_oracle/strict/psnr_mean_valid` — diagnostic;
  isolates content recovery from stopping.

Decomposition at final eval:

| Oracle PSNR | Learned-stop PSNR | Conclusion |
|---|---|---|
| up over first-eval peak | up | Desaturated MRT works, content + EOS learnable. |
| up over first-eval peak | flat | Content works, EOS needs explicit supervision. |
| flat | flat | MRT signal still insufficient; escalate (candidates, temperature, learned critic). |

Diagnostics to watch (logged by `mrt_candidate_diagnostics`).

Under a non-saturating reward, GT risk is exactly 0, so the q_GT * r_GT
contribution to `mrt/expected_risk` is identically zero and
`mrt/expected_risk == mrt/sampled_expected_risk_contribution`. The useful
disambiguation when expected_risk falls is:

- `mrt/sampled_q_sum` (= 1 - q_GT) falls while
  `mrt/sampled_conditional_expected_risk` is roughly flat
  -> probability mass shrank toward GT; sampled candidates did not improve.
  Same degeneracy as the saturated-reward run, manifesting through q
  instead of through risks.
- `mrt/sampled_q_sum` roughly flat while
  `mrt/sampled_conditional_expected_risk` falls
  -> sampled candidates themselves became visually better. Real visual
  signal. This is the success mode.

Gradient balance check (after 100-500 steps):

- `optimization/mrt_grad_norm / optimization/weighted_ce_grad_norm` in [2x, 10x]
  -> MRT is primary, gradient balance is right.
- ratio < 2x -> raise `MRT_WEIGHT`.
- ratio > 10x -> lower `MRT_WEIGHT`.

Do NOT adjust `MRT_MSE_WEIGHT` to fix gradient balance; that breaks risk-curve
comparability across runs.

Exploration triggers (only if reward is healthy but PSNR plateaus):

- `mrt/sampled_risk_std` near zero -> sampled candidates indistinguishable.
  Widen exploration (more candidates, higher temperature) before retuning.
- `mrt/sampled_risk_max - mrt/sampled_risk_min` small and shrinking ->
  local-only climb, same fix.

```bash
CLEANUP=0 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_mrt_primary_learned_stop_desat_1k_8k/submit_h100.sh
```
