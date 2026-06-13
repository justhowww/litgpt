# MRT-primary oracle-length FIM diagnostic

This run isolates decoder-risk optimization:

- FIM-only training and evaluation;
- `1024-8192` byte gaps everywhere, matching the balanced-EOS baseline;
- exact oracle generation length with no EOS supervision or EOS action;
- MRT every optimizer step with 8 candidates;
- `1.0 * MRT + 0.05 * CE`;
- initialization from the 1,000-step balanced-EOS checkpoint.

EOS is excluded from targets, candidate sampling, and MRT scoring. Ordinary CE
still suppresses its logit through full-vocabulary normalization; it receives
no positive terminator target. PSM vocab sizes with and without EOS both pad to
264, so the balanced-EOS checkpoint tensors remain shape-compatible.

Launch on H100:

```bash
CLEANUP=0 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_mrt_primary_oracle_1k_8k/submit_h100.sh
```
