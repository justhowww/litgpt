# Stage 2 PSM MRT with smooth MSE risk

This run matches `stage2_fim_psm_mrt_1k_8k` except for the visual-risk
mapping:

```text
risk(decoded) = MSE / (MSE + 0.002)
risk(decode failure) = 1
```

The smooth mapping remains bounded but preserves ordering among candidates
whose PSNR is below the former 27 dB clipping threshold. It initializes from
the same 1,000-step supervised PSM+EOS checkpoint as the clipped-risk run.

Launch on H100:

```bash
CLEANUP=0 \
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
bash scripts/hpc/zaratan/stage2_fim_psm_mrt_smooth_1k_8k/submit_h100.sh
```
