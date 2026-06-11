# Stage 2.5: EOS-weighted PSM

This is the matched follow-up to `stage2_fim_psm`. It keeps the model, PSM
layout, data mixture, optimizer, seed, context length, and gap distribution
fixed while assigning `10x` loss to positive `SEQ_EOS` targets.

The full run also evaluates 50 fixed samples every 5,000 steps with:

- learned stopping and oracle-length model generation;
- normal playback and FFmpeg `-err_detect explode`;
- FIM ground-truth, deleted-gap, and deterministic-random baselines.

Run the smoke test:

```bash
bash scripts/hpc/zaratan/stage2_fim_psm_eos_weight/submit_smoke.sh
```

Launch the 10,000-step H100 gate:

```bash
STEPS=10000 \
  bash scripts/hpc/zaratan/stage2_fim_psm_eos_weight/submit_h100.sh
```

The default output is
`runs/byte-stage2-fim-psm-eos-weight10/`. Continue beyond 10,000 steps only if
exact stopping improves without materially reducing oracle-length FIM quality.
