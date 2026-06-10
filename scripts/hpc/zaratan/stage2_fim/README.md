# Stage 2: Mixed AR/FIM Pretraining

These launchers train from scratch with a default 50% AR / 50% FIM sample
mixture. They use a separate output directory and cannot resume Stage 1
checkpoints accidentally.

```bash
bash scripts/hpc/zaratan/stage2_fim/submit_smoke.sh
P_FIM=0.5 STEPS=100000 \
  bash scripts/hpc/zaratan/stage2_fim/submit_h100.sh
```

The default output directory is
`scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage2-fim/`.

Every 1,000 steps, separate AR and FIM decoder probes run on fixed validation
samples. TensorBoard records them under `reconstruction/ar/*` and
`reconstruction/fim/*`.
