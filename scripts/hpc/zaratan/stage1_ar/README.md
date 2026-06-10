# Stage 1: AR Pretraining

These launchers train full target VCL NAL units autoregressively with
`p_fim=0`.

```bash
bash scripts/hpc/zaratan/stage1_ar/submit_smoke.sh
bash scripts/hpc/zaratan/stage1_ar/submit_h100.sh
```

The default output directory is
`scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage1/`.
