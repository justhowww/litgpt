# Offline Sampling Sweep

Runs `scripts/byte/eval/eval_checkpoints.py` over a fixed checkpoint list and
several sampling settings. This is an offline diagnostic, not a training job.

The sweep is intentionally one-dimensional:

- temperatures: `0.1, 0.3, 0.5, 0.7, 1.0` with unrestricted byte sampling
- top-k: `1, 2, 5, 10` at temperature 1.0
- top-p: `0.7, 0.8, 0.9, 0.95` at temperature 1.0

Each setting writes an independent eval directory under `OUT_ROOT`.
The launcher also writes one combined comparison table:

```text
OUT_ROOT/sampling_sweep_summary.csv
```

Rows are prefixed with `setting`, `temperature`, `top_k`, and `top_p`, followed
by the checkpoint-level metrics from each setting.

Run:

```bash
NUM_SAMPLES=20 \
NUM_VISUALIZATIONS=4 \
BEST_OF_N=64 \
CHECKPOINT_STEPS="500 1000 1500 2000 2500 3000 3500 4000 4500" \
bash scripts/hpc/zaratan/eval_sampling_sweep/submit_h100.sh
```

Defaults evaluate the within-video CE-only run:

```text
$HOME/scratch.metzler-prj/OpenVid-1M_Data/data/runs/byte-stage2-fim-psm-ce-only-within-video-1k-8k
```

Override `RUN_NAME`, `RUN_DIR`, `OUT_ROOT`, or `SPLIT_BY_VIDEO=1` for other
experiments. Existing settings with `summary.csv` are skipped by default;
set `RESUME_SKIP=0` to overwrite them.

Beam search is not included yet. The current diagnostic separates greedy from
best@N stochastic samples; if conservative sampling still fails, beam search is
the next decode-side search to add.
