# Small-sample JPEG-LM runs

Only these new runs use YAML. Existing training scripts and checkpoints are
unchanged. Each file is complete; there is no inheritance or ambient override
of its scientific settings. The launcher validates it, records the resolved
settings and a frozen YAML copy under the new run directory, and refuses
to submit to a populated legacy run directory or to resume with changed
settings. The small-sample model uses 27 global layers and width 1024, not the
previous ~7B setup: its full-training state is not expected to fit safely on
two A100 40 GB cards. Both model families get the same dimensions, patch size,
and objective.

From the repository root on Zaratan, inspect without writing or submitting:

```bash
python scripts/hpc/zaratan/jpeglm_small_sample/submit.py \
  scripts/hpc/zaratan/jpeglm_small_sample/pythia.yaml --dry-run
```

Submit a run:

```bash
python scripts/hpc/zaratan/jpeglm_small_sample/submit.py \
  scripts/hpc/zaratan/jpeglm_small_sample/pythia.yaml
```

Use `qwen3.yaml` for the matched Qwen3 run. The initial configs use 256 manifest
rows and 5,000 optimizer steps; review those two settings before submission.
Each job uses one node with two A100s for training. Once `final` is saved, the
same allocation evaluates train and validation on one GPU each. An evaluation
failure does not remove the checkpoint; inspect `logs/small-sample-eval-*.err`
and rerun evaluation separately if needed.

After training, the small-sample job adds `learning/train/*` and
`learning/val/*` aliases to its TensorBoard run. Original tags are preserved.
The paired `full_sequence_ce_nats` curves are both CE, while
`missing_byte_ce_nats` excludes EOS and `bridge_ce_including_eos_nats` includes
it; those last two are deliberately not presented as an exact train/val pair.
The total optimized objective is named separately from validation CE. Only the
isolated small-sample job runs this aliasing step; larger runs are unaffected.

At optimizer step 100, each rank writes a readable
`memory_profile/step-00000100/rank-N.json` and a PyTorch allocation-history
`rank-N.pickle`. The JSON includes forward/backward/optimizer endpoints and
identified local parameter, optimizer-state, and gradient storage. The pickle
can be opened in PyTorch's CUDA memory visualizer to inspect transient
activations, FSDP all-gathers, and temporary buffers. These are evidence for
memory attribution, not an exact additive partition of total GPU memory;
non-PyTorch allocations such as NCCL are not captured by PyTorch's allocator.
Only the configured small-sample step enables recording, and larger runs keep
their existing behavior.

The frozen YAML requests corruption lengths of 64, 128, 256, 400, and 600 bytes
for both frame types. The `feasible_35_holes_v1` evaluation protocol scores five
IDR examples at each length and five P-frame examples at 64 and 128 bytes (35
holes per split). The 256-video subset has no eligible held-out P-frame window
for a 256-byte cut, so P/256B, P/400B, and P/600B are explicitly marked as
not evaluated in `summary.json`; they are never silently counted as failures.
The protocol applies identically to train and validation, and to Qwen3 and
Pythia. It records the exact selected holes in a shared manifest and refuses to
compare runs whose
sample definitions differ. It reports missing-byte CE in bits/byte (EOS
excluded), byte perplexity/accuracy, EOS probability/rank, I/P and length
breakdowns, per-byte NLL JSONL, and HTML syntax-annotated loss heatmaps. It is
teacher-forced only; it does not claim free-run repair success. Training holes
continue to change and are not exact replay pairs.
Each severity requires five distinct eligible windows, but a window may recur
at a different severity. Eligibility is checked against that severity's actual
cut length, not the largest length in the schedule.
This feasible selection is versioned as `feasible_35_holes_v1` under both
the shared sample-set directory and `eval_small_sample/final/`, so earlier
partial evaluation files are preserved and never mistaken for this protocol.

The summary also pools byte-weighted CE by parser ownership: stream/header
structure, content-dependent macroblock/prediction/residual coding, bytes
straddling both, and unclassified bytes. Each bucket reports its byte fraction
and contribution to overall missing-byte CE; unknown parser coverage is never
silently counted as syntax. The same breakdown appears overall and by frame
type/corruption length.

The launcher delegates to the existing `jpeglm_pretrain/submit.sh` and
`train.sh`; it does not replace their behavior for any previous run.

To rerun only the teacher-forced evaluation after a successful training job,
inside a one-GPU allocation with the project environment active:

```bash
python scripts/hpc/zaratan/jpeglm_small_sample/eval_tf.py \
  /home/huangyh/scratch.metzler-prj/OpenVid-1M_Data/data-jpeglm/runs/byte-jpeglm-small-27x1024-pythia-patch256/small_sample_config.yaml \
  --split val
```

The evaluator protects existing output; move aside an incomplete
`eval_small_sample/final/feasible_35_holes_v1/<split>` directory before
retrying the same protocol. It does not delete or overwrite anything
automatically.

To rerun both evaluation splits in the two-A100 Slurm job without retraining,
set `SMALL_SAMPLE_EVAL_ONLY=1` when invoking `submit.py` with the unchanged
YAML. This requires an existing final checkpoint and training split, and the
evaluator still refuses to overwrite any existing split output.

```bash
SMALL_SAMPLE_EVAL_ONLY=1 python scripts/hpc/zaratan/jpeglm_small_sample/submit.py \
  scripts/hpc/zaratan/jpeglm_small_sample/qwen3.yaml
```
