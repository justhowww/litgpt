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

`qwen3_idr50_20k.yaml` is a separate follow-up run. Its training loader first
chooses an eligible IDR or non-IDR frame with equal probability, then chooses
uniformly within that class and draws the gap/position as before. JPEG-LM has
no B-frames, so its non-IDR class is P. If a GOP has only one eligible class,
the loader uses that class and the logged realized fraction exposes the change.
The held-out in-loop validation loader keeps its original uniform frame choice;
the fixed-hole evaluator still uses the same shared 35-hole set. Training logs
the local-rank realized IDR draw fraction and separate IDR/P missing-byte CE
over each 100-step block, without recomputing vocabulary CE. The 20k-step run
saves permanent checkpoints at 5k, 10k, 15k, and 20k; a rolling `latest`
points at each milestone. The first 5k steps have a **different cosine LR**
than the original 5k-step run, so their comparison is not a pure sampling-only
ablation; a matched-schedule control would be needed to attribute all changes.

Submit the new training run from the repository root on Zaratan:

```bash
python scripts/hpc/zaratan/jpeglm_small_sample/submit.py \
  scripts/hpc/zaratan/jpeglm_small_sample/qwen3_idr50_20k.yaml
```

After step 5,000 has been saved, evaluate that milestone on both splits
without restarting training:

```bash
SMALL_SAMPLE_EVAL_ONLY=1 SMALL_SAMPLE_EVAL_CHECKPOINT=step-00005000 \
python scripts/hpc/zaratan/jpeglm_small_sample/submit.py \
  scripts/hpc/zaratan/jpeglm_small_sample/qwen3_idr50_20k.yaml
```

The architecture-matched Pythia follow-up uses
`pythia_idr50_20k.yaml`. To queue it only after an existing Qwen job
finishes successfully, pass its Slurm ID with `--after-jobid`. This is a
submission dependency, not a saved training hyperparameter. Each small-sample
job performs its own train- and validation-split teacher-forced evaluations
after its training and final checkpoint finish.

```bash
python scripts/hpc/zaratan/jpeglm_small_sample/submit.py \
  scripts/hpc/zaratan/jpeglm_small_sample/pythia_idr50_20k.yaml \
  --after-jobid 22584445
```

To keep Pythia training and the Qwen/Pythia evaluations as three separate
dependent jobs, set `SMALL_SAMPLE_SKIP_EVAL=1` on the Pythia training
submission. Then submit the Qwen evaluation with
`SMALL_SAMPLE_EVAL_ONLY=1 SMALL_SAMPLE_EVAL_CHECKPOINT=step-00020000`
dependent on Pythia; finally submit the Pythia `final` evaluation dependent on
that Qwen evaluation. The Qwen step-20k checkpoint has the same terminal
training step as `final`, but uses a separate evaluation output directory;
the original Qwen job already evaluates `final` inline before it finishes.

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
