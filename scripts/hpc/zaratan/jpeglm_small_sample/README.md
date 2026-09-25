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

The evaluation fixes the corruption length to 64, 128, 256, 400, or 600 bytes,
with two IDR and two P-frame examples per length in each split. It records the
exact selected holes in a shared manifest and refuses to compare runs whose
sample definitions differ. It reports missing-byte CE in bits/byte (EOS
excluded), byte perplexity/accuracy, EOS probability/rank, I/P and length
breakdowns, per-byte NLL JSONL, and HTML syntax-annotated loss heatmaps. It is
teacher-forced only; it does not claim free-run repair success. Training holes
continue to change and are not exact replay pairs.

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
`eval_small_sample/final/<split>` directory before retrying. It does not delete
or overwrite anything automatically.
