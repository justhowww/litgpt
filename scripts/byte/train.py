"""Launch H.264 byte-domain pretraining.

This launcher keeps the byte-specific model and data settings explicit while
reusing LitGPT's pretraining loop, checkpointing, logging, and distributed setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from litgpt.args import EvalArgs, TrainArgs
from litgpt.byte.grpo import GRPOConfig
from litgpt.byte.mrt import MRTConfig, MRT_RISK_MODES
from litgpt.config import Config
from litgpt.byte.data import (
    DATASET_MODES,
    FIM_FORMATS,
    FIM_LOSS_SCOPES,
    REFERENCE_MODES,
    ByteDataConfig,
    ByteDataModule,
    vocab_size_for_fim_format,
)
from litgpt.byte.reconstruction import ReconstructionEvalConfig
from litgpt.byte.free_run_eval import FreeRunEvalConfig
from litgpt.byte.h264_mask import SLICE_LAYOUT_MACROBLOCK, SLICE_LAYOUTS
from litgpt.pretrain import setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-name", default="byte-stage1")
    parser.add_argument(
        "--nal-index-path",
        type=Path,
        default=None,
        help=(
            "Precomputed SQLite NAL index. When omitted, the data module uses "
            "nal_index.sqlite next to the manifest if it exists."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out/byte-stage1"))
    parser.add_argument(
        "--block-size",
        type=int,
        default=16384,
        help=(
            "Transformer context length in positions. With --byte-patch-size K, "
            "one position represents K byte/control ids and the raw-byte window "
            "budget is approximately block_size*K."
        ),
    )
    parser.add_argument(
        "--byte-patch-size",
        type=int,
        choices=(1, 4, 8),
        default=1,
        help=(
            "Consecutive byte/control ids per transformer position. Values above "
            "one use a MEGABYTE global/local Transformer and require training "
            "from scratch."
        ),
    )
    parser.add_argument(
        "--megabyte-local-layers",
        type=int,
        default=4,
        help="Number of shared causal Transformer layers inside each byte patch.",
    )
    parser.add_argument(
        "--megabyte-local-embd",
        type=int,
        default=512,
        help="Embedding width of the MEGABYTE local Transformer.",
    )
    parser.add_argument(
        "--megabyte-local-heads",
        type=int,
        default=8,
        help="Attention heads in the MEGABYTE local Transformer.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--reconstruction-eval-interval", type=int, default=0)
    parser.add_argument("--reconstruction-eval-samples", type=int, default=5)
    parser.add_argument(
        "--reconstruction-visualization-samples",
        type=int,
        default=0,
        help=(
            "Log this many fixed FIM decoder comparison panels to TensorBoard; "
            "0 disables image logging."
        ),
    )
    parser.add_argument(
        "--reconstruction-task",
        choices=("ar", "fim", "both"),
        default="ar",
        help="Evaluate AR slices, FIM missing spans, or both decoder tasks.",
    )
    parser.add_argument("--reconstruction-timeout-sec", type=int, default=30)
    parser.add_argument("--reconstruction-max-target-bytes", type=int, default=2048)
    # Free-running (H0 generation) validation probe: parser-based survival-length +
    # validity on a fixed SPS-anchored continuation set. Measures what val CE
    # cannot (exposure-bias / illegal-token tail). 0 disables. Window/AR only.
    parser.add_argument("--free-run-eval-interval", type=int, default=0)
    parser.add_argument("--free-run-eval-clips", type=int, default=8)
    parser.add_argument("--free-run-prefix-frames", type=int, default=8)
    parser.add_argument("--free-run-cont-frames", type=int, default=4)
    parser.add_argument("--free-run-temperature", type=float, default=1.0)
    parser.add_argument("--free-run-max-gen-multiple", type=float, default=2.0)
    parser.add_argument(
        "--free-run-slice-layout",
        choices=SLICE_LAYOUTS,
        default=SLICE_LAYOUT_MACROBLOCK,
    )
    parser.add_argument(
        "--reconstruction-oracle-length",
        action="store_true",
        help="Also generate exactly the known target length while excluding control tokens.",
    )
    parser.add_argument(
        "--reconstruction-learned-eos",
        action="store_true",
        help=(
            "Allow EOS stopping during reconstruction even when EOS is absent "
            "from dataset labels."
        ),
    )
    parser.add_argument(
        "--reconstruction-error-exploding",
        action="store_true",
        help=(
            "Also report the legacy FFmpeg -err_detect explode diagnostic. "
            "Primary reconstruction metrics always disable concealment and "
            "use strict syntax checking."
        ),
    )
    parser.add_argument(
        "--reconstruction-fim-baselines",
        action="store_true",
        help="Evaluate ground-truth, deleted-gap, and deterministic-random FIM baselines.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        default="ffmpeg",
        help="FFmpeg executable used by reconstruction validation.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-manifest-rows",
        type=int,
        default=0,
        help="Limit clips indexed by the DataModule; 0 uses the full manifest.",
    )
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--num-ref-slices", type=int, default=1)
    parser.add_argument("--reference-mode", choices=REFERENCE_MODES, default="normal")
    parser.add_argument("--target-nal-types", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--dataset-mode",
        choices=DATASET_MODES,
        default="slice",
        help="'slice' = ByteSliceDataset (ref+target slice). 'window' = "
        "ByteStreamWindowDataset, the multi-frame contiguous-stream window: AR (H0) "
        "at --p-fim 0, masked-span infill at --p-fim > 0. Window FIM reuses "
        "--fim-format/--fim-min-gap/--fim-max-gap/--use-eos and reads "
        "--slice-header-guard-bytes as a per-frame head guard; the remaining "
        "slice-only flags are ignored.",
    )
    parser.add_argument(
        "--window-min-frames",
        type=int,
        default=2,
        help="Minimum VCL frames per stream window (dataset-mode=window).",
    )
    parser.add_argument("--p-fim", type=float, default=0.0)
    parser.add_argument(
        "--fixed-fim-holes",
        action="store_true",
        help=(
            "Backward-compatible alias for --fixed-fim-holes-per-window 1."
        ),
    )
    parser.add_argument(
        "--fixed-fim-holes-per-window",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Finite FIM diversity curriculum for window mode. Cache K distinct "
            "holes per training window and sample uniformly among them; 0 redraws "
            "a fresh changing hole on every FIM access. The AR/FIM mixture remains "
            "controlled independently by --p-fim."
        ),
    )
    parser.add_argument("--fim-format", choices=FIM_FORMATS, default="bridge")
    parser.add_argument(
        "--fim-loss-scope",
        choices=FIM_LOSS_SCOPES,
        default="span",
        help=(
            "FIM supervision scope: 'span' trains only the missing span and EOS; "
            "'full' trains next-token prediction across the complete reordered "
            "FIM sequence."
        ),
    )
    parser.add_argument(
        "--use-eos",
        action="store_true",
        help="Append SEQ_EOS after each AR target / FIM span so the model learns "
        "to terminate. Default off uses oracle lengths for generation.",
    )
    parser.add_argument(
        "--ce-loss-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to supervised byte CE gradients.",
    )
    parser.add_argument(
        "--ce-byte-only",
        action="store_true",
        help=(
            "Normalize supervised CE over raw byte ids 0-255 only. "
            "All supervised labels must therefore be bytes."
        ),
    )
    parser.add_argument(
        "--eos-loss-weight",
        type=float,
        default=1.0,
        help="Positive loss weight for SEQ_EOS targets. Requires --use-eos when not 1.",
    )
    parser.add_argument(
        "--eos-aux-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a class-balanced EOS-vs-non-EOS auxiliary loss. "
            "Requires --use-eos when positive."
        ),
    )
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument("--no-sps-pps-conditioning", action="store_true")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.01,
        help=(
            "Fraction held out for validation. With --split-by-video this is "
            "the fraction of source videos; without it, the fraction of slice "
            "samples (the legacy slice-level split)."
        ),
    )
    parser.add_argument(
        "--split-by-video",
        action="store_true",
        help=(
            "Split train/val by source video (h264_path), not by slice. "
            "Eliminates within-video leakage and produces a genuinely held-out "
            "video evaluation set."
        ),
    )
    parser.add_argument("--no-region-id", action="store_true")
    parser.add_argument("--no-offset-id", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initial-checkpoint-dir",
        type=Path,
        default=None,
        help="Initialize model weights from a LitGPT checkpoint directory.",
    )
    parser.add_argument("--mrt-interval", type=int, default=0)
    parser.add_argument("--mrt-start-step", type=int, default=0)
    parser.add_argument("--mrt-num-candidates", type=int, default=16)
    parser.add_argument("--mrt-context-pool-size", type=int, default=64)
    parser.add_argument("--mrt-max-target-bytes", type=int, default=2048)
    parser.add_argument(
        "--mrt-oracle-length",
        action="store_true",
        help="Generate and score exactly the known missing-span byte length.",
    )
    parser.add_argument(
        "--mrt-learned-eos",
        action="store_true",
        help="Allow MRT to learn EOS stopping without supervised EOS targets.",
    )
    parser.add_argument(
        "--no-mrt-ground-truth",
        dest="mrt_include_ground_truth",
        action="store_false",
        help=(
            "Drop GT from the MRT candidate pool; all num_candidates are "
            "model-sampled. Default behavior keeps GT as an upper anchor."
        ),
    )
    parser.set_defaults(mrt_include_ground_truth=True)
    parser.add_argument("--mrt-temperature", type=float, default=1.0)
    parser.add_argument("--mrt-candidate-alpha", type=float, default=1.0)
    parser.add_argument("--mrt-weight", type=float, default=4.0)
    parser.add_argument(
        "--mrt-risk-mode",
        choices=MRT_RISK_MODES,
        default="clipped_mse",
        help=(
            "Visual-risk mapping. clipped_mse preserves the original "
            "min(max_risk, mse_weight*MSE) experiment; smooth_mse uses "
            "MSE/(MSE+mse_tau) and preserves ordering without hard clipping."
        ),
    )
    parser.add_argument("--mrt-mse-weight", type=float, default=1000.0)
    parser.add_argument(
        "--mrt-mse-tau",
        type=float,
        default=0.002,
        help="MSE midpoint for smooth_mse risk: MSE=tau maps to risk 0.5.",
    )
    parser.add_argument("--mrt-decode-failure-weight", type=float, default=2.0)
    parser.add_argument("--mrt-max-risk", type=float, default=2.0)
    parser.add_argument("--mrt-decode-workers", type=int, default=8)
    parser.add_argument("--grpo-interval", type=int, default=0)
    parser.add_argument("--grpo-start-step", type=int, default=0)
    parser.add_argument("--grpo-group-size", type=int, default=64)
    parser.add_argument("--grpo-ar-pool-size", type=int, default=16)
    parser.add_argument("--grpo-fim-pool-size", type=int, default=16)
    parser.add_argument("--grpo-max-target-bytes", type=int, default=2048)
    parser.add_argument("--grpo-temperature", type=float, default=1.0)
    parser.add_argument("--grpo-top-k", type=int, default=0)
    parser.add_argument("--grpo-top-p", type=float, default=1.0)
    parser.add_argument(
        "--grpo-kl-coeff",
        type=float,
        default=0.02,
        help="KL-to-reference-policy penalty weight. 0 disables the KL term "
        "(and the frozen reference checkpoint load).",
    )
    parser.add_argument(
        "--grpo-psnr-cap",
        type=float,
        default=40.0,
        help="PSNR (dB) normalization ceiling for the reward's decode-success shaping.",
    )
    parser.add_argument("--grpo-decode-failure-reward", type=float, default=-1.0)
    parser.add_argument(
        "--grpo-mu",
        type=int,
        default=1,
        help=(
            "Inner gradient steps per sampled group via the clipped ratio "
            "surrogate, before resampling. mu=1 (default) is plain on-policy "
            "GRPO; mu>1 reuses one (expensive) rollout for extra updates."
        ),
    )
    parser.add_argument("--grpo-clip-range", type=float, default=0.2)
    parser.add_argument(
        "--grpo-learned-eos",
        action="store_true",
        help=(
            "Rollouts stop via the model's own SEQ_EOS prediction (unknown "
            "length, widened generation budget) instead of oracle-length "
            "generation (told the true target length, EOS masked out). "
            "Requires the checkpoint to have been trained with --use-eos."
        ),
    )
    parser.add_argument(
        "--grpo-generation-budget-multiplier",
        type=float,
        default=2.0,
        help="Learned-EOS generation budget, as a multiple of the true target length.",
    )
    parser.add_argument("--grpo-timeout-sec", type=int, default=30)
    parser.add_argument("--grpo-decode-workers", type=int, default=8)
    parser.add_argument("--grpo-ffmpeg-binary", default="ffmpeg")
    parser.add_argument(
        "--grpo-reference-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Frozen initial-policy checkpoint the GRPO KL penalty is measured "
            "against. Required when --grpo-kl-coeff > 0. Kept independent of "
            "--initial-checkpoint-dir/--resume so the KL anchor survives a "
            "resumed phase-2 run; typically the same checkpoint as "
            "--initial-checkpoint-dir on a fresh phase-2 launch."
        ),
    )
    parser.add_argument("--precision", default=None)
    parser.add_argument("--logger-name", default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fixed_fim_holes_per_window < 0:
        raise ValueError("--fixed-fim-holes-per-window must be non-negative")
    if args.fixed_fim_holes:
        if args.fixed_fim_holes_per_window not in (0, 1):
            raise ValueError(
                "--fixed-fim-holes is an alias for K=1 and conflicts with "
                "--fixed-fim-holes-per-window > 1"
            )
        args.fixed_fim_holes_per_window = 1
    if args.ce_loss_weight < 0:
        raise ValueError("--ce-loss-weight must be non-negative")
    if args.eos_loss_weight <= 0:
        raise ValueError("--eos-loss-weight must be positive")
    if args.eos_loss_weight != 1.0 and not args.use_eos:
        raise ValueError("--eos-loss-weight requires --use-eos")
    if args.eos_aux_loss_weight < 0:
        raise ValueError("--eos-aux-loss-weight must be non-negative")
    if args.eos_aux_loss_weight > 0 and not args.use_eos:
        raise ValueError("--eos-aux-loss-weight requires --use-eos")
    if args.ce_byte_only and args.use_eos:
        raise ValueError("--ce-byte-only cannot train supervised EOS targets")
    if args.fixed_fim_holes_per_window > 0 and (
        args.dataset_mode != "window" or args.p_fim <= 0.0
    ):
        raise ValueError(
            "finite fixed FIM holes require --dataset-mode window and --p-fim > 0"
        )
    if args.resume and args.initial_checkpoint_dir is not None:
        raise ValueError("--resume and --initial-checkpoint-dir are mutually exclusive")
    if args.byte_patch_size > 1 and args.reconstruction_eval_interval > 0:
        raise ValueError(
            "Reconstruction evaluation is not yet ported to byte patches; "
            "set --reconstruction-eval-interval 0"
        )
    if args.byte_patch_size > 1 and args.mrt_interval > 0:
        raise ValueError(
            "MRT generation is not yet ported to byte patches; set --mrt-interval 0"
        )
    if (
        args.mrt_interval > 0
        and not args.use_eos
        and not args.mrt_oracle_length
        and not args.mrt_learned_eos
    ):
        raise ValueError(
            "MRT requires --use-eos, --mrt-oracle-length, or --mrt-learned-eos"
        )
    if args.mrt_oracle_length and args.mrt_learned_eos:
        raise ValueError(
            "--mrt-oracle-length and --mrt-learned-eos are mutually exclusive"
        )
    # Reconstruction probes support running learned-EOS generation as the
    # primary policy while also evaluating an oracle-length candidate on the
    # same checkpoint; the eval path emits both as separate metric prefixes.
    if args.mrt_interval > 0 and args.p_fim <= 0:
        raise ValueError("MRT requires a non-zero --p-fim")
    if args.grpo_interval > 0 and args.p_fim <= 0:
        raise ValueError("GRPO requires a non-zero --p-fim for its FIM pool")
    if (
        args.grpo_interval > 0
        and args.grpo_kl_coeff > 0
        and args.grpo_reference_checkpoint_dir is None
    ):
        raise ValueError(
            "--grpo-kl-coeff > 0 requires --grpo-reference-checkpoint-dir"
        )
    if args.grpo_interval > 0 and args.grpo_learned_eos and not args.use_eos:
        raise ValueError("--grpo-learned-eos requires --use-eos")
    # Window FIM's hole spans many NALs (on a slice-max-mbs=1 corpus a 64-1400 byte
    # hole covers ~100+ of them), so the window-AR offset convention -- arange within
    # each NAL, reset at every boundary -- has no coherent value across the generated
    # span. The dataset emits a placeholder that is only meaningful with offset ids
    # off; refuse the combination rather than train on quietly wrong positions.
    if args.dataset_mode == "window" and args.p_fim > 0 and not args.no_offset_id:
        raise ValueError(
            "window FIM requires --no-offset-id: a multi-NAL hole has no single "
            "within-NAL offset. Phase 1 already runs --no-region-id --no-offset-id "
            "(AVC-LM-faithful), so a phase-1-comparable FIM run wants both."
        )
    # The reconstruction probe is slice-only: it reads sample.target_index (a
    # SliceSample field WindowSample does not have) and derives the replacement span
    # from slice-mode offset ids, which window FIM does not carry. Fail at startup
    # rather than AttributeError hours into a run.
    if args.dataset_mode == "window" and args.reconstruction_eval_interval > 0:
        raise ValueError(
            "--reconstruction-eval-interval is not supported in window mode "
            "(litgpt/byte/reconstruction.py assumes a single target NAL per sample). "
            "Use --free-run-eval-interval for the AR probe; window-FIM reconstruction "
            "needs the probe ported first."
        )
    # A100 Tensor Cores accelerate float32 matmuls used outside bf16 AMP regions.
    torch.set_float32_matmul_precision("high")

    use_region_id = not args.no_region_id
    use_offset_id = not args.no_offset_id
    max_tokens = args.steps * args.global_batch_size * args.block_size

    model_config = Config(
        name=args.model_name,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        vocab_size=vocab_size_for_fim_format(args.fim_format, args.use_eos),
        padding_multiple=8,
        use_region_id=use_region_id,
        use_offset_id=use_offset_id,
        offset_vocab_size=(
            args.block_size * args.byte_patch_size
            if use_offset_id and args.byte_patch_size > 1
            else None
        ),
        byte_patch_size=args.byte_patch_size,
        megabyte_local_n_layer=args.megabyte_local_layers,
        megabyte_local_n_embd=args.megabyte_local_embd,
        megabyte_local_n_head=args.megabyte_local_heads,
    )
    data_config = ByteDataConfig(
        byte_patch_size=args.byte_patch_size,
        p_fim=args.p_fim,
        fixed_fim_holes=args.fixed_fim_holes,
        fixed_fim_holes_per_window=args.fixed_fim_holes_per_window,
        fim_format=args.fim_format,
        fim_loss_scope=args.fim_loss_scope,
        use_eos=args.use_eos,
        num_ref_slices=args.num_ref_slices,
        reference_mode=args.reference_mode,
        target_nal_types=tuple(args.target_nal_types),
        fim_min_gap=args.fim_min_gap,
        fim_max_gap=args.fim_max_gap,
        slice_header_guard_bytes=args.slice_header_guard_bytes,
        num_workers=args.num_workers,
        condition_on_sps_pps=not args.no_sps_pps_conditioning,
        default_max_seq_length=args.block_size,
        val_fraction=args.val_fraction,
        split_by_video=args.split_by_video,
        dataset_mode=args.dataset_mode,
        window_min_frames=args.window_min_frames,
    )
    max_manifest_rows = None if args.max_manifest_rows == 0 else args.max_manifest_rows
    data = ByteDataModule(
        manifest_path=args.manifest,
        config=data_config,
        max_manifest_rows=max_manifest_rows,
        nal_index_path=args.nal_index_path,
        # Record the exact train split so a later training-set eval can target precisely
        # what was trained (--train-split-file in eval_stream_continuation.py).
        train_split_dump_path=args.out_dir / "train_split.json",
    )
    train = TrainArgs(
        save_interval=args.save_interval,
        log_interval=1,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_tokens=max_tokens,
        max_norm=1.0,
        min_lr=args.min_learning_rate,
        lr_warmup_steps=args.warmup_steps,
        tie_embeddings=False,
    )
    eval_args = EvalArgs(
        interval=args.eval_interval,
        max_iters=args.eval_iters,
        initial_validation=True,
        final_validation=True,
    )
    optimizer = {
        "class_path": "torch.optim.AdamW",
        "init_args": {
            "lr": args.learning_rate,
            "weight_decay": 0.1,
            "betas": (0.9, 0.95),
        },
    }
    reconstruction_eval = (
        ReconstructionEvalConfig(
            interval=args.reconstruction_eval_interval,
            num_samples=args.reconstruction_eval_samples,
            timeout_sec=args.reconstruction_timeout_sec,
            max_target_bytes=args.reconstruction_max_target_bytes,
            ffmpeg_binary=args.ffmpeg_binary,
            task=args.reconstruction_task,
            evaluate_oracle_length=args.reconstruction_oracle_length,
            evaluate_error_exploding=args.reconstruction_error_exploding,
            evaluate_fim_baselines=args.reconstruction_fim_baselines,
            learned_eos_stopping=args.reconstruction_learned_eos,
            num_visualization_samples=args.reconstruction_visualization_samples,
        )
        if args.reconstruction_eval_interval > 0
        else None
    )
    free_run_eval = (
        FreeRunEvalConfig(
            interval=args.free_run_eval_interval,
            num_clips=args.free_run_eval_clips,
            prefix_frames=args.free_run_prefix_frames,
            cont_frames=args.free_run_cont_frames,
            temperature=args.free_run_temperature,
            max_gen_multiple=args.free_run_max_gen_multiple,
            seed=args.seed,
            slice_layout=args.free_run_slice_layout,
        )
        if args.free_run_eval_interval > 0
        else None
    )
    mrt = MRTConfig(
        interval=args.mrt_interval,
        start_step=args.mrt_start_step,
        num_candidates=args.mrt_num_candidates,
        context_pool_size=args.mrt_context_pool_size,
        max_target_bytes=args.mrt_max_target_bytes,
        oracle_length=args.mrt_oracle_length,
        learned_eos=args.mrt_learned_eos,
        include_ground_truth=args.mrt_include_ground_truth,
        temperature=args.mrt_temperature,
        candidate_alpha=args.mrt_candidate_alpha,
        weight=args.mrt_weight,
        risk_mode=args.mrt_risk_mode,
        mse_weight=args.mrt_mse_weight,
        mse_tau=args.mrt_mse_tau,
        decode_failure_weight=args.mrt_decode_failure_weight,
        max_risk=args.mrt_max_risk,
        timeout_sec=args.reconstruction_timeout_sec,
        decode_workers=args.mrt_decode_workers,
        ffmpeg_binary=args.ffmpeg_binary,
    )
    if mrt.enabled:
        mrt.validate()

    grpo = GRPOConfig(
        interval=args.grpo_interval,
        start_step=args.grpo_start_step,
        group_size=args.grpo_group_size,
        ar_pool_size=args.grpo_ar_pool_size,
        fim_pool_size=args.grpo_fim_pool_size,
        max_target_bytes=args.grpo_max_target_bytes,
        temperature=args.grpo_temperature,
        top_k=args.grpo_top_k,
        top_p=args.grpo_top_p,
        kl_coeff=args.grpo_kl_coeff,
        psnr_cap=args.grpo_psnr_cap,
        decode_failure_reward=args.grpo_decode_failure_reward,
        mu=args.grpo_mu,
        clip_range=args.grpo_clip_range,
        learned_eos=args.grpo_learned_eos,
        generation_budget_multiplier=args.grpo_generation_budget_multiplier,
        timeout_sec=args.grpo_timeout_sec,
        decode_workers=args.grpo_decode_workers,
        ffmpeg_binary=args.grpo_ffmpeg_binary,
    )
    if grpo.enabled:
        grpo.validate()

    setup(
        model_name=args.model_name,
        model_config=model_config,
        out_dir=args.out_dir,
        precision=args.precision,
        resume="auto" if args.resume else False,
        initial_checkpoint_dir=args.initial_checkpoint_dir,
        data=data,
        train=train,
        eval=eval_args,
        optimizer=optimizer,
        devices=args.devices,
        num_nodes=args.num_nodes,
        tokenizer_dir=None,
        logger_name=args.logger_name,
        seed=args.seed,
        compile_model=args.compile,
        reconstruction_eval=reconstruction_eval,
        ce_loss_weight=args.ce_loss_weight,
        ce_byte_only=args.ce_byte_only,
        eos_loss_weight=args.eos_loss_weight,
        eos_aux_loss_weight=args.eos_aux_loss_weight,
        mrt=mrt,
        free_run_eval=free_run_eval,
        grpo=grpo,
        grpo_reference_checkpoint_dir=args.grpo_reference_checkpoint_dir,
    )


if __name__ == "__main__":
    main()
