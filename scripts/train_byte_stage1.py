"""Launch stage-1 H.264 byte-domain pretraining.

This launcher keeps the byte-specific model and data settings explicit while
reusing LitGPT's pretraining loop, checkpointing, logging, and distributed setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import Config
from litgpt.data.byte_data import (
    FIM_FORMATS,
    REFERENCE_MODES,
    ByteDataConfig,
    ByteDataModule,
    vocab_size_for_fim_format,
)
from litgpt.eval.byte_reconstruction import ReconstructionEvalConfig
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
            "Model context length. The preliminary 5,940-sample scan retained "
            "full metadata+reference+target context for 93.42%% of samples and "
            "excluded 0%% of targets at 16K."
        ),
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
        "--reconstruction-task",
        choices=("ar", "fim", "both"),
        default="ar",
        help="Evaluate AR slices, FIM missing spans, or both decoder tasks.",
    )
    parser.add_argument("--reconstruction-timeout-sec", type=int, default=30)
    parser.add_argument("--reconstruction-max-target-bytes", type=int, default=2048)
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
    parser.add_argument("--p-fim", type=float, default=0.0)
    parser.add_argument("--fim-format", choices=FIM_FORMATS, default="bridge")
    parser.add_argument(
        "--use-eos",
        action="store_true",
        help="Append SEQ_EOS after each AR target / FIM span so the model learns "
        "to terminate. Default off uses oracle lengths for generation.",
    )
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument("--no-sps-pps-conditioning", action="store_true")
    parser.add_argument("--no-region-id", action="store_true")
    parser.add_argument("--no-offset-id", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--logger-name", default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    )
    data_config = ByteDataConfig(
        p_fim=args.p_fim,
        fim_format=args.fim_format,
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
    )
    max_manifest_rows = None if args.max_manifest_rows == 0 else args.max_manifest_rows
    data = ByteDataModule(
        manifest_path=args.manifest,
        config=data_config,
        max_manifest_rows=max_manifest_rows,
        nal_index_path=args.nal_index_path,
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
        )
        if args.reconstruction_eval_interval > 0
        else None
    )

    setup(
        model_name=args.model_name,
        model_config=model_config,
        out_dir=args.out_dir,
        precision=args.precision,
        resume="auto" if args.resume else False,
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
    )


if __name__ == "__main__":
    main()
