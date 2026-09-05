"""Verify that one GRPO update moves probability toward higher reward.

This is a checkpoint-backed wiring diagnostic, not a training run. It uses the
production online context sampler, MEGABYTE rollout, decoder reward, candidate
batch builder, scorer, and GRPO loss. The model is modified only in memory and
no checkpoint is saved.

The diagnostic intentionally samples from the full policy distribution
(``top_k=0``, ``top_p=1``), so the behavior distribution and the log-probability
distribution used by GRPO are identical.

Example:
    python scripts/byte/eval/verify_grpo_update_direction.py \
        DATA/manifest.jsonl \
        --nal-index-path DATA/nal_index.sqlite \
        --checkpoint-dir RUN/step-00057000 \
        --task ar \
        --out-json RUN/grpo_direction_ar.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte.data import WINDOW_UNITS, ByteDataConfig, ByteDataModule  # noqa: E402
from litgpt.byte.grpo import (  # noqa: E402
    GRPOConfig,
    build_group_patch_inputs,
    grpo_clipped_loss,
    grpo_update_direction_metrics,
    group_token_log_probabilities,
    mean_log_probability,
    prepare_grpo_step,
    prepare_grpo_step_ar,
)
from litgpt.byte.grpo_context import OnlineGRPOContextSampler  # noqa: E402
from litgpt.byte.reconstruction import _unwrap_model  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("ar", "fim"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-manifest-rows", type=int, default=45000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draw-index", type=int, default=0)
    parser.add_argument(
        "--max-group-attempts",
        type=int,
        default=32,
        help="Try later deterministic contexts when a group has no reward variance.",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-score-gain", type=float, default=0.0)
    parser.add_argument("--min-pairwise-improved", type=float, default=0.5)
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument("--max-target-bytes", type=int, default=1400)
    parser.add_argument("--window-min-frames", type=int, default=2)
    parser.add_argument("--window-unit", choices=WINDOW_UNITS, default="byte_budget")
    parser.add_argument("--ar-prefix-frames", type=int, default=4)
    parser.add_argument("--ar-cont-frames", type=int, default=2)
    parser.add_argument("--ar-slice-layout", default="macroblock")
    parser.add_argument("--generation-budget-multiplier", type=float, default=2.0)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    args = parser.parse_args()
    if args.max_manifest_rows <= 0:
        parser.error("--max-manifest-rows must be positive")
    if args.max_group_attempts <= 0:
        parser.error("--max-group-attempts must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.min_pairwise_improved <= 1:
        parser.error("--min-pairwise-improved must be in [0, 1]")
    return args


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _gradient_l2_norm(model: nn.Module) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


def _true_ranges(mask: Tensor) -> list[list[int]]:
    """Compactly preserve one flattened scored-token mask as [start, end) ranges."""
    indices = mask.flatten().nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            ranges.append([start, previous + 1])
            start = value
        previous = value
    ranges.append([start, previous + 1])
    return ranges


def _build_sampler(
    args: argparse.Namespace,
    model: nn.Module,
    config: GRPOConfig,
) -> OnlineGRPOContextSampler:
    raw_model = _unwrap_model(model)
    patch_size = int(raw_model.config.byte_patch_size)
    if patch_size <= 1:
        raise ValueError("This diagnostic currently targets MEGABYTE checkpoints")
    data_config = ByteDataConfig(
        byte_patch_size=patch_size,
        p_fim=0.5,
        fim_format="psm",
        fim_loss_scope="full",
        use_eos=True,
        val_fraction=args.val_fraction,
        split_by_video=False,
        seed=args.seed,
        num_workers=1,
        fim_min_gap=args.fim_min_gap,
        fim_max_gap=args.fim_max_gap,
        slice_header_guard_bytes=args.slice_header_guard_bytes,
        condition_on_sps_pps=True,
        default_max_seq_length=int(raw_model.max_seq_length),
        dataset_mode="window",
        window_min_frames=args.window_min_frames,
        window_unit=args.window_unit,
    )
    data = ByteDataModule(
        manifest_path=args.manifest,
        config=data_config,
        max_manifest_rows=args.max_manifest_rows,
        nal_index_path=args.nal_index_path,
    )
    data.connect(
        tokenizer=None,
        batch_size=1,
        max_seq_length=int(raw_model.max_seq_length),
    )
    print("Building the deterministic training split...", flush=True)
    data.setup()
    if data.train_dataset is None:
        raise RuntimeError("ByteDataModule did not produce a training dataset")
    return OnlineGRPOContextSampler.from_dataset(data.train_dataset, config)


def _prepare_non_degenerate_group(
    args: argparse.Namespace,
    model: nn.Module,
    sampler: OnlineGRPOContextSampler,
    config: GRPOConfig,
    device: torch.device,
):
    for draw_index in range(
        args.draw_index, args.draw_index + args.max_group_attempts
    ):
        selection = sampler.sample(args.task, draw_index)
        if selection is None:
            print(f"draw={draw_index}: no eligible context", flush=True)
            continue
        if args.task == "ar":
            result = prepare_grpo_step_ar(
                model, selection.sample, config, device
            )
        else:
            result = prepare_grpo_step(model, selection.sample, config, device)
        print(
            f"draw={draw_index} dataset_index={selection.dataset_index} "
            f"status={result.status} candidates={result.candidate_count} "
            f"decoded={result.decoded_count} reward_std={result.reward_std}",
            flush=True,
        )
        if result.prepared is not None and result.has_policy_signal:
            return selection, result.prepared
    raise RuntimeError(
        "No group with nonzero reward variance was found; increase "
        "--max-group-attempts or --group-size"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading policy from {args.checkpoint_dir}", flush=True)
    model = load_model(args.checkpoint_dir, device)
    model.eval()  # Keep scoring deterministic; eval mode still permits gradients.
    raw_model = _unwrap_model(model)
    patch_size = int(raw_model.config.byte_patch_size)

    config = GRPOConfig(
        interval=1,
        group_size=args.group_size,
        context_sampling="online",
        context_seed=args.seed,
        max_target_bytes=args.max_target_bytes,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        kl_coeff=0.0,
        timeout_sec=args.timeout_sec,
        decode_workers=args.decode_workers,
        ffmpeg_binary=args.ffmpeg_binary,
        mu=1,
        learned_eos=True,
        generation_budget_multiplier=args.generation_budget_multiplier,
        ar_prefix_frames=args.ar_prefix_frames,
        ar_cont_frames=args.ar_cont_frames,
        ar_slice_layout=args.ar_slice_layout,
    )
    config.validate()
    sampler = _build_sampler(args, model, config)
    selection, prepared = _prepare_non_degenerate_group(
        args, model, sampler, config, device
    )

    # prepare_grpo_step[_ar] is inference-mode code. Clone its inference tensors
    # after returning before they participate in an autograd expression.
    rewards = prepared.rewards.clone().detach()
    advantages = prepared.advantages.clone().detach()
    score_eos = args.task == "fim"
    inputs, labels, scored_token_mask = build_group_patch_inputs(
        prepared.sample,
        prepared.candidates,
        patch_size,
        device,
        append_eos_on_stop=score_eos,
    )

    with torch.no_grad(), _autocast(device):
        old_gathered = group_token_log_probabilities(
            model, inputs, labels, include_eos=score_eos
        ).detach()
        before_log_probs = mean_log_probability(
            old_gathered, scored_token_mask
        ).detach()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device):
        gathered = group_token_log_probabilities(
            model, inputs, labels, include_eos=score_eos
        )
        loss, loss_metrics = grpo_clipped_loss(
            gathered,
            old_gathered,
            scored_token_mask,
            advantages,
            reference_gathered=None,
            kl_coeff=0.0,
            clip_range=0.2,
        )
    loss.backward()
    grad_norm_pre_clip = _gradient_l2_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    grad_norm_post_clip = _gradient_l2_norm(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del gathered, loss

    with torch.no_grad(), _autocast(device):
        after_gathered = group_token_log_probabilities(
            model, inputs, labels, include_eos=score_eos
        )
        after_log_probs = mean_log_probability(
            after_gathered, scored_token_mask
        ).detach()

    direction = grpo_update_direction_metrics(
        before_log_probs, after_log_probs, rewards, advantages
    )
    finite_grad = math.isfinite(grad_norm_pre_clip) and grad_norm_pre_clip > 0
    passed = (
        finite_grad
        and direction["policy_score_delta"] > args.min_score_gain
        and direction["pairwise_improved_fraction"]
        >= args.min_pairwise_improved
    )

    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(prepared.candidates):
        candidates.append(
            {
                "index": index,
                "bytes_hex": candidate.data.hex(),
                "num_bytes": len(candidate.data),
                "stopped": candidate.stopped,
                "reward": float(rewards[index]),
                "advantage": float(advantages[index]),
                "decoded": bool(prepared.decoded[index]),
                "psnr": (
                    None
                    if bool(torch.isnan(prepared.psnrs[index]))
                    else float(prepared.psnrs[index])
                ),
                "scored_token_count": int(scored_token_mask[index].sum()),
                "scored_token_mask_shape": list(scored_token_mask[index].shape),
                "scored_token_mask_true_ranges": _true_ranges(
                    scored_token_mask[index]
                ),
                "mean_log_probability_before": float(before_log_probs[index]),
                "mean_log_probability_after": float(after_log_probs[index]),
                "mean_log_probability_delta": float(
                    after_log_probs[index] - before_log_probs[index]
                ),
            }
        )

    report = {
        "passed": passed,
        "task": args.task,
        "checkpoint_dir": str(args.checkpoint_dir),
        "manifest": str(args.manifest),
        "dataset_index": selection.dataset_index,
        "draw_index": selection.draw_index,
        "selection_attempt": selection.attempt,
        "hole_spec": selection.hole_spec,
        "h264_path": str(prepared.sample.h264_path),
        "group_size": len(prepared.candidates),
        "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "max_grad_norm": args.max_grad_norm,
        },
        "loss_before_update": float(loss_metrics["grpo/policy_loss"]),
        "ratio_mean_before_update": float(loss_metrics["grpo/ratio_mean"]),
        "grad_norm_pre_clip": grad_norm_pre_clip,
        "grad_norm_post_clip": grad_norm_post_clip,
        "direction": direction,
        "thresholds": {
            "min_score_gain": args.min_score_gain,
            "min_pairwise_improved": args.min_pairwise_improved,
        },
        "candidates": candidates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== One-update GRPO direction check ===", flush=True)
    print(f"task: {args.task}", flush=True)
    print(f"gradient: {grad_norm_pre_clip:.6e} -> {grad_norm_post_clip:.6e}", flush=True)
    print(
        "reward-weighted policy score: "
        f"{direction['policy_score_before']:.8f} -> "
        f"{direction['policy_score_after']:.8f} "
        f"(delta={direction['policy_score_delta']:+.8e})",
        flush=True,
    )
    print(
        "pairwise improved: "
        f"{int(direction['pairwise_improved'])}/"
        f"{int(direction['pairwise_comparable'])} "
        f"({direction['pairwise_improved_fraction']:.1%})",
        flush=True,
    )
    print(f"result: {'PASS' if passed else 'FAIL'}", flush=True)
    print(f"report: {args.out_json}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
