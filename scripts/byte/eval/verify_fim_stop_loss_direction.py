"""Verify one parser-derived FIM stop/no-stop update in memory.

The script finds one on-policy rollout endpoint that legally reconnects to the
fixed suffix, pairs it with nearby earlier invalid stopping points, applies one
stop-loss-only SGD update without saving the model, and verifies the expected
EOS probability direction at every state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte.grpo import (  # noqa: E402
    build_fim_stop_patch_inputs,
    build_group_patch_inputs,
    fim_ground_truth_reconnects,
    fim_stop_loss,
    fim_stop_probabilities_and_ranks,
    fim_stop_states,
    group_token_log_probabilities,
    grpo_clipped_loss,
    prepare_grpo_step,
)
from litgpt.byte.reconstruction import _unwrap_model  # noqa: E402
from scripts.byte.eval.eval_grpo_fixed_reward import (  # noqa: E402
    _build_sampler,
    _grpo_config,
)
from scripts.byte.eval.helpers.checkpoint_eval_helpers import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument("--context-seed", type=int, default=42)
    parser.add_argument("--max-context-draws", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--stop-negative-samples", type=int, default=4)
    parser.add_argument("--max-manifest-rows", type=int, default=45000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument("--max-target-bytes", type=int, default=1400)
    parser.add_argument("--window-min-frames", type=int, default=2)
    parser.add_argument("--fim-slice-layout", choices=("macroblock", "frame"), default="macroblock")
    parser.add_argument("--ar-prefix-frames", type=int, default=4)
    parser.add_argument("--ar-cont-frames", type=int, default=2)
    parser.add_argument("--ar-slice-layout", default="macroblock")
    parser.add_argument("--generation-budget-multiplier", type=float, default=2.0)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    args = parser.parse_args()
    if args.learning_rate <= 0 or args.stop_loss_weight <= 0:
        parser.error("learning rate and stop-loss weight must be positive")
    return args


def _gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return math.sqrt(total)


def _eos_scores(model, inputs, labels, supervised) -> tuple[torch.Tensor, torch.Tensor]:
    return fim_stop_probabilities_and_ranks(model, inputs, labels, supervised)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = _grpo_config(args)
    model = load_model(args.checkpoint_dir, device)
    model.eval()
    sampler = _build_sampler(args, model, config)

    prepared = None
    states = []
    selected = None
    fallback = None
    gt_sanity_failures = 0
    for draw_index in range(args.max_context_draws):
        selection = sampler.sample("fim", draw_index)
        if selection is None:
            continue
        effective_seed = args.rollout_seed + draw_index * 1_000_003
        torch.manual_seed(effective_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(effective_seed)
        result = prepare_grpo_step(model, selection.sample, config, device)
        if result.prepared is None:
            continue
        if not fim_ground_truth_reconnects(
            result.prepared.sample, slice_layout=args.fim_slice_layout
        ):
            gt_sanity_failures += 1
            continue
        candidate_states = fim_stop_states(
            result.prepared.sample,
            result.prepared.candidates,
            slice_layout=args.fim_slice_layout,
            negative_samples=args.stop_negative_samples,
            max_positive_states=1,
        )
        if candidate_states:
            current = (result.prepared, candidate_states, selection)
            if fallback is None:
                fallback = current
            # Prefer a group that calibrates stop-gradient scale against a
            # nonzero GRPO gradient. The parser-direction test itself can still
            # use the first certified fallback if every such group is degenerate.
            if result.has_policy_signal:
                prepared, states, selected = current
                break
    if prepared is None and fallback is not None:
        prepared, states, selected = fallback
    if prepared is None or selected is None:
        raise RuntimeError(
            "No parser-certified stop state found; increase --max-context-draws "
            "or inspect parser coverage before training"
        )

    patch_size = int(_unwrap_model(model).config.byte_patch_size)
    stop_inputs, stop_labels, stop_supervised, should_stop = (
        build_fim_stop_patch_inputs(
            prepared.sample, states, patch_size, device
        )
    )

    # Report an unweighted on-policy GRPO gradient norm from the same group as
    # a practical scale reference for lambda_stop. Zero is allowed when this
    # particular group has no reward variance; the direction check remains valid.
    policy_inputs, policy_labels, policy_supervised = build_group_patch_inputs(
        prepared.sample,
        prepared.candidates,
        patch_size,
        device,
        append_eos_on_stop=True,
    )
    model.zero_grad(set_to_none=True)
    policy_gathered = group_token_log_probabilities(
        model, policy_inputs, policy_labels, include_eos=True
    )
    old_policy_gathered = policy_gathered.detach()
    policy_loss, _ = grpo_clipped_loss(
        policy_gathered,
        old_policy_gathered,
        policy_supervised,
        prepared.advantages.clone(),
        None,
        0.0,
        0.2,
    )
    policy_loss.backward()
    policy_gradient_norm = _gradient_norm(model)
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        before, ranks_before = _eos_scores(
            model, stop_inputs, stop_labels, stop_supervised
        )
        before = before.detach().float()
        ranks_before = ranks_before.detach().cpu()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    gathered = group_token_log_probabilities(
        model, stop_inputs, stop_labels, include_eos=True
    )
    stop_loss_value, stop_metrics = fim_stop_loss(
        gathered, stop_labels, stop_supervised, should_stop
    )
    weighted_loss = args.stop_loss_weight * stop_loss_value
    optimizer.zero_grad(set_to_none=True)
    weighted_loss.backward()
    stop_gradient_norm = _gradient_norm(model)
    finite_gradient = math.isfinite(stop_gradient_norm) and stop_gradient_norm > 0
    optimizer.step()
    model.eval()
    with torch.no_grad():
        after, ranks_after = _eos_scores(
            model, stop_inputs, stop_labels, stop_supervised
        )
        after = after.detach().float()
        ranks_after = ranks_after.detach().cpu()

    positive_before = float(before[should_stop].mean())
    negative_before = float(before[~should_stop].mean())
    positive_after = float(after[should_stop].mean())
    negative_after = float(after[~should_stop].mean())
    positive_improved_fraction = float(
        (after[should_stop] > before[should_stop]).float().mean()
    )
    negative_improved_fraction = float(
        (after[~should_stop] < before[~should_stop]).float().mean()
    )
    margin_before = positive_before - negative_before
    margin_after = positive_after - negative_after
    after_positive_loss = -torch.log(
        after[should_stop].clamp_min(torch.finfo(after.dtype).tiny)
    ).mean()
    after_negative_loss = -torch.log1p(
        -after[~should_stop].clamp(max=1.0 - torch.finfo(after.dtype).eps)
    ).mean()
    stop_loss_after = float(0.5 * (after_positive_loss + after_negative_loss))
    stop_loss_before = float(stop_loss_value.detach())
    passed = bool(
        finite_gradient
        and positive_after > positive_before
        and negative_after < negative_before
        and margin_after > margin_before
        and stop_loss_after < stop_loss_before
    )

    state_rows = []
    for state, before_probability, after_probability, rank_before, rank_after in zip(
        states, before, after, ranks_before, ranks_after
    ):
        state_rows.append(
            {
                "candidate_index": state.candidate_index,
                "prefix_length": state.prefix_length,
                "should_stop": state.should_stop,
                "eos_probability_before": float(before_probability),
                "eos_probability_after": float(after_probability),
                "delta": float(after_probability - before_probability),
                "eos_rank_before": int(rank_before),
                "eos_rank_after": int(rank_after),
            }
        )
    report = {
        "result": "PASS" if passed else "FAIL",
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_index": selected.dataset_index,
        "draw_index": selected.draw_index,
        "hole_spec": selected.hole_spec,
        "h264_path": str(selected.sample.h264_path),
        "rollout_seed": args.rollout_seed,
        "ground_truth_reconnection_sanity": True,
        "ground_truth_sanity_failures_before_selection": gt_sanity_failures,
        "learning_rate": args.learning_rate,
        "stop_loss_weight": args.stop_loss_weight,
        "stop_loss_before": stop_loss_before,
        "stop_loss_after": stop_loss_after,
        "stop_gradient_norm": stop_gradient_norm,
        "policy_gradient_norm_same_group": policy_gradient_norm,
        "weighted_stop_to_unweighted_same_group_policy_gradient_ratio": (
            None
            if policy_gradient_norm == 0
            else stop_gradient_norm / policy_gradient_norm
        ),
        "calibration_scope": (
            "Heuristic scale comparison against the unweighted, no-KL GRPO "
            "policy gradient from this selected group; not the total CE+GRPO "
            "training-gradient norm"
        ),
        "estimated_sgd_parameter_update_norm": args.learning_rate
        * stop_gradient_norm,
        "positive_probability_before": positive_before,
        "positive_probability_after": positive_after,
        "negative_probability_before": negative_before,
        "negative_probability_after": negative_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "positive_improved_fraction": positive_improved_fraction,
        "negative_improved_fraction": negative_improved_fraction,
        "gradient_finite_nonzero": finite_gradient,
        "states": state_rows,
        "loss_metrics_before": stop_metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n=== FIM stop/no-stop one-update direction check ===")
    print(f"positive EOS: {positive_before:.8f} -> {positive_after:.8f}")
    print(f"negative EOS: {negative_before:.8f} -> {negative_after:.8f}")
    print(f"margin:       {margin_before:+.8f} -> {margin_after:+.8f}")
    print(f"gradient:     {stop_gradient_norm:.6e}")
    print(f"result:       {'PASS' if passed else 'FAIL'}")
    print(f"report:       {args.out}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
