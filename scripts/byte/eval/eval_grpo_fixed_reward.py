"""Compare implemented GRPO reward across checkpoints on fixed contexts.

This is Experiment 1B: it asks whether several GRPO updates produce a policy
with higher expected *implemented* reward. Every checkpoint sees the same AR
and FIM contexts and the same rollout seeds. Each context/seed draws a candidate
group. Rollout seeds are repeated measurements within a context; confidence
intervals are computed across the frozen contexts, not across candidates or
context-seed pairs.

This evaluator does not test whether the reward agrees with the final AR/FIM
evaluation; that is a separate candidate-level alignment audit (Experiment 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from litgpt.byte.data import ByteDataConfig, ByteDataModule  # noqa: E402
from litgpt.byte.grpo import (  # noqa: E402
    FIMStopState,
    GRPOConfig,
    PreparedGRPOStep,
    build_fim_stop_patch_inputs,
    fim_endpoint_reconnects,
    fim_ground_truth_reconnects,
    fim_stop_probabilities_and_ranks,
    fim_stop_states,
    prepare_grpo_step,
    prepare_grpo_step_ar,
)
from litgpt.byte.grpo_context import (  # noqa: E402
    GRPOContextSelection,
    OnlineGRPOContextSampler,
)
from litgpt.byte.reconstruction import _unwrap_model  # noqa: E402
from scripts.byte.eval.helpers.checkpoint_eval_helpers import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--nal-index-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", nargs="+", choices=("ar", "fim"), default=["ar", "fim"])
    parser.add_argument("--num-contexts", type=int, default=8)
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-manifest-rows", type=int, default=45000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--context-seed", type=int, default=42)
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--slice-header-guard-bytes", type=int, default=64)
    parser.add_argument("--max-target-bytes", type=int, default=1400)
    parser.add_argument("--window-min-frames", type=int, default=2)
    parser.add_argument("--ar-prefix-frames", type=int, default=4)
    parser.add_argument("--ar-cont-frames", type=int, default=2)
    parser.add_argument("--ar-slice-layout", default="macroblock")
    parser.add_argument(
        "--fim-slice-layout", choices=("macroblock", "frame"), default="macroblock"
    )
    parser.add_argument("--stop-negative-samples", type=int, default=4)
    parser.add_argument("--stop-max-positive-states", type=int, default=4)
    parser.add_argument("--generation-budget-multiplier", type=float, default=2.0)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [path.name for path in args.checkpoint_dirs]
    if len(args.labels) != len(args.checkpoint_dirs):
        parser.error("--labels must contain one label per --checkpoint-dirs entry")
    if len(set(args.labels)) != len(args.labels):
        parser.error("--labels must be unique")
    if args.num_contexts <= 0 or args.group_size < 2:
        parser.error("--num-contexts must be positive and --group-size at least 2")
    if not args.rollout_seeds:
        parser.error("--rollout-seeds must not be empty")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("--top-k must be non-negative and --top-p in (0, 1]")
    return args


def _grpo_config(args: argparse.Namespace) -> GRPOConfig:
    config = GRPOConfig(
        interval=1,
        group_size=args.group_size,
        context_sampling="online",
        context_seed=args.context_seed,
        max_target_bytes=args.max_target_bytes,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
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
        fim_slice_layout=args.fim_slice_layout,
    )
    config.validate()
    return config


def _build_sampler(
    args: argparse.Namespace,
    model: nn.Module,
    config: GRPOConfig,
) -> OnlineGRPOContextSampler:
    raw_model = _unwrap_model(model)
    patch_size = int(raw_model.config.byte_patch_size)
    if patch_size <= 1:
        raise ValueError("This evaluator currently targets MEGABYTE checkpoints")
    data_config = ByteDataConfig(
        byte_patch_size=patch_size,
        p_fim=0.5,
        fim_format="psm",
        fim_loss_scope="full",
        use_eos=True,
        val_fraction=args.val_fraction,
        split_by_video=False,
        seed=args.context_seed,
        num_workers=1,
        fim_min_gap=args.fim_min_gap,
        fim_max_gap=args.fim_max_gap,
        slice_header_guard_bytes=args.slice_header_guard_bytes,
        condition_on_sps_pps=True,
        default_max_seq_length=int(raw_model.max_seq_length),
        dataset_mode="window",
        window_min_frames=args.window_min_frames,
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


def _fixed_contexts(
    sampler: OnlineGRPOContextSampler,
    tasks: list[str],
    num_contexts: int,
) -> dict[str, list[GRPOContextSelection]]:
    selected: dict[str, list[GRPOContextSelection]] = {}
    for task in tasks:
        contexts: list[GRPOContextSelection] = []
        draw_index = 0
        max_draws = max(32, num_contexts * 32)
        while len(contexts) < num_contexts and draw_index < max_draws:
            selection = sampler.sample(task, draw_index)
            draw_index += 1
            if selection is not None:
                contexts.append(selection)
        if len(contexts) != num_contexts:
            raise RuntimeError(
                f"Only found {len(contexts)}/{num_contexts} eligible {task} contexts"
            )
        selected[task] = contexts
        print(f"Fixed {len(contexts)} {task.upper()} contexts", flush=True)
    return selected


def _prepare(
    task: str,
    model: nn.Module,
    selection: GRPOContextSelection,
    config: GRPOConfig,
    device: torch.device,
):
    if task == "ar":
        return prepare_grpo_step_ar(model, selection.sample, config, device)
    return prepare_grpo_step(model, selection.sample, config, device)


def _candidate_rows(prepared: PreparedGRPOStep) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(prepared.candidates):
        psnr = prepared.psnrs[index]
        rows.append(
            {
                "candidate_index": index,
                "bytes_hex": candidate.data.hex(),
                "bytes_sha256": hashlib.sha256(candidate.data).hexdigest(),
                "num_bytes": len(candidate.data),
                "stopped": candidate.stopped,
                "reward": float(prepared.rewards[index]),
                "decoded": bool(prepared.decoded[index]),
                "psnr": None if bool(torch.isnan(psnr)) else float(psnr),
            }
        )
    return rows


@torch.inference_mode()
def _stop_state_rows(
    model: nn.Module,
    sample,
    states: list[FIMStopState],
    device: torch.device,
) -> list[dict[str, Any]]:
    if not states:
        return []
    patch_size = int(_unwrap_model(model).config.byte_patch_size)
    inputs, labels, supervised, _ = build_fim_stop_patch_inputs(
        sample, states, patch_size, device
    )
    probabilities, ranks = fim_stop_probabilities_and_ranks(
        model, inputs, labels, supervised
    )
    return [
        {
            "candidate_index": state.candidate_index,
            "prefix_length": state.prefix_length,
            "should_stop": state.should_stop,
            "eos_probability": float(probability),
            "eos_rank": int(rank),
        }
        for state, probability, rank in zip(states, probabilities, ranks)
    ]


def _mean_ci95(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    center = statistics.mean(values)
    if len(values) == 1:
        return center, center, center
    # Context counts are intentionally small (normally eight), so a normal
    # 1.96 multiplier would understate uncertainty. Two-sided Student-t 95%
    # critical values for df=1..30; the asymptotic value is sufficient above.
    t95 = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
        2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
        2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
        2.048, 2.045, 2.042,
    )
    degrees_of_freedom = len(values) - 1
    critical = t95[degrees_of_freedom - 1] if degrees_of_freedom <= 30 else 1.96
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return center, center - half_width, center + half_width


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["candidate_count"] > 0]
    rewards = [
        candidate["reward"]
        for row in available
        for candidate in row["candidates"]
    ]
    rewards_by_context: dict[int, list[float]] = {}
    for row in available:
        rewards_by_context.setdefault(int(row["context_index"]), []).extend(
            float(candidate["reward"]) for candidate in row["candidates"]
        )
    context_reward_means = [
        statistics.mean(context_rewards)
        for context_rewards in rewards_by_context.values()
    ]
    reward_mean, reward_ci_low, reward_ci_high = _mean_ci95(context_reward_means)
    candidates = [candidate for row in available for candidate in row["candidates"]]
    psnrs = [candidate["psnr"] for candidate in candidates if candidate["psnr"] is not None]
    stop_states = [state for row in available for state in row.get("stop_states", [])]
    positive_stop_probabilities = [
        state["eos_probability"] for state in stop_states if state["should_stop"]
    ]
    negative_stop_probabilities = [
        state["eos_probability"] for state in stop_states if not state["should_stop"]
    ]
    positive_stop_ranks = [
        state["eos_rank"] for state in stop_states if state["should_stop"]
    ]
    negative_stop_ranks = [
        state["eos_rank"] for state in stop_states if not state["should_stop"]
    ]
    reconnectable = [
        candidate["suffix_reconnectable"]
        for candidate in candidates
        if candidate.get("suffix_reconnectable") is not None
    ]
    has_reconnection_labels = bool(reconnectable)
    generated_lengths = [candidate["num_bytes"] for candidate in candidates]
    valid_stops = [
        candidate
        for candidate in candidates
        if candidate.get("stopped")
        and candidate.get("suffix_reconnectable") is True
    ]
    premature_stops = [
        candidate
        for candidate in candidates
        if candidate.get("stopped")
        and candidate.get("suffix_reconnectable") is False
    ]
    positive_mean = (
        statistics.mean(positive_stop_probabilities)
        if positive_stop_probabilities
        else None
    )
    negative_mean = (
        statistics.mean(negative_stop_probabilities)
        if negative_stop_probabilities
        else None
    )
    return {
        "context_units": len(context_reward_means),
        "context_seed_repeats": len(available),
        "candidate_count": len(candidates),
        "reward_mean": statistics.mean(rewards) if rewards else None,
        "reward_context_mean": reward_mean,
        "reward_context_ci95_low": reward_ci_low,
        "reward_context_ci95_high": reward_ci_high,
        "decode_rate": (
            statistics.mean(float(candidate["decoded"]) for candidate in candidates)
            if candidates
            else None
        ),
        "stop_rate": (
            statistics.mean(float(candidate["stopped"]) for candidate in candidates)
            if candidates
            else None
        ),
        "valid_eos_rate": (
            len(valid_stops) / len(candidates)
            if candidates and has_reconnection_labels
            else None
        ),
        "premature_eos_rate": (
            len(premature_stops) / len(candidates)
            if candidates and has_reconnection_labels
            else None
        ),
        "no_eos_by_budget_rate": (
            statistics.mean(float(not candidate["stopped"]) for candidate in candidates)
            if candidates and has_reconnection_labels
            else None
        ),
        "suffix_reconnect_rate": (
            statistics.mean(float(value) for value in reconnectable)
            if reconnectable
            else None
        ),
        "psnr_mean_successful": statistics.mean(psnrs) if psnrs else None,
        "stop_supervision_units": sum(bool(row.get("stop_states")) for row in available),
        "stop_positive_states": len(positive_stop_probabilities),
        "stop_negative_states": len(negative_stop_probabilities),
        "stop_positive_eos_probability": positive_mean,
        "stop_negative_eos_probability": negative_mean,
        "stop_positive_eos_rank_mean": (
            statistics.mean(positive_stop_ranks) if positive_stop_ranks else None
        ),
        "stop_negative_eos_rank_mean": (
            statistics.mean(negative_stop_ranks) if negative_stop_ranks else None
        ),
        "stop_probability_margin": (
            None
            if positive_mean is None or negative_mean is None
            else positive_mean - negative_mean
        ),
        "generated_bytes_mean": (
            statistics.mean(generated_lengths) if generated_lengths else None
        ),
        "generated_bytes_median": (
            statistics.median(generated_lengths) if generated_lengths else None
        ),
        "generated_bytes_min": min(generated_lengths) if generated_lengths else None,
        "generated_bytes_max": max(generated_lengths) if generated_lengths else None,
        "informative_groups": sum(row["status"] == "ready" for row in rows),
        "zero_variance_groups": sum(
            row["status"] == "zero_reward_variance" for row in rows
        ),
        "status_hist": dict(Counter(row["status"] for row in rows)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = _grpo_config(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    first_model = load_model(args.checkpoint_dirs[0], device)
    first_model.eval()
    sampler = _build_sampler(args, first_model, config)
    contexts = _fixed_contexts(sampler, args.tasks, args.num_contexts)
    for selection in contexts.get("fim", []):
        if not fim_ground_truth_reconnects(
            selection.sample, slice_layout=args.fim_slice_layout
        ):
            raise RuntimeError(
                "Ground-truth FIM middle failed parser reconnection sanity for "
                f"dataset_index={selection.dataset_index}; stop-label metrics "
                "would not be trustworthy"
            )
    fixed_stop_states: dict[tuple[int, int], list[FIMStopState]] = {}

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for checkpoint_index, (label, checkpoint_dir) in enumerate(
        zip(args.labels, args.checkpoint_dirs)
    ):
        model = (
            first_model
            if checkpoint_index == 0
            else load_model(checkpoint_dir, device)
        )
        model.eval()
        summaries[label] = {}
        print(f"\n===== checkpoint={label} =====", flush=True)
        for task in args.tasks:
            task_rows: list[dict[str, Any]] = []
            for context_index, selection in enumerate(contexts[task]):
                for rollout_seed in args.rollout_seeds:
                    effective_seed = rollout_seed + context_index * 1_000_003
                    torch.manual_seed(effective_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(effective_seed)
                    result = _prepare(task, model, selection, config, device)
                    candidates = (
                        []
                        if result.prepared is None
                        else _candidate_rows(result.prepared)
                    )
                    if task == "fim" and result.prepared is not None:
                        reconnects = fim_endpoint_reconnects(
                            result.prepared.sample,
                            result.prepared.candidates,
                            slice_layout=args.fim_slice_layout,
                        )
                        for candidate_row, reconnects_suffix in zip(
                            candidates, reconnects
                        ):
                            candidate_row["suffix_reconnectable"] = reconnects_suffix
                    stop_key = (context_index, rollout_seed)
                    if task == "fim" and checkpoint_index == 0:
                        fixed_stop_states[stop_key] = (
                            []
                            if result.prepared is None
                            else fim_stop_states(
                                result.prepared.sample,
                                result.prepared.candidates,
                                slice_layout=args.fim_slice_layout,
                                negative_samples=args.stop_negative_samples,
                                max_positive_states=args.stop_max_positive_states,
                            )
                        )
                    states = fixed_stop_states.get(stop_key, []) if task == "fim" else []
                    stop_state_rows = _stop_state_rows(
                        model, selection.sample, states, device
                    )
                    row = {
                        "checkpoint": label,
                        "checkpoint_dir": str(checkpoint_dir),
                        "task": task,
                        "context_index": context_index,
                        "draw_index": selection.draw_index,
                        "dataset_index": selection.dataset_index,
                        "selection_attempt": selection.attempt,
                        "hole_spec": selection.hole_spec,
                        "h264_path": str(selection.sample.h264_path),
                        "rollout_seed": rollout_seed,
                        "effective_seed": effective_seed,
                        "status": result.status,
                        "candidate_count": len(candidates),
                        "candidates": candidates,
                        "stop_states": stop_state_rows,
                        "stop_probe_source_checkpoint": (
                            args.labels[0] if stop_state_rows else None
                        ),
                    }
                    task_rows.append(row)
                    all_rows.append(row)
                    reward_text = (
                        "none"
                        if result.prepared is None
                        else f"{result.prepared.mean_reward:.4f}"
                    )
                    print(
                        f"[{task}] context={context_index} seed={rollout_seed} "
                        f"status={result.status} decoded={result.decoded_count}/"
                        f"{result.candidate_count} mean_reward={reward_text}",
                        flush=True,
                    )
            summaries[label][task] = _summarize(task_rows)
            summary = summaries[label][task]
            print(
                f"[{label}/{task}] reward={summary['reward_mean']!s} "
                f"context_CI95=[{summary['reward_context_ci95_low']!s}, "
                f"{summary['reward_context_ci95_high']!s}] "
                f"decode={summary['decode_rate']!s} stop={summary['stop_rate']!s}",
                f" reconnect={summary['suffix_reconnect_rate']!s}",
                f" eos_margin={summary['stop_probability_margin']!s}",
                flush=True,
            )
        if checkpoint_index > 0:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    unit_reward: dict[tuple[str, str, int, int], float] = {}
    for row in all_rows:
        if row["candidate_count"]:
            unit_reward[
                (
                    row["checkpoint"],
                    row["task"],
                    row["context_index"],
                    row["rollout_seed"],
                )
            ] = statistics.mean(
                candidate["reward"] for candidate in row["candidates"]
            )

    base_label = args.labels[0]
    for label in args.labels:
        for task in args.tasks:
            current = summaries[label][task]
            base = summaries[base_label][task]
            current["reward_delta_from_base"] = (
                None
                if current["reward_mean"] is None or base["reward_mean"] is None
                else current["reward_mean"] - base["reward_mean"]
            )
            paired_context_deltas = []
            paired_seed_pairs = 0
            for context_index in range(args.num_contexts):
                context_deltas = []
                for rollout_seed in args.rollout_seeds:
                    base_key = (base_label, task, context_index, rollout_seed)
                    current_key = (label, task, context_index, rollout_seed)
                    if base_key in unit_reward and current_key in unit_reward:
                        paired_seed_pairs += 1
                        context_deltas.append(
                            unit_reward[current_key] - unit_reward[base_key]
                        )
                if context_deltas:
                    paired_context_deltas.append(statistics.mean(context_deltas))
            paired_mean, paired_low, paired_high = _mean_ci95(
                paired_context_deltas
            )
            current["paired_reward_delta_contexts"] = len(paired_context_deltas)
            current["paired_reward_delta_seed_pairs"] = paired_seed_pairs
            current["paired_reward_delta_mean"] = paired_mean
            current["paired_reward_delta_ci95_low"] = paired_low
            current["paired_reward_delta_ci95_high"] = paired_high

    details_path = args.out_dir / "fixed_reward_details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    report = {
        "config": {
            "manifest": str(args.manifest),
            "nal_index_path": str(args.nal_index_path),
            "checkpoint_dirs": [str(path) for path in args.checkpoint_dirs],
            "labels": args.labels,
            "tasks": args.tasks,
            "num_contexts": args.num_contexts,
            "rollout_seeds": args.rollout_seeds,
            "group_size": args.group_size,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "context_seed": args.context_seed,
            "fim_slice_layout": args.fim_slice_layout,
            "stop_negative_samples": args.stop_negative_samples,
            "stop_max_positive_states": args.stop_max_positive_states,
        },
        "summaries": summaries,
    }
    summary_path = args.out_dir / "fixed_reward_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDetails: {details_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
