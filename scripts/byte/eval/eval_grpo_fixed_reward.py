"""Compare implemented GRPO reward across checkpoints on fixed contexts.

This is Experiment 1B: it asks whether several GRPO updates produce a policy
with higher expected *implemented* reward. Every checkpoint sees the same AR
and FIM contexts and the same rollout seeds. Each context/seed draws a candidate
group, so the reported confidence interval reflects variation across fixed
context-seed units rather than variation in online context difficulty.

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
    GRPOConfig,
    PreparedGRPOStep,
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


def _mean_ci95(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    center = statistics.mean(values)
    if len(values) == 1:
        return center, center, center
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return center, center - half_width, center + half_width


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["candidate_count"] > 0]
    rewards = [
        candidate["reward"]
        for row in available
        for candidate in row["candidates"]
    ]
    unit_reward_means = [
        statistics.mean(candidate["reward"] for candidate in row["candidates"])
        for row in available
    ]
    reward_mean, reward_ci_low, reward_ci_high = _mean_ci95(unit_reward_means)
    candidates = [candidate for row in available for candidate in row["candidates"]]
    psnrs = [candidate["psnr"] for candidate in candidates if candidate["psnr"] is not None]
    return {
        "context_seed_units": len(available),
        "candidate_count": len(candidates),
        "reward_mean": statistics.mean(rewards) if rewards else None,
        "reward_unit_mean": reward_mean,
        "reward_unit_ci95_low": reward_ci_low,
        "reward_unit_ci95_high": reward_ci_high,
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
        "psnr_mean_successful": statistics.mean(psnrs) if psnrs else None,
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
                f"CI95=[{summary['reward_unit_ci95_low']!s}, "
                f"{summary['reward_unit_ci95_high']!s}] "
                f"decode={summary['decode_rate']!s} stop={summary['stop_rate']!s}",
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
            paired_deltas = []
            for context_index in range(args.num_contexts):
                for rollout_seed in args.rollout_seeds:
                    base_key = (base_label, task, context_index, rollout_seed)
                    current_key = (label, task, context_index, rollout_seed)
                    if base_key in unit_reward and current_key in unit_reward:
                        paired_deltas.append(
                            unit_reward[current_key] - unit_reward[base_key]
                        )
            paired_mean, paired_low, paired_high = _mean_ci95(paired_deltas)
            current["paired_reward_delta_units"] = len(paired_deltas)
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
        },
        "summaries": summaries,
    }
    summary_path = args.out_dir / "fixed_reward_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDetails: {details_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
