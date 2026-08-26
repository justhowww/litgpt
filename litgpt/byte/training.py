"""Training helpers for byte-domain language modeling.

These functions isolate byte batches, EOS calibration, task-aware validation,
and reconstruction logging from LitGPT's generic pretraining orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from lightning import Fabric
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader

from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    IGNORE_INDEX,
    REGION_BRIDGE,
    REGION_TARGET,
    SEQ_EOS_ID,
)
from litgpt.byte.grpo import (
    GRPOConfig,
    build_group_patch_inputs,
    group_token_log_probabilities,
    grpo_clipped_loss,
    prepare_grpo_step,
    prepare_grpo_step_ar,
    should_run_grpo as _should_run_grpo,
)
from litgpt.byte.grpo_context import (
    GRPOContextSelection,
    OnlineGRPOContextSampler,
)
from litgpt.byte.mrt import (
    MRTConfig,
    candidate_mean_log_probability,
    mrt_candidate_diagnostics,
    prepare_mrt_step,
    should_run_mrt,
)
from litgpt.byte.reconstruction import (
    ReconstructionEvalConfig,
    ReconstructionSample,
    ReconstructionVisualization,
    _unwrap_model,
    run_reconstruction_probe,
    save_reconstruction_sample_manifest,
    select_reconstruction_samples,
)
from litgpt.byte.free_run_eval import (
    FreeRunEvalConfig,
    FreeRunSample,
    prepare_free_run_samples,
    run_free_run_eval,
)
from litgpt.utils import chunked_cross_entropy


@dataclass
class ByteTrainingRuntime:
    """Prepared byte-domain state used by the generic pretraining loop."""

    ce_loss_weight: float = 1.0
    ce_byte_only: bool = False
    eos_loss_weight: float = 1.0
    eos_aux_loss_weight: float = 0.0
    reconstruction_config: ReconstructionEvalConfig | None = None
    reconstruction_samples: dict[str, list[ReconstructionSample]] = field(
        default_factory=dict
    )
    mrt_config: MRTConfig | None = None
    mrt_samples: list[ReconstructionSample] = field(default_factory=list)
    free_run_config: FreeRunEvalConfig | None = None
    free_run_samples: list[FreeRunSample] = field(default_factory=list)
    grpo_config: GRPOConfig | None = None
    grpo_ar_pool: list[FreeRunSample] = field(default_factory=list)
    grpo_fim_pool: list[ReconstructionSample] = field(default_factory=list)
    grpo_context_sampler: OnlineGRPOContextSampler | None = None
    grpo_context_log_path: Path | None = None
    grpo_rank: int = 0
    grpo_world_size: int = 1
    # Frozen initial-policy copy for the GRPO KL penalty. Populated by the
    # caller (litgpt.pretrain.main) once weights are loaded, since building it
    # requires the fully-materialized model -- prepare() runs before that.
    reference_model: nn.Module | None = None

    @classmethod
    def prepare(
        cls,
        fabric: Fabric,
        train_dataset,
        val_dataset,
        out_dir: Path,
        *,
        ce_loss_weight: float,
        ce_byte_only: bool,
        eos_loss_weight: float,
        eos_aux_loss_weight: float,
        reconstruction_config: ReconstructionEvalConfig | None,
        mrt_config: MRTConfig | None,
        free_run_config: FreeRunEvalConfig | None = None,
        grpo_config: GRPOConfig | None = None,
    ) -> "ByteTrainingRuntime":
        """Prepare byte probes and online-training context sources."""
        reconstruction_samples: dict[str, list[ReconstructionSample]] = {}
        if reconstruction_config is not None:
            tasks = (
                ("ar", "fim")
                if reconstruction_config.task == "both"
                else (reconstruction_config.task,)
            )
            reconstruction_samples = {
                task: select_reconstruction_samples(
                    val_dataset,
                    reconstruction_config.num_samples,
                    reconstruction_config.max_target_bytes,
                    task,
                    force_eos_stopping=reconstruction_config.learned_eos_stopping,
                )
                for task in tasks
            }
            if fabric.global_rank == 0:
                for task, samples in reconstruction_samples.items():
                    manifest_name = (
                        f"reconstruction_samples_{task}.json"
                        if len(reconstruction_samples) > 1
                        else "reconstruction_samples.json"
                    )
                    save_reconstruction_sample_manifest(
                        samples, out_dir / manifest_name
                    )
            if fabric.world_size != 1:
                fabric.print(
                    "Byte reconstruction validation currently requires a "
                    "single-device run; disabling it."
                )
                reconstruction_config = None
                reconstruction_samples = {}

        mrt_samples: list[ReconstructionSample] = []
        if mrt_config is not None and mrt_config.enabled:
            mrt_config.validate()
            if fabric.world_size != 1:
                raise ValueError("Online byte MRT currently supports one device only")
            mrt_samples = select_reconstruction_samples(
                train_dataset,
                mrt_config.context_pool_size,
                mrt_config.max_target_bytes,
                "fim",
                force_eos_stopping=mrt_config.learned_eos,
            )
            if not mrt_samples:
                raise ValueError(
                    "MRT is enabled but no eligible FIM samples were found"
                )
            if fabric.global_rank == 0:
                save_reconstruction_sample_manifest(
                    mrt_samples, out_dir / "mrt_training_samples.json"
                )
            fabric.print(
                f"Online MRT enabled: {len(mrt_samples)} FIM contexts, "
                f"{mrt_config.num_candidates} candidates every "
                f"{mrt_config.interval} steps, risk={mrt_config.risk_mode}"
            )

        grpo_ar_pool: list[FreeRunSample] = []
        grpo_fim_pool: list[ReconstructionSample] = []
        grpo_context_sampler: OnlineGRPOContextSampler | None = None
        grpo_context_log_path: Path | None = None
        if grpo_config is not None and grpo_config.enabled:
            grpo_config.validate()
            if fabric.world_size > 1 and isinstance(
                fabric.strategy, FSDPStrategy
            ):
                raise ValueError(
                    "Multi-GPU byte GRPO requires replicated DDP, not FSDP"
                )
            grpo_context_log_path = out_dir / (
                "grpo_contexts.jsonl"
                if fabric.world_size == 1
                else f"grpo_contexts_rank{fabric.global_rank:03d}.jsonl"
            )
            if grpo_config.context_sampling == "online":
                grpo_context_sampler = OnlineGRPOContextSampler.from_dataset(
                    train_dataset, grpo_config
                )
                fabric.print(
                    "Online GRPO enabled with just-in-time contexts from "
                    f"{len(grpo_context_sampler.indices)} training windows "
                    f"(alternating AR/FIM), group_size={grpo_config.group_size} "
                    f"per rank across {fabric.world_size} rank(s), "
                    f"every {grpo_config.interval} steps, "
                    f"kl_coeff={grpo_config.kl_coeff}"
                )
            else:
                # AR reuses free_run_eval's clip selection (prefix_frames/
                # cont_frames/slice_layout mirror FreeRunEvalConfig).  The
                # resulting samples are scored by prepare_grpo_step_ar, not by
                # the free-run probe itself.
                grpo_ar_pool = prepare_free_run_samples(
                    train_dataset,
                    FreeRunEvalConfig(
                        interval=1,
                        num_clips=grpo_config.ar_pool_size,
                        prefix_frames=grpo_config.ar_prefix_frames,
                        cont_frames=grpo_config.ar_cont_frames,
                        slice_layout=grpo_config.ar_slice_layout,
                    ),
                )
                grpo_fim_pool = select_reconstruction_samples(
                    train_dataset,
                    grpo_config.fim_pool_size,
                    grpo_config.max_target_bytes,
                    "fim",
                    force_eos_stopping=grpo_config.learned_eos,
                )
                if not grpo_ar_pool or not grpo_fim_pool:
                    raise ValueError(
                        "GRPO is enabled but no eligible AR and/or FIM samples "
                        "were found"
                    )
                if fabric.global_rank == 0:
                    # AR pool has no ReconstructionSample-shaped manifest writer.
                    save_reconstruction_sample_manifest(
                        grpo_fim_pool, out_dir / "grpo_training_samples_fim.json"
                    )
                fabric.print(
                    f"Online GRPO enabled: {len(grpo_ar_pool)} AR + "
                    f"{len(grpo_fim_pool)} FIM fixed contexts (alternating), "
                    f"group_size={grpo_config.group_size} per rank across "
                    f"{fabric.world_size} rank(s), every "
                    f"{grpo_config.interval} steps, kl_coeff={grpo_config.kl_coeff}"
                )

        free_run_samples: list[FreeRunSample] = []
        if free_run_config is not None and free_run_config.enabled:
            # Selection is deterministic, so every rank builds the same fixed clips.
            # On multi-GPU the probe runs on rank 0 via FSDP summon_full_params
            # (see evaluate_free_run), so it is NOT disabled for world_size > 1.
            free_run_samples = prepare_free_run_samples(val_dataset, free_run_config)
            fabric.print(
                f"Free-run validation: {len(free_run_samples)} SPS-anchored "
                f"continuation clips every {free_run_config.interval} steps"
            )
            if not free_run_samples:
                free_run_config = None
            elif fabric.world_size > 1:
                fabric.print(
                    "Free-run validation will run on rank 0 via FSDP "
                    "summon_full_params on multi-GPU; verify the first probe does "
                    "not hang (else set --free-run-eval-interval 0 and eval offline)."
                )

        if ce_loss_weight < 0:
            raise ValueError("CE loss weight must be non-negative")
        return cls(
            ce_loss_weight=ce_loss_weight,
            ce_byte_only=ce_byte_only,
            eos_loss_weight=eos_loss_weight,
            eos_aux_loss_weight=eos_aux_loss_weight,
            reconstruction_config=reconstruction_config,
            reconstruction_samples=reconstruction_samples,
            mrt_config=mrt_config,
            mrt_samples=mrt_samples,
            free_run_config=free_run_config,
            free_run_samples=free_run_samples,
            grpo_config=grpo_config,
            grpo_ar_pool=grpo_ar_pool,
            grpo_fim_pool=grpo_fim_pool,
            grpo_context_sampler=grpo_context_sampler,
            grpo_context_log_path=grpo_context_log_path,
            grpo_rank=fabric.global_rank,
            grpo_world_size=fabric.world_size,
        )

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return byte_training_loss(
            logits,
            targets,
            ce_byte_only=self.ce_byte_only,
            eos_loss_weight=self.eos_loss_weight,
            eos_aux_loss_weight=self.eos_aux_loss_weight,
        )

    def should_run_mrt(self, next_step: int, is_accumulating: bool) -> bool:
        return (
            not is_accumulating
            and self.mrt_config is not None
            and should_run_mrt(next_step, self.mrt_config)
        )

    def should_run_grpo(self, next_step: int, is_accumulating: bool) -> bool:
        return (
            not is_accumulating
            and self.grpo_config is not None
            and _should_run_grpo(next_step, self.grpo_config)
        )

    def _log_grpo_context(
        self,
        *,
        next_step: int,
        update_index: int,
        task: str,
        sample: FreeRunSample | ReconstructionSample | None,
        selection: GRPOContextSelection | None,
        status: str,
        prepared=None,
    ) -> None:
        """Append a reproducibility record for one GRPO context draw."""
        if self.grpo_context_log_path is None:
            return
        record = {
            "step": next_step,
            "update_index": update_index,
            "task": task,
            "sampling": (
                self.grpo_config.context_sampling if self.grpo_config else "unknown"
            ),
            "status": status,
            "rank": self.grpo_rank,
            "world_size": self.grpo_world_size,
            "dataset_index": (
                selection.dataset_index if selection is not None else None
            ),
            "draw_index": selection.draw_index if selection is not None else None,
            "selection_attempt": selection.attempt if selection is not None else None,
            "hole_spec": (
                list(selection.hole_spec)
                if selection is not None and selection.hole_spec is not None
                else None
            ),
            "h264_path": str(sample.h264_path) if sample is not None else None,
        }
        if prepared is not None:
            record.update(
                {
                    "mean_reward": prepared.mean_reward,
                    "reward_min": float(prepared.rewards.min()),
                    "reward_max": float(prepared.rewards.max()),
                    "decode_rate": prepared.decode_rate,
                    "group_size": len(prepared.candidates),
                }
            )
        with self.grpo_context_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _all_ranks_ready(fabric: Fabric, local_ready: bool) -> bool:
        """Require every DDP rank to enter or skip the GRPO backward together."""
        if fabric.world_size == 1:
            return local_ready
        ready = torch.tensor(
            1.0 if local_ready else 0.0,
            device=fabric.device,
        )
        ready = fabric.all_reduce(ready, reduce_op="min")
        return bool(ready.item())

    @staticmethod
    def _distributed_mean_metrics(
        fabric: Fabric,
        values: dict[str, float],
    ) -> dict[str, float]:
        """Average a fixed set of scalar diagnostics across DDP ranks."""
        if fabric.world_size == 1 or not values:
            return values
        keys = sorted(values)
        packed = torch.tensor(
            [values[key] for key in keys],
            device=fabric.device,
            dtype=torch.float32,
        )
        packed = fabric.all_reduce(packed, reduce_op="mean")
        return {key: float(packed[i]) for i, key in enumerate(keys)}

    def run_grpo(
        self,
        fabric: Fabric,
        optimizer,
        model: nn.Module,
        max_norm: float,
        next_step: int,
    ) -> dict[str, float | str] | None:
        """Sample one group and run mu clipped-surrogate gradient steps on it.

        Self-contained: owns its own zero_grad/backward/clip/step cycle,
        separate from the shared CE optimizer step. Must only be called after
        the CE step for this iteration has already been applied (see the
        call site in litgpt.pretrain.fit) so it starts from a clean gradient
        slate and doesn't clobber CE's accumulated gradient.

        Alternates between AR and FIM each call. Contexts come either from the
        legacy fixed pools or from the deterministic just-in-time sampler.
        """
        config = self.grpo_config
        if config is None:
            return None
        update_index = (next_step - config.start_step) // config.interval - 1
        pool_name = "ar" if update_index % 2 == 0 else "fim"
        task_draw_index = update_index // 2
        # Consecutive positions in the task's shuffled order are partitioned
        # across ranks.  A four-rank update therefore trains on four contexts,
        # while resume remains a pure function of step/rank/world-size.
        context_draw_index = (
            task_draw_index * fabric.world_size + fabric.global_rank
        )
        selection: GRPOContextSelection | None = None
        sample: FreeRunSample | ReconstructionSample | None = None
        if config.context_sampling == "online":
            if self.grpo_context_sampler is not None:
                selection = self.grpo_context_sampler.sample(
                    pool_name,
                    context_draw_index,
                    stride=fabric.world_size,
                )
                sample = selection.sample if selection is not None else None
        else:
            if self.grpo_ar_pool and self.grpo_fim_pool:
                pool = self.grpo_ar_pool if pool_name == "ar" else self.grpo_fim_pool
                sample = pool[context_draw_index % len(pool)]

        if not self._all_ranks_ready(fabric, sample is not None):
            self._log_grpo_context(
                next_step=next_step,
                update_index=update_index,
                task=pool_name,
                sample=sample,
                selection=selection,
                status=(
                    "no_eligible_context"
                    if sample is None
                    else "peer_has_no_eligible_context"
                ),
            )
            return {
                "grpo/skipped": 1.0,
                "grpo/pool": pool_name,
                "grpo/skip_reason": "distributed_context_unavailable",
            }
        assert sample is not None

        if pool_name == "ar":
            prepared = prepare_grpo_step_ar(model, sample, config, fabric.device)
        else:
            prepared = prepare_grpo_step(model, sample, config, fabric.device)
        if not self._all_ranks_ready(fabric, prepared is not None):
            self._log_grpo_context(
                next_step=next_step,
                update_index=update_index,
                task=pool_name,
                sample=sample,
                selection=selection,
                status=(
                    "rollout_skipped"
                    if prepared is None
                    else "peer_rollout_skipped"
                ),
            )
            return {
                "grpo/skipped": 1.0,
                "grpo/pool": pool_name,
                "grpo/skip_reason": "distributed_rollout_skipped",
            }
        assert prepared is not None
        # prepare_grpo_step[_ar] run under torch.inference_mode() (correct,
        # faster than no_grad, for pure sampling/scoring with no gradient
        # ever needed there) -- but that permanently tags every tensor
        # created inside as an "inference tensor", which can never
        # participate in autograd afterward even once back in normal mode.
        # advantages gets multiplied against a gradient-tracked ratio below,
        # so it must be cloned here, now that we're outside that context --
        # cloning inside prepare_grpo_step[_ar] itself would not help, since
        # the clone would still happen while inference-mode is active.
        advantages = prepared.advantages.clone()

        raw_model = _unwrap_model(model)
        patch_size = int(raw_model.config.byte_patch_size)
        # AR stops on a structural frame-count condition, not a sampled EOS
        # token -- there is nothing to append to a stopped AR candidate's
        # target sequence, unlike FIM's learned-EOS stopping.
        inputs, labels, supervised = build_group_patch_inputs(
            prepared.sample,
            prepared.candidates,
            patch_size,
            fabric.device,
            append_eos_on_stop=(pool_name != "ar"),
        )

        with torch.no_grad(), fabric.autocast():
            old_gathered = group_token_log_probabilities(model, inputs, labels).detach()
            reference_gathered = None
            if self.reference_model is not None and config.kl_coeff > 0:
                reference_gathered = group_token_log_probabilities(
                    self.reference_model, inputs, labels
                ).detach()

        loss_metrics: dict[str, float] = {}
        final_loss = 0.0
        for _ in range(config.mu):
            optimizer.zero_grad()
            with fabric.autocast():
                gathered = group_token_log_probabilities(model, inputs, labels)
                loss, loss_metrics = grpo_clipped_loss(
                    gathered,
                    old_gathered,
                    supervised,
                    advantages,
                    reference_gathered,
                    config.kl_coeff,
                    config.clip_range,
                )
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=max_norm)
            optimizer.step()
            final_loss = float(loss.detach())
        optimizer.zero_grad()

        mean_metrics = {
            "grpo/mean_reward": prepared.mean_reward,
            "grpo/reward_std": float(prepared.rewards.std(unbiased=False)),
            "grpo/decode_rate": prepared.decode_rate,
            f"grpo/{pool_name}/mean_reward": prepared.mean_reward,
            f"grpo/{pool_name}/decode_rate": prepared.decode_rate,
            "grpo/loss": final_loss,
        }
        mean_metrics.update(loss_metrics)
        mean_metrics = self._distributed_mean_metrics(fabric, mean_metrics)

        reward_min = torch.tensor(
            float(prepared.rewards.min()), device=fabric.device
        )
        reward_max = torch.tensor(
            float(prepared.rewards.max()), device=fabric.device
        )
        group_size = torch.tensor(
            float(len(prepared.candidates)), device=fabric.device
        )
        if fabric.world_size > 1:
            reward_min = fabric.all_reduce(reward_min, reduce_op="min")
            reward_max = fabric.all_reduce(reward_max, reduce_op="max")
            group_size = fabric.all_reduce(group_size, reduce_op="sum")

        metrics: dict[str, float | str] = {
            "grpo/skipped": 0.0,
            "grpo/pool": pool_name,
            "grpo/task": prepared.task,
            "grpo/reward_min": float(reward_min),
            "grpo/reward_max": float(reward_max),
            # Advantages are normalized within each rank's one-context group.
            # Across ranks these are several independent groups, not one larger
            # GRPO group, so preserve group_size's original local meaning.
            "grpo/group_size": float(group_size) / fabric.world_size,
            "grpo/rollouts_per_update": float(group_size),
            "grpo/contexts_per_update": float(fabric.world_size),
            "grpo/mu": float(config.mu),
        }
        metrics.update(mean_metrics)

        finite_psnrs = prepared.psnrs[torch.isfinite(prepared.psnrs)]
        psnr_stats = torch.tensor(
            [
                float(finite_psnrs.sum()) if finite_psnrs.numel() else 0.0,
                float(finite_psnrs.numel()),
            ],
            device=fabric.device,
        )
        if fabric.world_size > 1:
            psnr_stats = fabric.all_reduce(psnr_stats, reduce_op="sum")
        if float(psnr_stats[1]) > 0:
            metrics["grpo/psnr_mean"] = float(psnr_stats[0] / psnr_stats[1])
        if selection is not None and fabric.global_rank == 0:
            metrics["grpo/context_dataset_index_rank0"] = float(
                selection.dataset_index
            )
            metrics["grpo/context_selection_attempt_rank0"] = float(
                selection.attempt
            )
        self._log_grpo_context(
            next_step=next_step,
            update_index=update_index,
            task=pool_name,
            sample=sample,
            selection=selection,
            status="updated",
            prepared=prepared,
        )
        return metrics

    def log_grpo(
        self, fabric: Fabric, metrics: dict[str, float | str], step: int
    ) -> None:
        if fabric.global_rank != 0:
            return
        pool = metrics.get("grpo/pool", "?")
        if metrics.get("grpo/skipped"):
            fabric.log_dict(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step
            )
            reason = metrics.get("grpo/skip_reason", "rollout_skipped")
            fabric.print(f"GRPO step {step} [{pool}] skipped: {reason}")
            return
        # fabric.log_dict requires scalar values; string fields (pool/task) are
        # print-only diagnostics, not logged as metrics.
        loggable = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        fabric.log_dict(loggable, step=step)
        fabric.print(
            f"GRPO step {step} [{pool}/{metrics.get('grpo/task', '?')}] "
            f"reward: {metrics['grpo/mean_reward']:.4f} "
            f"(std {metrics['grpo/reward_std']:.4f}), "
            f"decode: {metrics['grpo/decode_rate']:.1%}, "
            f"loss: {metrics['grpo/loss']:.4f}, "
            f"mu: {int(metrics.get('grpo/mu', 1))}, "
            f"clip_frac: {metrics.get('grpo/clip_fraction', 0.0):.3f}"
            + (
                f", kl: {metrics['grpo/kl_to_reference']:.4f}"
                if "grpo/kl_to_reference" in metrics
                else ""
            )
        )

    def run_mrt(
        self,
        fabric: Fabric,
        model: nn.Module,
        next_step: int,
    ) -> dict[str, float] | None:
        """Apply one sparse online MRT backward pass and return log metrics."""
        config = self.mrt_config
        if config is None or not self.mrt_samples:
            return None
        update_index = (next_step - config.start_step) // config.interval - 1
        sample = self.mrt_samples[update_index % len(self.mrt_samples)]
        prepared = prepare_mrt_step(model, sample, config, fabric.device)
        if prepared is None:
            return {"mrt/skipped": 1.0}

        for candidate, coefficient in zip(
            prepared.candidates, prepared.coefficients
        ):
            if abs(float(coefficient)) < 1e-8:
                continue
            with fabric.autocast():
                score = candidate_mean_log_probability(
                    model,
                    prepared.sample,
                    candidate,
                    fabric.device,
                    temperature=config.temperature,
                )
            # MRT uses one context per optimizer step, so unlike CE it is not
            # divided by gradient accumulation.
            fabric.backward(config.weight * coefficient * score)
        metrics = {
            "mrt/skipped": 0.0,
            "mrt/expected_risk": prepared.expected_risk,
            "mrt/weighted_expected_risk": config.weight
            * prepared.expected_risk,
            "mrt/decode_rate": prepared.decode_rate,
            "mrt/ground_truth_probability": prepared.ground_truth_probability,
            "mrt/num_unique_candidates": float(len(prepared.candidates)),
            "mrt/risk_min": float(prepared.risks.min()),
            "mrt/risk_mean": float(prepared.risks.mean()),
            "mrt/risk_max": float(prepared.risks.max()),
            "mrt/risk_std": float(prepared.risks.std(unbiased=False)),
        }
        metrics.update(mrt_candidate_diagnostics(prepared))
        decoded_mses = prepared.candidate_mses[
            torch.isfinite(prepared.candidate_mses)
        ]
        if decoded_mses.numel() > 0:
            metrics.update(
                {
                    "mrt/mse_min": float(decoded_mses.min()),
                    "mrt/mse_mean": float(decoded_mses.mean()),
                    "mrt/mse_p50": float(torch.quantile(decoded_mses, 0.5)),
                    "mrt/mse_p90": float(torch.quantile(decoded_mses, 0.9)),
                    "mrt/mse_max": float(decoded_mses.max()),
                }
            )
        return metrics

    @staticmethod
    def gradient_l2_norm(model: nn.Module) -> float:
        """Return the global L2 norm of the gradients currently on the model."""
        total = None
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            squared_norm = parameter.grad.detach().float().square().sum()
            total = squared_norm if total is None else total + squared_norm
        return 0.0 if total is None else float(total.sqrt())

    @staticmethod
    def capture_gradients(model: nn.Module) -> tuple[Tensor | None, ...]:
        """Snapshot current gradients so a later backward contribution is measurable."""
        return tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        )

    @staticmethod
    def gradient_delta_l2_norm(
        model: nn.Module, before: tuple[Tensor | None, ...]
    ) -> float:
        """Return the L2 norm of gradients added since ``before`` was captured."""
        total = None
        for parameter, previous in zip(model.parameters(), before):
            current = parameter.grad
            if current is None and previous is None:
                continue
            if current is None:
                delta = -previous
            elif previous is None:
                delta = current.detach()
            else:
                delta = current.detach() - previous
            squared_norm = delta.float().square().sum()
            total = squared_norm if total is None else total + squared_norm
        return 0.0 if total is None else float(total.sqrt())

    def log_mrt(
        self, fabric: Fabric, metrics: dict[str, float], step: int
    ) -> None:
        # TensorBoard's x-axis is the device-independent optimizer step. The
        # per-rank microiteration count changes when a checkpoint moves between
        # launchers with different world sizes.
        fabric.log_dict(metrics, step=step)
        if metrics["mrt/skipped"]:
            fabric.print(
                f"MRT step {step} skipped: reference frame did not decode strictly"
            )
        else:
            fabric.print(
                f"MRT step {step} | risk: {metrics['mrt/expected_risk']:.4f}, "
                f"objective: {metrics['optimization/combined_objective_sampled']:.4f}, "
                f"CE(raw/weighted): {metrics['optimization/raw_ce_step']:.4f}/"
                f"{metrics['optimization/weighted_ce_step']:.4f}, "
                f"grad(CE/MRT/combined): "
                f"{metrics['optimization/weighted_ce_grad_norm']:.4f}/"
                f"{metrics['optimization/mrt_grad_norm']:.4f}/"
                f"{metrics['optimization/combined_grad_norm_pre_clip']:.4f}, "
                f"spread: {metrics['mrt/risk_std']:.4f}, "
                f"sampled spread: {metrics['mrt/sampled_risk_std']:.4f}, "
                f"decode: {metrics['mrt/decode_rate']:.1%}, "
                f"unique: {int(metrics['mrt/num_unique_candidates'])}"
            )

    def reconstruction_due(self, step: int) -> bool:
        return (
            self.reconstruction_config is not None
            and bool(self.reconstruction_samples)
            and step % self.reconstruction_config.interval == 0
        )

    def evaluate_reconstruction(
        self,
        fabric: Fabric,
        model: nn.Module,
        log_step: int,
        *,
        final: bool = False,
    ) -> None:
        """Run and log all configured AR/FIM decoder probes."""
        config = self.reconstruction_config
        if config is None:
            return
        for task, samples in self.reconstruction_samples.items():
            if not samples:
                continue
            visualizations: list[ReconstructionVisualization] = []
            metrics = run_reconstruction_probe(
                model,
                samples,
                config,
                fabric.device,
                visualizations=visualizations,
            )
            prefix = "Final " if final else ""
            fabric.print(
                format_reconstruction_metrics(
                    f"{prefix}{task.upper()} reconstruction validation",
                    metrics,
                )
            )
            fabric.log_dict(
                namespace_reconstruction_metrics(metrics, task),
                step=log_step,
            )
            self._log_reconstruction_visualizations(
                fabric, task, visualizations, log_step
            )

    def free_run_due(self, step: int) -> bool:
        return (
            self.free_run_config is not None
            and bool(self.free_run_samples)
            and step % self.free_run_config.interval == 0
        )

    def evaluate_free_run(
        self, fabric: Fabric, model: nn.Module, log_step: int
    ) -> None:
        """Free-run generation probe: parser-based survival-length + validity.

        Measures what val CE cannot (exposure-bias / the illegal-token tail). See
        litgpt/byte/free_run_eval.py.
        """
        if self.free_run_config is None or not self.free_run_samples:
            return
        # Under FSDP, EVERY model forward all-gathers the sharded params -- a collective that
        # all ranks must enter together. So the free-run generation runs on EVERY rank in
        # lockstep (only rank 0's metrics are logged below). Running it on rank 0 alone
        # deadlocks: rank 0's per-forward unshard all-gather waits forever for the idle
        # ranks (summon_full_params does NOT prevent forward's own unshard). Sampled
        # generation would diverge across ranks (different #forwards -> mismatched
        # collectives -> deadlock), so _run_free_run_guarded forces greedy on multi-GPU.
        metrics = self._run_free_run_guarded(fabric, model)

        if fabric.global_rank == 0 and metrics:
            fabric.print(
                "Free-run validation | survival_bytes "
                f"{metrics.get('val_freerun/survival_bytes', 0.0):.0f} | "
                f"full_cont_rate {metrics.get('val_freerun/full_continuation_rate', 0.0):.3f} | "
                f"start_codes {metrics.get('val_freerun/start_codes_emitted', 0.0):.2f}"
            )
            fabric.log_dict(metrics, step=log_step)
        if fabric.world_size > 1:
            fabric.barrier()

    def _run_free_run_guarded(
        self, fabric: Fabric, model: nn.Module
    ) -> dict[str, float]:
        config = self.free_run_config
        if fabric.world_size > 1 and config.temperature != 0.0:
            # Sampled generation would take a different number of AR steps on each rank,
            # mismatching the per-forward FSDP all-gathers -> deadlock. Force greedy so all
            # ranks generate identically and their collectives stay in lockstep.
            from dataclasses import replace

            config = replace(config, temperature=0.0)
            if fabric.global_rank == 0:
                fabric.print(
                    "Free-run probe: forcing greedy (temp 0) under multi-GPU so the "
                    "generation is identical across ranks."
                )
        try:
            return run_free_run_eval(
                model, self.free_run_samples, fabric.device, config
            )
        except Exception as exc:  # a probe must never crash training
            fabric.print(
                f"Free-run validation failed ({type(exc).__name__}: {exc}); skipping."
            )
            return {}

    @staticmethod
    def _log_reconstruction_visualizations(
        fabric: Fabric,
        task: str,
        visualizations: list[ReconstructionVisualization],
        step: int,
    ) -> None:
        """Log fixed decoder comparisons when a TensorBoard writer is active."""
        if fabric.global_rank != 0 or not visualizations:
            return
        column_order = (
            "ground_truth",
            "deleted_gap_strict",
            "deleted_gap_default",
            "model_learned_strict",
        )
        column_labels = (
            "Ground truth",
            "Deleted gap, strict",
            "Deleted gap, FFmpeg default",
            "Model reconstruction, strict",
        )
        for logger in fabric.loggers:
            writer = getattr(logger, "experiment", None)
            if not callable(getattr(writer, "add_image", None)):
                continue
            for visualization in visualizations:
                try:
                    reference = visualization.frames["ground_truth"]
                    missing = torch.zeros_like(reference)
                    missing[..., 0] = 1.0  # Red tile denotes a failed decode.
                    separator = torch.ones(
                        (reference.shape[0], 4, reference.shape[2]),
                        dtype=reference.dtype,
                    )
                    columns = [
                        visualization.frames.get(name, missing)
                        for name in column_order
                    ]
                    panel_parts: list[Tensor] = []
                    for column_index, column in enumerate(columns):
                        if column_index:
                            panel_parts.append(separator)
                        panel_parts.append(column)
                    panel = torch.cat(panel_parts, dim=1).clamp(0, 1)
                    tag = (
                        f"reconstruction/{task}/frames/"
                        f"sample_{visualization.sample_index:02d}"
                    )
                    writer.add_image(
                        tag, panel, global_step=step, dataformats="HWC"
                    )
                    if callable(getattr(writer, "add_text", None)):
                        rows = ["| Column | Decode status |", "|---|---|"]
                        rows.extend(
                            f"| {label} | {visualization.statuses.get(name, 'not decoded')} |"
                            for label, name in zip(column_labels, column_order)
                        )
                        rows.append("\nRed frame = decode failed or produced no frame.")
                        writer.add_text(
                            f"{tag}_legend",
                            "\n".join(rows),
                            global_step=step,
                        )
                except Exception as exc:
                    fabric.print(
                        "TensorBoard reconstruction image logging failed for "
                        f"{task} sample {visualization.sample_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )


def _fsdp_summon_full_params(model: nn.Module):
    """Return an all-ranks context that materializes full FSDP params (so rank 0
    can run a local forward for the free-run probe), or ``None`` if no FSDP module
    is found. MUST be entered by every rank together — call it on all ranks and
    enter the resulting context on all ranks to avoid a collective deadlock.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception:
        return None
    seen: list[int] = []
    stack: list[nn.Module | None] = [model]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.append(id(current))
        if isinstance(current, FSDP):
            return FSDP.summon_full_params(current, writeback=False, recurse=True)
        for attr in ("_forward_module", "module", "_orig_mod"):
            stack.append(getattr(current, attr, None))
    return None


def get_model_inputs_and_targets(
    batch, max_seq_length: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Adapt byte dict batches while preserving LitGPT's shifted text batches."""
    if isinstance(batch, dict):
        input_ids = batch["input_ids"][:, :max_seq_length].contiguous().long()
        targets = batch["labels"][:, :max_seq_length].contiguous().long()
        model_inputs = {"idx": input_ids}
        if input_ids.dim() == 3:
            # The MEGABYTE local Transformer receives these shifted by one byte:
            # logits for byte i are produced before target byte i is embedded.
            model_inputs["patch_targets"] = targets
        if "region_ids" in batch:
            model_inputs["region_ids"] = (
                batch["region_ids"][:, :max_seq_length].contiguous().long()
            )
        if "offset_ids" in batch:
            model_inputs["offset_ids"] = (
                batch["offset_ids"][:, :max_seq_length].contiguous().long()
            )
        return model_inputs, targets

    input_ids = batch[:, 0:max_seq_length].contiguous().long()
    targets = batch[:, 1 : (max_seq_length + 1)].contiguous().long()
    return {"idx": input_ids}, targets


def byte_weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eos_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Weight positive SEQ_EOS targets without changing ordinary byte labels."""
    if eos_loss_weight <= 0:
        raise ValueError("eos_loss_weight must be positive")
    if eos_loss_weight == 1.0:
        return chunked_cross_entropy(logits, targets)

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    losses = torch.nn.functional.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    supervised = flat_targets != IGNORE_INDEX
    weights = torch.ones_like(losses)
    weights[flat_targets == SEQ_EOS_ID] = eos_loss_weight
    return (losses * weights).sum() / supervised.sum().clamp_min(1)


def balanced_eos_auxiliary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Penalize early EOS and missed EOS with equal positive/negative weight."""
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_targets = targets.reshape(-1)
    supervised = flat_targets != IGNORE_INDEX
    positive = supervised & (flat_targets == SEQ_EOS_ID)
    negative = supervised & (flat_targets != SEQ_EOS_ID)

    eos_binary_logits = flat_logits[:, SEQ_EOS_ID] - torch.logsumexp(
        torch.cat(
            [
                flat_logits[:, :SEQ_EOS_ID],
                flat_logits[:, SEQ_EOS_ID + 1 :],
            ],
            dim=-1,
        ),
        dim=-1,
    )
    binary_targets = positive.to(flat_logits.dtype)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        eos_binary_logits,
        binary_targets,
        reduction="none",
    )
    positive_loss = losses[positive].sum() / positive.sum().clamp_min(1)
    negative_loss = losses[negative].sum() / negative.sum().clamp_min(1)
    return 0.5 * (positive_loss + negative_loss)


def byte_training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ce_byte_only: bool = False,
    eos_loss_weight: float = 1.0,
    eos_aux_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Combine byte CE with an optional balanced EOS calibration objective."""
    if eos_aux_loss_weight < 0:
        raise ValueError("eos_aux_loss_weight must be non-negative")
    if ce_byte_only:
        supervised_targets = targets[targets != IGNORE_INDEX]
        if bool((supervised_targets >= BYTE_VOCAB_SIZE).any()):
            raise ValueError(
                "Byte-only CE requires every supervised target to be a raw byte"
            )
        # Exclude EOS and all structural/control tokens from CE normalization.
        # This gives their logits zero CE gradient in oracle-length ablations.
        logits = logits[..., :BYTE_VOCAB_SIZE]
    loss = byte_weighted_cross_entropy(logits, targets, eos_loss_weight)
    if eos_aux_loss_weight == 0:
        return loss
    return loss + eos_aux_loss_weight * balanced_eos_auxiliary_loss(logits, targets)


def namespace_reconstruction_metrics(
    metrics: dict[str, float], task: str
) -> dict[str, float]:
    """Namespace shared reconstruction metrics by AR/FIM task."""
    return {
        key.replace("reconstruction/", f"reconstruction/{task}/", 1): value
        for key, value in metrics.items()
    }


def format_reconstruction_metrics(label: str, metrics: dict[str, float]) -> str:
    """Format one decoder-level validation result for console logs."""
    message = (
        f"{label} | strict decode rate: {metrics['reconstruction/decode_rate']:.2%} "
        f"({int(metrics['reconstruction/decoded'])}/{int(metrics['reconstruction/attempted'])}), "
        f"invalid generation: {int(metrics['reconstruction/invalid_generation'])}, "
        f"timeouts: {int(metrics['reconstruction/timeouts'])}, "
        f"missing frames: {int(metrics['reconstruction/missing_target_frames'])}, "
        f"unexpected failures: {int(metrics['reconstruction/unexpected_failures'])}"
    )
    if "reconstruction/stop_exact_rate" in metrics:
        message += (
            ", stop exact/early/late: "
            f"{metrics['reconstruction/stop_exact_rate']:.0%}/"
            f"{metrics['reconstruction/stop_early_rate']:.0%}/"
            f"{metrics['reconstruction/stop_late_rate']:.0%}, "
            f"len abs err: {metrics['reconstruction/gen_len_abs_err_mean']:.1f}"
        )
        if "reconstruction/stop_no_stop_rate" in metrics:
            message += (
                f", no stop: {metrics['reconstruction/stop_no_stop_rate']:.0%}"
            )
    if "reconstruction/psnr_mean_valid" in metrics:
        message += (
            f", PSNR: {metrics['reconstruction/psnr_mean_valid']:.2f}, "
            f"SSIM: {metrics['reconstruction/ssim_mean_valid']:.4f}"
        )
    return message


@torch.no_grad()
def validate(
    fabric: Fabric,
    model: nn.Module,
    val_dataloader: DataLoader,
    max_iters: int,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Validate aggregate loss plus byte-domain AR/FIM and EOS metrics."""
    fabric.barrier()
    if verbose:
        fabric.print("Validating ...")
    model.eval()

    losses = []
    task_regions = {"ar": REGION_TARGET, "fim": REGION_BRIDGE}
    task_loss_sum = {name: 0.0 for name in task_regions}
    task_tok_count = {name: 0 for name in task_regions}
    eos_probability_sum = {name: 0.0 for name in task_regions}
    eos_rank_sum = {name: 0.0 for name in task_regions}
    eos_count = {name: 0 for name in task_regions}
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        model_inputs, targets = get_model_inputs_and_targets(
            batch, model.max_seq_length
        )
        logits = model(**model_inputs)
        losses.append(chunked_cross_entropy(logits, targets))

        region_ids = (
            batch.get("target_region_ids")
            if isinstance(batch, dict)
            else None
        )
        if region_ids is None:
            region_ids = model_inputs.get("region_ids")
        elif isinstance(region_ids, torch.Tensor):
            region_ids = region_ids[:, : model.max_seq_length]
        if region_ids is None:
            continue
        for name, region in task_regions.items():
            masked = torch.where(
                region_ids == region,
                targets,
                torch.full_like(targets, IGNORE_INDEX),
            )
            count = int((masked != IGNORE_INDEX).sum())
            if count > 0:
                task_loss_sum[name] += (
                    float(chunked_cross_entropy(logits, masked)) * count
                )
                task_tok_count[name] += count
            eos_mask = (targets == SEQ_EOS_ID) & (region_ids == region)
            if eos_mask.any():
                eos_logits = logits[eos_mask].float()
                eos_scores = eos_logits[:, SEQ_EOS_ID]
                eos_probability_sum[name] += float(
                    torch.softmax(eos_logits, dim=-1)[:, SEQ_EOS_ID].sum()
                )
                eos_rank_sum[name] += float(
                    (eos_logits > eos_scores.unsqueeze(-1))
                    .sum(dim=-1)
                    .add(1)
                    .sum()
                )
                eos_count[name] += int(eos_mask.sum())

    val_loss = torch.stack(losses).mean()
    task_losses = {
        name: task_loss_sum[name] / task_tok_count[name]
        for name in task_regions
        if task_tok_count[name] > 0
    }
    eos_metrics = {}
    for name in task_regions:
        if eos_count[name] > 0:
            eos_metrics[f"val_eos_probability_{name}"] = (
                eos_probability_sum[name] / eos_count[name]
            )
            eos_metrics[f"val_eos_rank_{name}"] = (
                eos_rank_sum[name] / eos_count[name]
            )
    model.train()
    fabric.barrier()
    return val_loss, task_losses, eos_metrics
