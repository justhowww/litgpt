"""Training helpers for byte-domain language modeling.

These functions isolate byte batches, EOS calibration, task-aware validation,
and reconstruction logging from LitGPT's generic pretraining orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from lightning import Fabric
from torch.utils.data import DataLoader

from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    IGNORE_INDEX,
    REGION_BRIDGE,
    REGION_TARGET,
    SEQ_EOS_ID,
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
    run_reconstruction_probe,
    save_reconstruction_sample_manifest,
    select_reconstruction_samples,
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
    ) -> "ByteTrainingRuntime":
        """Select fixed probe/MRT contexts once before distributed wrapping."""
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
        self, fabric: Fabric, metrics: dict[str, float], step: int, iter_num: int
    ) -> None:
        fabric.log_dict(metrics, step=iter_num - 1)
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


def get_model_inputs_and_targets(
    batch, max_seq_length: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Adapt byte dict batches while preserving LitGPT's shifted text batches."""
    if isinstance(batch, dict):
        input_ids = batch["input_ids"][:, :max_seq_length].contiguous().long()
        targets = batch["labels"][:, :max_seq_length].contiguous().long()
        model_inputs = {"idx": input_ids}
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

        region_ids = model_inputs.get("region_ids")
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
