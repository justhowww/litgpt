"""Opt-in, one-update CUDA memory evidence for small-sample comparisons.

The JSON counts identifiable persistent tensor storage. The PyTorch allocator
snapshot retains stack traces and allocation history for transient FSDP,
activation, and kernel buffers that cannot be inferred from phase deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _storage_bytes(tensors: Any) -> int:
    """Count distinct local CUDA storages, including aliased tensor views once."""

    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            continue
        try:
            storage = tensor.untyped_storage()
            size = storage.nbytes()
            key = (str(tensor.device), storage.data_ptr())
        except (AttributeError, RuntimeError, NotImplementedError, TypeError):
            # Some tensor wrappers do not expose storage. The reported category
            # is therefore an identified lower bound, never an exact partition.
            continue
        if size and key not in seen:
            seen.add(key)
            total += size
    return total


def _optimizer_tensors(optimizer: torch.optim.Optimizer):
    state_map = getattr(optimizer, "state", None)
    if state_map is None and hasattr(optimizer, "optimizer"):
        state_map = getattr(optimizer.optimizer, "state", None)
    for state in (state_map or {}).values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                yield value


class SmallSampleMemoryProfiler:
    """Profile one complete optimizer update on each distributed rank."""

    def __init__(self, *, step: int, out_dir: Path, rank: int) -> None:
        if step <= 0 or not torch.cuda.is_available():
            raise ValueError("A positive profile step and CUDA are required")
        self.step = step
        self.rank = rank
        self.out_dir = out_dir / "memory_profile" / f"step-{step:08d}"
        self.active = False
        self.done = False
        self.microbatch = 0
        self.phases: list[dict[str, float | str | int]] = []

    def _failure(self, error: Exception) -> None:
        # Profiling is diagnostic; an unavailable snapshot API or full disk must
        # not destroy a model run that may have no checkpoint yet.
        message = f"{type(error).__name__}: {error}"
        if self.active:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except (AttributeError, RuntimeError, TypeError):
                pass
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / f"rank-{self.rank}-error.json").write_text(
                json.dumps(
                    {"step": self.step, "rank": self.rank, "error": message}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        print(f"[memory-profile] rank={self.rank} unavailable: {message}", flush=True)
        self.active = False
        self.done = True

    def _sample(self, phase: str) -> None:
        torch.cuda.synchronize()
        reading: dict[str, float | str | int] = {
            "phase": phase,
            "microbatch": self.microbatch,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }
        device_used = getattr(torch.cuda, "device_memory_used", None)
        if device_used is not None:
            # Includes CUDA-context/NCCL memory and potentially other processes;
            # the difference from reserved is only a diagnostic, not attribution.
            try:
                reading["total_device_used_gb"] = (
                    device_used(torch.cuda.current_device()) / 1e9
                )
            except (RuntimeError, TypeError):
                pass
        self.phases.append(reading)

    def start_if_due(self, next_step: int) -> None:
        if self.done or next_step != self.step:
            return
        if self.active:
            self.microbatch += 1
            return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._record_memory_history(max_entries=100_000)
            self.active = True
            self.microbatch = 1
            self._sample("before_first_forward")
        except (AttributeError, RuntimeError, TypeError, OSError) as error:
            self._failure(error)

    def sample(self, phase: str) -> None:
        if self.active:
            self._sample(phase)

    def finish(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        if not self.active:
            return
        self._sample("after_zero_grad")
        parameters = list(model.parameters())
        persistent = {
            "local_model_parameter_storage_gb": _storage_bytes(parameters) / 1e9,
            "local_gradient_storage_gb": _storage_bytes(
                parameter.grad for parameter in parameters if parameter.grad is not None
            )
            / 1e9,
            "local_optimizer_state_storage_gb": _storage_bytes(
                _optimizer_tensors(optimizer)
            )
            / 1e9,
        }
        identified_gb = sum(persistent.values())
        end_allocated_gb = float(self.phases[-1]["allocated_gb"])
        report = {
            "step": self.step,
            "rank": self.rank,
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_capacity_gb": torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
            / 1e9,
            "phases": self.phases,
            "persistent_identified_storage": persistent,
            "after_zero_grad_unidentified_allocated_gb": max(
                0.0, end_allocated_gb - identified_gb
            ),
            "run_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "run_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "interpretation": (
                "Persistent storage is a local lower bound, not an exact GPU-memory "
                "partition; unidentified memory is a residual, not an activation "
                "estimate. Phase readings are endpoints, not phase peaks. Inspect the "
                "allocation-history snapshot for activations, FSDP all-gathers, and "
                "temporary buffers. Non-PyTorch allocations such as NCCL are absent "
                "from the PyTorch allocator snapshot."
            ),
        }
        snapshot = self.out_dir / f"rank-{self.rank}.pickle"
        summary = self.out_dir / f"rank-{self.rank}.json"
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot))
            summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except (AttributeError, RuntimeError, TypeError, OSError) as error:
            self._failure(error)
        finally:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except (AttributeError, RuntimeError, TypeError):
                pass
            self.active = False
            self.done = True
        if summary.is_file():
            print(
                f"[memory-profile] rank={self.rank} report={summary} snapshot={snapshot}",
                flush=True,
            )
