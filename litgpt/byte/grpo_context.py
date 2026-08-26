"""Deterministic just-in-time context selection for online byte GRPO."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from torch.utils.data import Subset

from litgpt.byte.data import ByteStreamWindowDataset
from litgpt.byte.free_run_eval import (
    FreeRunEvalConfig,
    FreeRunSample,
    build_free_run_sample,
)
from litgpt.byte.grpo import GRPOConfig
from litgpt.byte.reconstruction import (
    ReconstructionSample,
    build_window_fim_sample,
)


GRPO_CONTEXT_TASKS = {"ar", "fim"}
MAX_CONTEXT_ATTEMPTS = 32


@dataclass(frozen=True)
class GRPOContextSelection:
    """One materialized GRPO context plus enough metadata to reproduce it."""

    task: str
    draw_index: int
    dataset_index: int
    attempt: int
    sample: FreeRunSample | ReconstructionSample
    hole_spec: tuple[int, int, int, int] | None = None


@dataclass
class OnlineGRPOContextSampler:
    """Visit training windows in deterministic shuffled epochs.

    Prompt tensors are built only for the selected window.  The order is a pure
    function of ``(seed, task, draw_index)``, so a resumed run reconstructs the
    same context without serializing sampler state.  AR and FIM use independent
    permutations; every revisit of a FIM window draws a new deterministic hole.
    """

    dataset: ByteStreamWindowDataset
    indices: tuple[int, ...]
    config: GRPOConfig
    _order_cache: dict[str, tuple[int, tuple[int, ...]]] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_dataset(
        cls,
        dataset: object,
        config: GRPOConfig,
    ) -> "OnlineGRPOContextSampler":
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            indices = tuple(int(idx) for idx in dataset.indices)
        else:
            base_dataset = dataset
            indices = (
                tuple(range(len(base_dataset)))
                if hasattr(base_dataset, "__len__")
                else ()
            )
        if not isinstance(base_dataset, ByteStreamWindowDataset):
            raise ValueError(
                "Online GRPO context sampling requires ByteStreamWindowDataset"
            )
        if not indices:
            raise ValueError("Online GRPO context sampling received an empty split")
        return cls(base_dataset, indices, config)

    def _order(self, task: str, epoch: int) -> tuple[int, ...]:
        cached = self._order_cache.get(task)
        if cached is not None and cached[0] == epoch:
            return cached[1]
        task_offset = 0 if task == "ar" else 1_000_000_007
        rng = random.Random(
            self.config.context_seed + task_offset + epoch * 2_000_000_011
        )
        order = list(self.indices)
        rng.shuffle(order)
        frozen = tuple(order)
        self._order_cache[task] = (epoch, frozen)
        return frozen

    def dataset_index(
        self,
        task: str,
        draw_index: int,
        attempt: int = 0,
        stride: int = 1,
    ) -> int:
        """Return the deterministic window index for one draw/attempt."""
        if task not in GRPO_CONTEXT_TASKS:
            raise ValueError(f"Unknown GRPO context task: {task}")
        if draw_index < 0 or attempt < 0 or stride <= 0:
            raise ValueError(
                "GRPO context draw index/attempt must be non-negative and stride positive"
            )
        position = draw_index + attempt * stride
        epoch, slot = divmod(position, len(self.indices))
        return self._order(task, epoch)[slot]

    def _fim_hole_spec(
        self,
        dataset_index: int,
        draw_index: int,
    ) -> tuple[int, int, int, int] | None:
        # Do not use Python's process-randomized hash().  This arithmetic seed
        # is stable across processes and resume, while draw_index makes a new
        # hole when the same window is revisited in a later shuffled epoch.
        hole_seed = (
            self.config.context_seed
            + 3_000_000_019 * (draw_index + 1)
            + 5_000_000_033 * (dataset_index + 1)
        )
        return self.dataset.draw_fim_hole_spec(
            dataset_index, random.Random(hole_seed)
        )

    def sample(
        self,
        task: str,
        draw_index: int,
        *,
        stride: int = 1,
    ) -> GRPOContextSelection | None:
        """Materialize one eligible context without retaining its prompt tensors."""
        attempts = min(MAX_CONTEXT_ATTEMPTS, len(self.indices))
        for attempt in range(attempts):
            dataset_index = self.dataset_index(
                task, draw_index, attempt, stride
            )
            if task == "ar":
                sample = build_free_run_sample(
                    self.dataset,
                    dataset_index,
                    FreeRunEvalConfig(
                        interval=1,
                        num_clips=1,
                        prefix_frames=self.config.ar_prefix_frames,
                        cont_frames=self.config.ar_cont_frames,
                        slice_layout=self.config.ar_slice_layout,
                    ),
                )
                hole_spec = None
            else:
                hole_spec = self._fim_hole_spec(dataset_index, draw_index)
                sample = (
                    None
                    if hole_spec is None
                    else build_window_fim_sample(
                        self.dataset,
                        dataset_index,
                        hole_spec,
                        self.config.max_target_bytes,
                        force_eos_stopping=self.config.learned_eos,
                    )
                )
            if sample is not None:
                return GRPOContextSelection(
                    task=task,
                    draw_index=draw_index,
                    dataset_index=dataset_index,
                    attempt=attempt,
                    sample=sample,
                    hole_spec=hole_spec,
                )
        return None
