"""Incremental MEGABYTE inference over byte/control-token prompts.

The global Transformer advances once per completed patch. The shared local
Transformer then predicts the bytes inside the next patch autoregressively.
Codec constraints remain byte-level and are intentionally applied by callers
between :meth:`next_logits` and :meth:`append`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from litgpt.byte.data import PAD_ID, REGION_PAD, patch_byte_sample


@dataclass
class _GeneratedPatch:
    tokens: list[int]
    region_ids: list[int]
    offset_ids: list[int]
    generated: list[bool]
    global_position: int


def megabyte_prompt_patches(
    prompt_ids: Tensor,
    region_ids: Tensor,
    offset_ids: Tensor,
    patch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Left-pad one-byte prompts and reshape them into global input patches."""
    if patch_size <= 1:
        raise ValueError("patch_size must be greater than one")
    if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1:
        raise ValueError("MEGABYTE inference currently requires a [1,L] prompt")
    if region_ids.shape != prompt_ids.shape or offset_ids.shape != prompt_ids.shape:
        raise ValueError("prompt ids, region ids, and offset ids must have equal shapes")
    pad = (-prompt_ids.size(1)) % patch_size
    if pad:
        def _left_pad(tensor: Tensor, value: int) -> Tensor:
            prefix = torch.full(
                (1, pad),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return torch.cat((prefix, tensor), dim=1)

        prompt_ids = _left_pad(prompt_ids, PAD_ID)
        region_ids = _left_pad(region_ids, REGION_PAD)
        offset_ids = _left_pad(offset_ids, 0)
    positions = prompt_ids.size(1) // patch_size
    return (
        prompt_ids.view(1, positions, patch_size),
        region_ids.view(1, positions, patch_size),
        offset_ids.view(1, positions, patch_size),
    )


def megabyte_max_new_bytes(
    raw: nn.Module,
    prompt_token_count: int,
    supervision_start: int | None = None,
) -> int:
    """Maximum local bytes reachable without exceeding the global KV cache.

    ``supervision_start`` is the raw position of the first training target. When
    provided, bytes after that position are known teacher-forced target bytes,
    not a newly left-padded prompt. Accounting for them preserves the training
    patch phase during continuation.
    """
    patch_size = int(raw.config.byte_patch_size)
    if patch_size <= 1:
        return max(0, int(raw.max_seq_length) - prompt_token_count + 1)
    if supervision_start is None:
        prompt_patches = (prompt_token_count + patch_size - 1) // patch_size
        current_bytes = 0
    else:
        if not 0 <= supervision_start < prompt_token_count:
            raise ValueError("supervision_start must index the raw prompt")
        prompt_patches = (supervision_start + 1 + patch_size - 1) // patch_size
        known_targets = prompt_token_count - supervision_start - 1
        prompt_patches += known_targets // patch_size
        current_bytes = known_targets % patch_size
    remaining_output_patches = int(raw.max_seq_length) - prompt_patches + 1
    return max(0, remaining_output_patches * patch_size - current_bytes)


def megabyte_teacher_forced_sample(
    input_ids: Tensor,
    labels: Tensor,
    region_ids: Tensor,
    offset_ids: Tensor,
    patch_size: int,
) -> dict[str, Tensor]:
    """Patch one exact training-style shifted sample for offline evaluation."""
    if input_ids.dim() != 1:
        raise ValueError("teacher-forced patch conversion expects one sample")
    sample = patch_byte_sample(
        {
            "input_ids": input_ids,
            "labels": labels,
            "region_ids": region_ids,
            "offset_ids": offset_ids,
            "token_counts": {
                "raw": int(input_ids.numel()),
                "raw_plus_prompt_template": int(input_ids.numel()),
            },
            "sample_meta": {},
        },
        patch_size,
    )
    return {
        key: sample[key].unsqueeze(0)
        for key in (
            "input_ids",
            "labels",
            "region_ids",
            "offset_ids",
            "target_region_ids",
        )
    }


class MegabyteInference:
    """One-sample cached global decoder plus sequential local byte decoder."""

    def __init__(
        self,
        raw: nn.Module,
        prompt_ids: Tensor,
        region_ids: Tensor,
        offset_ids: Tensor,
        device: torch.device,
        *,
        supervision_start: int | None = None,
    ) -> None:
        patch_size = int(raw.config.byte_patch_size)
        if patch_size <= 1:
            raise ValueError("MegabyteInference requires byte_patch_size > 1")
        self.raw = raw
        self.device = device
        self.patch_size = patch_size
        if supervision_start is None:
            seed_end = prompt_ids.size(1)
        else:
            if not 0 <= supervision_start < prompt_ids.size(1):
                raise ValueError("supervision_start must index the raw prompt")
            seed_end = supervision_start + 1
        patched_ids, patched_regions, patched_offsets = megabyte_prompt_patches(
            prompt_ids[:, :seed_end],
            region_ids[:, :seed_end],
            offset_ids[:, :seed_end],
            patch_size,
        )
        known_ids = prompt_ids[:, seed_end:]
        known_regions = region_ids[:, seed_end:]
        known_offsets = offset_ids[:, seed_end:]
        complete_known = known_ids.size(1) // patch_size
        complete_bytes = complete_known * patch_size
        if complete_known:
            patched_ids = torch.cat(
                (
                    patched_ids,
                    known_ids[:, :complete_bytes].view(1, complete_known, patch_size),
                ),
                dim=1,
            )
            patched_regions = torch.cat(
                (
                    patched_regions,
                    known_regions[:, :complete_bytes].view(
                        1, complete_known, patch_size
                    ),
                ),
                dim=1,
            )
            patched_offsets = torch.cat(
                (
                    patched_offsets,
                    known_offsets[:, :complete_bytes].view(
                        1, complete_known, patch_size
                    ),
                ),
                dim=1,
            )
        self.prompt_patches = patched_ids.size(1)
        if self.prompt_patches > int(raw.max_seq_length):
            raise ValueError(
                f"prompt requires {self.prompt_patches} global positions but "
                f"the model supports {raw.max_seq_length}"
            )

        cache_dtype = (
            torch.bfloat16
            if device.type == "cuda"
            else next(raw.parameters()).dtype
        )
        raw.set_kv_cache(
            batch_size=1,
            max_seq_length=raw.max_seq_length,
            device=device,
            dtype=cache_dtype,
        )
        self._closed = False
        self._current_tokens = [int(x) for x in known_ids[0, complete_bytes:].tolist()]
        self._current_regions = [
            int(x) for x in known_regions[0, complete_bytes:].tolist()
        ]
        self._current_offsets = [
            int(x) for x in known_offsets[0, complete_bytes:].tolist()
        ]
        self._current_generated = [False] * len(self._current_tokens)
        self._completed: list[_GeneratedPatch] = []
        self._dirty_positions: set[int] = set()
        self._generated_count = 0
        self._next_global_position = self.prompt_patches
        try:
            with self._autocast():
                global_output = raw.megabyte_global_forward(
                    patched_ids,
                    input_pos=torch.arange(
                        self.prompt_patches, device=device, dtype=torch.long
                    ),
                    input_pos_maxp1=self.prompt_patches,
                    region_ids=patched_regions,
                    offset_ids=patched_offsets,
                )
            self._global_output = global_output[:, -1]
        except Exception:
            raw.clear_kv_cache()
            self._closed = True
            raise

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        )

    @property
    def generated_bytes(self) -> int:
        return self._generated_count

    @property
    def max_new_bytes(self) -> int:
        remaining_output_patches = (
            int(self.raw.max_seq_length) - self._next_global_position + 1
        )
        return max(
            0,
            remaining_output_patches * self.patch_size
            - len(self._current_tokens),
        )

    def _forward_patch(self, patch: _GeneratedPatch) -> Tensor:
        position = patch.global_position
        with self._autocast():
            output = self.raw.megabyte_global_forward(
                torch.tensor(
                    [[patch.tokens]], device=self.device, dtype=torch.long
                ),
                input_pos=torch.tensor([position], device=self.device),
                input_pos_maxp1=position + 1,
                region_ids=torch.tensor(
                    [[patch.region_ids]], device=self.device, dtype=torch.long
                ),
                offset_ids=torch.tensor(
                    [[patch.offset_ids]], device=self.device, dtype=torch.long
                ),
            )
        return output[:, -1]

    def _refresh_dirty_cache(self) -> None:
        if not self._dirty_positions:
            return
        latest_position = (
            self._completed[-1].global_position if self._completed else None
        )
        by_position = {patch.global_position: patch for patch in self._completed}
        for position in sorted(self._dirty_positions):
            output = self._forward_patch(by_position[position])
            if position == latest_position:
                self._global_output = output
        self._dirty_positions.clear()

    def _commit_full_patch(self) -> None:
        if len(self._current_tokens) != self.patch_size:
            return
        self._refresh_dirty_cache()
        position = self._next_global_position
        if position >= int(self.raw.max_seq_length):
            raise RuntimeError("MEGABYTE global context exhausted")
        patch = _GeneratedPatch(
            tokens=self._current_tokens,
            region_ids=self._current_regions,
            offset_ids=self._current_offsets,
            generated=self._current_generated,
            global_position=position,
        )
        self._global_output = self._forward_patch(patch)
        self._completed.append(patch)
        self._next_global_position += 1
        self._current_tokens = []
        self._current_regions = []
        self._current_offsets = []
        self._current_generated = []

    @torch.inference_mode()
    def next_logits(self) -> Tensor:
        """Return ``[1,V]`` logits for the next local byte/control token."""
        if self._closed:
            raise RuntimeError("MEGABYTE inference state is closed")
        if len(self._current_tokens) == self.patch_size:
            self._commit_full_patch()
        else:
            self._refresh_dirty_cache()
        previous = torch.tensor(
            [self._current_tokens], device=self.device, dtype=torch.long
        )
        with self._autocast():
            return self.raw.megabyte_local_next_logits(
                self._global_output, previous
            )

    def append(self, token: int, region_id: int, offset_id: int) -> None:
        """Commit one selected byte to the current local patch."""
        if self._closed:
            raise RuntimeError("MEGABYTE inference state is closed")
        if len(self._current_tokens) >= self.patch_size:
            raise RuntimeError("call next_logits before appending another patch")
        self._current_tokens.append(int(token))
        self._current_regions.append(int(region_id))
        self._current_offsets.append(int(offset_id))
        self._current_generated.append(True)
        self._generated_count += 1

    def rewrite_recent_metadata(
        self,
        count: int,
        *,
        region_ids: list[int],
        offset_ids: list[int],
    ) -> None:
        """Correct metadata after a start code becomes observable.

        A start code can cross a patch boundary. If that changes the most recent
        completed patch, its cached global K/V entry is overwritten before the
        next local prediction.
        """
        if count <= 0:
            return
        if len(region_ids) != count or len(offset_ids) != count:
            raise ValueError("metadata rewrite lengths must equal count")
        locations: list[tuple[_GeneratedPatch | None, int]] = []
        for index in range(len(self._current_tokens) - 1, -1, -1):
            if self._current_generated[index]:
                locations.append((None, index))
        for patch in reversed(self._completed):
            for index in range(self.patch_size - 1, -1, -1):
                if patch.generated[index]:
                    locations.append((patch, index))
        if count > len(locations):
            raise ValueError("cannot rewrite metadata before generated output")

        for source_index, (patch, byte_index) in enumerate(reversed(locations[:count])):
            if patch is None:
                self._current_regions[byte_index] = int(region_ids[source_index])
                self._current_offsets[byte_index] = int(offset_ids[source_index])
            else:
                patch.region_ids[byte_index] = int(region_ids[source_index])
                patch.offset_ids[byte_index] = int(offset_ids[source_index])
                self._dirty_positions.add(patch.global_position)

    def close(self) -> None:
        if not self._closed:
            self.raw.clear_kv_cache()
            self._closed = True

    def __enter__(self) -> "MegabyteInference":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
