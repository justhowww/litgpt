"""Free-running (H0 generation) validation probe for the window AR objective.

val CE is teacher-forced; it does not determine free-run validity (exposure-bias
drift + the illegal-token tail). This probe measures the thing that actually
decides H0: give the model a clean SPS-anchored prefix, free-run generate, parse
the result with the (ffmpeg-free) CAVLC parser, and report **survival-length**
(bytes generated before the first syntax desync) and **validity** (whether enough
continuation frames parse clean).

Survival-length is the continuous leading indicator: free-run survives ~1/eps
symbols where eps is the per-symbol illegal-token mass, so it moves *before* the
binary validity rate lifts off 0. Logged under ``val_freerun/*``.

Design notes:
- Prompts are built by truncating the window dataset's *own* item tensors at the
  prefix boundary, so region/offset/BOS conditioning is identical to training.
- Clips are filtered to prefixes that parse clean (SPS-anchored, self-contained);
  windows that start at an IDR without SPS/PPS (the _windows_for_video back-up
  gap) are skipped so the parser metric is meaningful.
- The rollout mirrors the validated continuation generator: KV-cache,
  inference_mode, per-NAL offset reset on detected start codes, external
  frame-count stop. Parsing is pure-Python (no ffmpeg), cheap enough for in-loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from litgpt.byte import h264_syntax as HS
from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    PARAMETER_SET_NAL_TYPES,
    REGION_TARGET,
    VCL_NAL_TYPES,
)
from litgpt.byte.reconstruction import _unwrap_model

START_CODE = (0, 0, 1)


@dataclass(frozen=True)
class FreeRunEvalConfig:
    """Configuration for the in-loop free-running generation probe."""

    interval: int = 0  # steps between probes; 0 disables
    num_clips: int = 8
    prefix_frames: int = 8
    cont_frames: int = 4
    max_gen_multiple: float = 2.0
    temperature: float = 1.0
    seed: int = 42

    @property
    def enabled(self) -> bool:
        return self.interval > 0 and self.num_clips > 0


@dataclass(frozen=True)
class FreeRunSample:
    """A fixed SPS-anchored continuation probe: clean prefix + generation budget."""

    h264_path: str
    prompt_ids: Tensor  # [BOS, prefix bytes...]
    prompt_region_ids: Tensor
    prompt_offset_ids: Tensor
    prefix_bytes: bytes
    gt_cont_bytes: int  # ground-truth continuation byte count (for the gen budget)
    cont_frames: int


def _window_byte_lengths(nals, start_nal: int, end_nal: int) -> list[int]:
    return [nals[i].end - nals[i].start for i in range(start_nal, end_nal)]


def prepare_free_run_samples(
    val_dataset, config: FreeRunEvalConfig
) -> list[FreeRunSample]:
    """Pick deterministic SPS-anchored continuation clips from the window dataset.

    Returns [] for any dataset that is not a ByteStreamWindowDataset (the probe is
    AR/window-only), mirroring select_reconstruction_samples' graceful no-op.
    """
    from litgpt.byte.data import ByteStreamWindowDataset

    base = val_dataset
    indices = None
    # Unwrap a torch Subset if present.
    if hasattr(val_dataset, "dataset") and hasattr(val_dataset, "indices"):
        base = val_dataset.dataset
        indices = list(val_dataset.indices)
    if not isinstance(base, ByteStreamWindowDataset):
        return []
    if indices is None:
        indices = list(range(len(base)))

    needed = config.prefix_frames + config.cont_frames
    samples: list[FreeRunSample] = []
    for idx in indices:
        win = base.samples[idx]
        if win.num_frames < needed:
            continue
        nals = base.nal_index[str(win.h264_path)]
        # Self-contained only: the window must start at an SPS so the parser has
        # parameter sets for the generated slices.
        if nals[win.start_nal].nal_type not in PARAMETER_SET_NAL_TYPES:
            continue

        # Byte offset of the end of the prefix_frames-th VCL frame, and of the
        # needed-th (for the GT continuation budget), within the window.
        lengths = _window_byte_lengths(nals, win.start_nal, win.end_nal)
        prefix_bytes_len = 0
        cont_bytes_len = 0
        vcl = 0
        cursor = 0
        for off, length in enumerate(lengths):
            cursor += length
            if nals[win.start_nal + off].nal_type in VCL_NAL_TYPES:
                vcl += 1
                if vcl == config.prefix_frames:
                    prefix_bytes_len = cursor
                if vcl == needed:
                    cont_bytes_len = cursor
                    break
        if prefix_bytes_len <= 0 or cont_bytes_len <= 0:
            continue

        item = base[idx]
        window_bytes = bytes(item["labels"].tolist())
        prefix_bytes = window_bytes[:prefix_bytes_len]
        # Confirm the prefix actually parses clean before trusting the metric.
        parsed_prefix = HS.parse_stream(prefix_bytes, parse_slice_data=True)
        if any(n.status != HS.ParseStatus.OK for n in parsed_prefix.nals):
            continue

        prompt_end = prefix_bytes_len + 1  # +1 for the BOS at position 0
        samples.append(
            FreeRunSample(
                h264_path=str(win.h264_path),
                prompt_ids=item["input_ids"][:prompt_end].clone(),
                prompt_region_ids=item["region_ids"][:prompt_end].clone(),
                prompt_offset_ids=item["offset_ids"][:prompt_end].clone(),
                prefix_bytes=prefix_bytes,
                gt_cont_bytes=cont_bytes_len - prefix_bytes_len,
                cont_frames=config.cont_frames,
            )
        )
        if len(samples) >= config.num_clips:
            break
    return samples


@torch.inference_mode()
def _generate(raw: nn.Module, sample: FreeRunSample, device, config: FreeRunEvalConfig) -> bytes:
    """Free-run rollout from the clean prefix; stop after cont_frames complete
    frames (cont_frames+1 start codes) or the byte budget."""
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset = sample.prompt_offset_ids.to(device).unsqueeze(0)
    prompt_len = prompt.size(1)
    max_gen = min(
        int(sample.gt_cont_bytes * config.max_gen_multiple) + 512,
        int(raw.max_seq_length) - len(sample.prefix_bytes) - 1,
    )
    if max_gen <= 0:
        return b""

    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw.parameters()).dtype
    raw.set_kv_cache(batch_size=1, max_seq_length=raw.max_seq_length, device=device, dtype=cache_dtype)
    generated: list[int] = []
    start_codes = 0
    gen_offset = 0
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = raw(
                prompt,
                input_pos=torch.arange(prompt_len, device=device),
                input_pos_maxp1=prompt_len,
                region_ids=region,
                offset_ids=offset,
            )
        for step in range(max_gen):
            next_logits = logits[:, -1, :BYTE_VOCAB_SIZE]
            if config.temperature <= 0:
                token = int(next_logits.argmax(dim=-1))
            else:
                probs = torch.softmax(next_logits.float() / config.temperature, dim=-1)
                token = int(torch.multinomial(probs, 1))
            generated.append(token)

            is_start = len(generated) >= 3 and tuple(generated[-3:]) == START_CODE
            token_offset = gen_offset
            if is_start:
                start_codes += 1
                if start_codes == sample.cont_frames + 1:
                    generated = generated[:-3]
                    break
                gen_offset = 3
            else:
                gen_offset = token_offset + 1
            if step == max_gen - 1:
                break
            position = prompt_len + step
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = raw(
                    torch.tensor([[token]], device=device),
                    input_pos=torch.tensor([position], device=device),
                    input_pos_maxp1=position + 1,
                    region_ids=torch.full((1, 1), REGION_TARGET, device=device, dtype=torch.long),
                    offset_ids=torch.full((1, 1), token_offset, device=device, dtype=torch.long),
                )
    finally:
        raw.clear_kv_cache()
    return bytes(generated)


def _survival_and_validity(prefix_bytes: bytes, generated: bytes, cont_frames: int) -> tuple[int, int, int]:
    """Parse prefix+generated; return (survival_bytes, valid_cont_frames, start_codes).

    survival_bytes = generated bytes before the first desync in a NAL that begins
    in the generated region (= len(generated) if the whole budget parses clean).
    """
    n_prefix = len(prefix_bytes)
    parsed = HS.parse_stream(prefix_bytes + generated, parse_slice_data=True)
    survival = len(generated)
    valid_cont = 0
    start_codes = 0
    for nal in parsed.nals:
        if nal.nal.start_code_start >= n_prefix:
            start_codes += 1
        if nal.nal.nal_type not in VCL_NAL_TYPES:
            continue
        if nal.nal.start_code_start < n_prefix:
            continue  # a prefix frame, not generated
        if nal.status == HS.ParseStatus.OK:
            valid_cont += 1
        else:
            desync_byte = nal.desync_byte if nal.desync_byte is not None else nal.nal.start_code_start
            survival = max(0, desync_byte - n_prefix)
            break
    return survival, min(valid_cont, cont_frames), start_codes


def run_free_run_eval(
    model: nn.Module,
    samples: list[FreeRunSample],
    device,
    config: FreeRunEvalConfig,
) -> dict[str, float]:
    """Free-run each fixed clip, parse, and aggregate survival/validity metrics."""
    if not samples:
        return {}
    raw = _unwrap_model(model)
    was_training = raw.training
    raw.eval()
    survivals: list[int] = []
    valids: list[int] = []
    start_codes_list: list[int] = []
    full_valid = 0
    gen_lens: list[int] = []
    try:
        for sample in samples:
            generated = _generate(raw, sample, device, config)
            gen_lens.append(len(generated))
            survival, valid_cont, start_codes = _survival_and_validity(
                sample.prefix_bytes, generated, sample.cont_frames
            )
            survivals.append(survival)
            valids.append(valid_cont)
            start_codes_list.append(start_codes)
            full_valid += int(valid_cont >= sample.cont_frames)
    finally:
        if was_training:
            raw.train()

    n = len(samples)
    return {
        "val_freerun/survival_bytes": sum(survivals) / n,
        "val_freerun/valid_cont_frames": sum(valids) / n,
        "val_freerun/full_continuation_rate": full_valid / n,
        "val_freerun/start_codes_emitted": sum(start_codes_list) / n,
        "val_freerun/gen_bytes": sum(gen_lens) / n,
        "val_freerun/num_clips": float(n),
    }
