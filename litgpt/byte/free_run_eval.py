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
def _forced_byte_mask(next_logits: Tensor, high_bits: int, n_high: int) -> Tensor:
    """Mask a [1, 256] byte-logit row to only bytes whose top ``n_high`` bits equal
    ``high_bits`` (an int of those bits, MSB-first). The model still freely chooses
    the remaining ``8 - n_high`` low bits. Used to splice a forced codeword prefix
    into a generated byte during the rescue test (see free_run_rollout forced_bits).
    """
    masked = torch.full_like(next_logits, float("-inf"))
    base = high_bits << (8 - n_high)
    for low in range(1 << (8 - n_high)):
        masked[0, base | low] = next_logits[0, base | low]
    return masked


def free_run_rollout(
    raw: nn.Module,
    prompt_ids: Tensor,
    region_ids: Tensor,
    offset_ids: Tensor,
    device,
    cont_frames: int,
    max_gen: int,
    temperature: float,
    forced_bits: list[int] | None = None,
    init_offset: int = 0,
) -> tuple[bytes, int]:
    """Shared free-run byte rollout, used by BOTH the in-loop val_freerun probe
    (``_generate``) and the standalone eval (``generate_continuation``) so their
    generation is guaranteed identical -- do not fork this loop.

    Prefill the ``[1, L]`` prompt (already batched + on device), then sample bytes
    to ``max_gen``, stopping after ``cont_frames`` complete frames
    (``cont_frames + 1`` start codes). Per-NAL offset resets on detected start
    codes; region is REGION_TARGET for every generated byte. ``temperature <= 0``
    is greedy. Returns (generated bytes with the partial next frame dropped,
    start-code count). Sets and clears the KV cache internally.

    ``forced_bits`` (rescue test): a bit sequence (MSB-first) forced into the FIRST
    generated byte(s) before free generation resumes -- used to splice a chosen legal
    codeword (plus the already-fixed high bits of its boundary byte) into the stream.
    Each generated byte consumes up to 8 forced bits as its high bits; the model fills
    the rest. When exhausted, generation is unconstrained. Default None = no forcing.
    """
    if max_gen <= 0:
        return b"", 0
    prompt_len = prompt_ids.size(1)
    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw.parameters()).dtype
    raw.set_kv_cache(batch_size=1, max_seq_length=raw.max_seq_length, device=device, dtype=cache_dtype)
    generated: list[int] = []
    start_codes = 0
    gen_offset = init_offset  # nonzero when the rescue resumes mid-NAL (not a fresh frame)
    forced = list(forced_bits) if forced_bits else []
    forced_used = 0
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = raw(
                prompt_ids,
                input_pos=torch.arange(prompt_len, device=device),
                input_pos_maxp1=prompt_len,
                region_ids=region_ids,
                offset_ids=offset_ids,
            )
        for step in range(max_gen):
            next_logits = logits[:, -1, :BYTE_VOCAB_SIZE]
            if forced_used < len(forced):
                n_high = min(8, len(forced) - forced_used)
                high_bits = 0
                for j in range(n_high):
                    high_bits = (high_bits << 1) | forced[forced_used + j]
                next_logits = _forced_byte_mask(next_logits, high_bits, n_high)
                forced_used += n_high
            if temperature <= 0:
                token = int(next_logits.argmax(dim=-1))
            else:
                probs = torch.softmax(next_logits.float() / temperature, dim=-1)
                token = int(torch.multinomial(probs, 1))
            generated.append(token)

            is_start = len(generated) >= 3 and tuple(generated[-3:]) == START_CODE
            token_offset = gen_offset
            if is_start:
                start_codes += 1
                if start_codes == cont_frames + 1:
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
    return bytes(generated), start_codes


@torch.inference_mode()
def _generate(raw: nn.Module, sample: FreeRunSample, device, config: FreeRunEvalConfig) -> bytes:
    """Free-run rollout from the clean prefix; thin wrapper over free_run_rollout."""
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset = sample.prompt_offset_ids.to(device).unsqueeze(0)
    max_gen = min(
        int(sample.gt_cont_bytes * config.max_gen_multiple) + 512,
        int(raw.max_seq_length) - len(sample.prefix_bytes) - 1,
    )
    generated, _start_codes = free_run_rollout(
        raw, prompt, region, offset, device, sample.cont_frames, max_gen, config.temperature
    )
    return generated


@dataclass(frozen=True)
class SurvivalResult:
    survival: int
    valid_cont: int
    start_codes: int
    desync_region: str | None  # syntax element at the desync (fine, e.g. coeff_token)
    desync_category: str | None  # its category (coarse, for exposure-normalized density)
    desync_reason: str | None  # parser exception class -> failure mechanism
    exposure: dict  # category -> survived generated bytes (density denominator)


def _desync_info(nal) -> tuple[str | None, str | None, str | None]:
    """(element_name, category, reason_kind) at a desync. Element/category come from
    the last recorded span at or before ``desync_bit`` (the break is in the element
    right after it); reason_kind is the parser's exception class (ValueError =
    codeword miss, KeyError/IndexError = out-of-range value). None where unavailable.
    """
    if nal is None:
        return None, None, None
    reason_kind = nal.reason_kind
    if not nal.spans:
        return None, None, reason_kind
    region = nal.spans[-1]
    dbit = nal.desync_bit
    if dbit is not None:
        prior = [s for s in nal.spans if s.bit_start <= dbit]
        if prior:
            region = prior[-1]
    return region.name, region.category.value, reason_kind


def _exposure_by_category(spans: list, lo: int, hi: int) -> dict[str, int]:
    """Bytes per H.264 category within [lo, hi) -- the survived generated region.
    First-writer-wins per byte (same approximation as the accuracy buckets). This is
    the denominator for desync density: desyncs / byte-of-exposure isolates a field's
    intrinsic brittleness from its raw byte-share.
    """
    if hi <= lo:
        return {}
    owner: list[str | None] = [None] * (hi - lo)
    for span in spans:
        b0 = max(span.byte_start, lo)
        b1 = min(span.byte_end, hi)
        for off in range(b0, b1):
            i = off - lo
            if owner[i] is None:
                owner[i] = span.category.value
    exposure: dict[str, int] = {}
    for cat in owner:
        if cat is not None:
            exposure[cat] = exposure.get(cat, 0) + 1
    return exposure


def _survival_and_validity(
    prefix_bytes: bytes, generated: bytes, cont_frames: int
) -> SurvivalResult:
    """Parse prefix+generated; return a SurvivalResult.

    survival = generated bytes before the first CAVLC desync at or after the prefix
    boundary. A desync counts even when it lands in a NAL that *began* in the prefix
    (the absorbed-garbage case, no leading start code). If nothing desyncs but the
    model never emitted a generated start code, survival is 0 (it never began a valid
    frame); only a started-and-clean budget gives survival = len(generated).

    desync_region/desync_category/desync_reason describe *where* and *how* the first
    desync happened; exposure holds survived generated bytes per category so the
    caller can form desync-per-byte density (brittleness) per field.
    """
    n_prefix = len(prefix_bytes)
    parsed = HS.parse_stream(prefix_bytes + generated, parse_slice_data=True)
    valid_cont = 0
    start_codes = 0
    first_desync: int | None = None  # earliest absolute desync offset >= n_prefix
    desync_nal = None
    for nal in parsed.nals:
        sc = nal.nal.start_code_start
        if sc >= n_prefix:
            start_codes += 1
        # Attribute a desync that lands in the generated region even if the NAL began
        # in the prefix (absorbed-garbage case) -- do NOT skip prefix-started NALs here.
        if nal.status != HS.ParseStatus.OK:
            db = nal.desync_byte if nal.desync_byte is not None else sc
            if db >= n_prefix and (first_desync is None or db < first_desync):
                first_desync = db
                desync_nal = nal
        # A valid generated continuation frame must begin in the generated region.
        if (
            nal.nal.nal_type in VCL_NAL_TYPES
            and sc >= n_prefix
            and nal.status == HS.ParseStatus.OK
        ):
            valid_cont += 1

    if first_desync is not None:
        survival = first_desync - n_prefix
    elif start_codes == 0:
        survival = 0  # model never emitted a generated frame boundary
    else:
        survival = len(generated)

    region, category, reason = _desync_info(desync_nal)
    exposure = _exposure_by_category(parsed.all_spans(), n_prefix, n_prefix + survival)
    return SurvivalResult(
        survival, min(valid_cont, cont_frames), start_codes, region, category, reason, exposure
    )


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
            r = _survival_and_validity(
                sample.prefix_bytes, generated, sample.cont_frames
            )
            survivals.append(r.survival)
            valids.append(r.valid_cont)
            start_codes_list.append(r.start_codes)
            full_valid += int(r.valid_cont >= sample.cont_frames)
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
