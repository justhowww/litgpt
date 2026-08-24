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

from litgpt.byte import h264_mask as HM
from litgpt.byte import h264_syntax as HS
from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    PARAMETER_SET_NAL_TYPES,
    REGION_META,
    REGION_TARGET,
    SEQ_EOS_ID,
    VCL_NAL_TYPES,
)
from litgpt.byte.megabyte_inference import (
    GeneratedCandidate,
    MegabyteInference,
    megabyte_max_new_bytes,
    megabyte_prompt_patches,
    sample_tokens,
)
from litgpt.byte.reconstruction import _count_real_frames, _unwrap_model

START_CODE = (0, 0, 1)


def _target_closed_vcl_nals(cont_frames: int, slice_layout: str) -> int:
    """Number of first_mb==0 VCL closures needed to end a rollout.

    A frame-layout VCL is itself the complete picture. In the legacy multi-slice
    layout, the next frame's first slice is the observable proof that the requested
    final frame ended, preserving the established AVC-LM stopping behavior.
    """
    if cont_frames <= 0:
        raise ValueError("cont_frames must be positive")
    HM.slice_max_mbs_for_layout(slice_layout)  # validate the public value
    return (
        cont_frames
        if slice_layout == HM.SLICE_LAYOUT_FRAME
        else cont_frames + 1
    )


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
    slice_layout: str = HM.SLICE_LAYOUT_MACROBLOCK

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
    # Absolute (file-relative) byte offsets and real-picture index, for
    # reward computation: splice a generated continuation in at
    # prefix_end_abs, restore the original tail from gt_end_abs onward, and
    # decode_frame the k-th continuation frame at frame_index_base + k
    # (real-picture index, matching decode_frame's ffmpeg -vf select
    # convention -- see reconstruction._count_real_frames).
    prefix_end_abs: int
    gt_end_abs: int
    frame_index_base: int


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

        # Byte offset of the end of the prefix_frames-th REAL frame, and of the
        # needed-th (for the GT continuation budget), within the window. Frame
        # boundaries are ground truth first_mb_in_slice == 0 (HS.slice_first_mb),
        # the SAME primitive free_run_rollout uses on generated bytes -- so this
        # in-loop probe and the standalone eval can never silently disagree on what
        # counts as a frame on a slice-max-mbs=1 corpus (one VCL NAL == one MB).
        lengths = _window_byte_lengths(nals, win.start_nal, win.end_nal)
        data = win.h264_path.read_bytes()
        prefix_bytes_len = 0
        cont_bytes_len = 0
        frames = 0
        cursor = 0
        ok = True
        for off, length in enumerate(lengths):
            nal = nals[win.start_nal + off]
            if nal.nal_type in VCL_NAL_TYPES:
                nal_bytes = data[nal.start + nal.start_code_len : nal.end]
                first_mb = HS.slice_first_mb(nal_bytes)
                if first_mb is None:
                    ok = False  # unparseable slice header -- skip this window
                    break
                if first_mb == 0:
                    frames += 1
                    if frames == config.prefix_frames + 1 and prefix_bytes_len == 0:
                        prefix_bytes_len = cursor  # end of the prior frame, exclusive
                    if frames == needed + 1:
                        cont_bytes_len = cursor
                        break
            cursor += length
        if not ok or prefix_bytes_len <= 0 or cont_bytes_len <= 0:
            continue

        # ar_item, not base[idx]: this probe is AR-only and needs labels to be the
        # window's raw bytes. Under p_fim > 0 base[idx] may return a FIM item whose
        # labels are IGNORE_INDEX outside the span, which is not a byte string.
        item = base.ar_item(idx)
        window_labels = item["labels"]
        if base.use_eos:
            if int(window_labels[-1]) != SEQ_EOS_ID:
                raise RuntimeError("window AR item is missing its configured SEQ_EOS")
            window_labels = window_labels[:-1]
        window_bytes = bytes(window_labels.tolist())
        prefix_bytes = window_bytes[:prefix_bytes_len]
        # Confirm the prefix actually parses clean before trusting the metric.
        parsed_prefix = HS.parse_stream(prefix_bytes, parse_slice_data=True)
        if any(n.status != HS.ParseStatus.OK for n in parsed_prefix.nals):
            continue

        window_start_abs = nals[win.start_nal].start
        frames_before_window = _count_real_frames(nals, data, win.start_nal)

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
                prefix_end_abs=window_start_abs + prefix_bytes_len,
                gt_end_abs=window_start_abs + cont_bytes_len,
                frame_index_base=frames_before_window + config.prefix_frames,
            )
        )
        if len(samples) >= config.num_clips:
            break
    return samples


@torch.inference_mode()
def _sample_token(
    logits_row: Tensor, temperature: float, top_k: int, top_p: float
) -> int:
    """Sample one byte from a [1, 256] logit row with optional temperature / top-k /
    top-p (nucleus) truncation. Order: temperature -> top-k -> top-p -> multinomial.

    AVC-LM / JPEG-LM sample at temperature 1.0 with top_k=50, top_p=0.9 -- this cuts
    the invalid tail (no tail-garbage) without collapsing to the argmax (greedy
    degenerates for these AR byte-LMs). temperature<=0 with no truncation = greedy.
    Any -inf entries (e.g. from forced_bits masking) stay excluded.
    """
    if temperature <= 0 and not top_k and not (0 < top_p < 1.0):
        return int(logits_row.argmax(dim=-1))
    logits = logits_row.float()
    if temperature > 0:
        logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(
            logits < kth, torch.full_like(logits, float("-inf")), logits
        )
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens whose cumulative prob BEFORE them already exceeds top_p
        # (keeps the first token that crosses the threshold -> at least one kept).
        remove_sorted = (cum - torch.softmax(sorted_logits, dim=-1)) > top_p
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = torch.where(remove, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))


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
    top_k: int = 0,
    top_p: float = 0.0,
    stop_pad_run: int = 0,
    constrain: bool = False,
    mask_residual_only: bool = False,
    prefix_bytes: bytes | None = None,
    trace: dict | None = None,
    slice_layout: str = HM.SLICE_LAYOUT_MACROBLOCK,
) -> tuple[bytes, int]:
    """Shared free-run byte rollout, used by BOTH the in-loop val_freerun probe
    (``_generate``) and the standalone eval (``generate_continuation``) so their
    generation is guaranteed identical -- do not fork this loop.

    Prefill the ``[1, L]`` prompt (already batched + on device), then sample bytes
    to ``max_gen``, stopping after ``cont_frames`` complete FRAMES -- not NALs. On
    a slice-max-mbs=1 corpus one VCL NAL is one macroblock, so frame boundaries are
    detected via HS.slice_first_mb (first_mb_in_slice == 0) on each closed VCL NAL,
    the same primitive used to index ground-truth frame boundaries -- generated and
    GT sides can never silently disagree on what counts as a frame. If a closed VCL
    NAL's first_mb_in_slice can't be determined (HS.slice_first_mb returns None --
    too short / malformed exp-Golomb / implausible value), that is treated as a
    desync and generation stops immediately rather than guessing or silently
    miscounting frames on garbage bytes. Per-NAL offset resets on detected start
    codes; region is REGION_TARGET for every generated byte. ``temperature <= 0``
    is greedy. Returns (generated bytes with the partial next frame dropped,
    start-code count). Sets and clears the KV cache internally.

    ``forced_bits`` (rescue test): a bit sequence (MSB-first) forced into the FIRST
    generated byte(s) before free generation resumes -- used to splice a chosen legal
    codeword (plus the already-fixed high bits of its boundary byte) into the stream.
    Each generated byte consumes up to 8 forced bits as its high bits; the model fills
    the rest. When exhausted, generation is unconstrained. Default None = no forcing.

    ``constrain`` (h264_mask): before sampling each byte, mask the logits to the bytes
    accepted by the supported H.264 NAL-header, slice-header, slice-data, and Annex-B
    automata. Requires ``prefix_bytes`` to seed parameter-set and picture state.
    ``mask_residual_only`` reproduces the old residual-only, fail-open mask for a
    controlled ablation. ``slice_layout`` is ``macroblock`` for AVC-LM's fixed
    one-MB slices or ``frame`` for one complete progressive picture per slice.
    An all-illegal full mask means the grammar boxed us in -> stop like any other
    desync.

    ``trace`` (diagnostics): an optional dict the rollout WRITES to and never reads --
    it cannot affect the generated bytes. Records ``stop_reason``: which of the exits
    below ended the run. The loop has six exits but a single return, so the reason is
    otherwise unrecoverable: a clean frame-target stop, a pad-run stop, a mask
    box-in and a first_mb desync stop can all yield byte strings that re-parse
    identically. Only the budget stop leaves a trace after the fact (a dangling start
    code), and even that is ambiguous. Default None = no tracing.
    """

    def _stop(reason: str) -> None:
        if trace is not None:
            trace["stop_reason"] = reason

    max_gen = min(
        max_gen,
        megabyte_max_new_bytes(
            raw,
            prompt_ids.size(1),
            supervision_start=(
                0 if int(raw.config.byte_patch_size) > 1 else None
            ),
        ),
    )
    if max_gen <= 0:
        _stop("no_budget")
        return b"", 0
    slice_max_mbs = HM.slice_max_mbs_for_layout(slice_layout)
    mask_state = None
    target_boundaries = _target_closed_vcl_nals(cont_frames, slice_layout)
    if constrain:
        mask_state = HM.MaskState(
            slice_max_mbs=slice_max_mbs,
            residual_only=mask_residual_only,
            fail_closed=not mask_residual_only,
        )
        for b in prefix_bytes or b"":
            HM.advance(mask_state, b)
        mask_state.generation_started = True
    prompt_len = prompt_ids.size(1)
    patched = int(raw.config.byte_patch_size) > 1
    megabyte = None
    if patched:
        megabyte = MegabyteInference(
            raw,
            prompt_ids,
            region_ids,
            offset_ids,
            device,
            supervision_start=0,
        )
    else:
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
    generated: list[int] = []
    start_codes = 0
    gen_offset = (
        init_offset  # nonzero when the rescue resumes mid-NAL (not a fresh frame)
    )
    forced = list(forced_bits) if forced_bits else []
    forced_used = 0
    prev_token = -1  # for stop_pad_run: length of the current identical-byte run
    run_len = 0
    current_region = REGION_TARGET  # region for the current NAL's bytes (fix (c))
    pending_header = False  # next non-start byte is a NAL header -> read its nal_type
    nal_start_idx: int | None = None  # index into `generated` of the current NAL's
    # header byte; None until the first start code
    frames_seen = 0  # count of closed VCL NALs with first_mb_in_slice == 0
    mask_argmax_rejected = 0
    mask_probability_mass_sum = 0.0
    mask_probability_mass_count = 0
    first_mask_intervention = None
    try:
        if megabyte is None:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = raw(
                    prompt_ids,
                    input_pos=torch.arange(prompt_len, device=device),
                    input_pos_maxp1=prompt_len,
                    region_ids=region_ids,
                    offset_ids=offset_ids,
                )
        for step in range(max_gen):
            next_logits = (
                megabyte.next_logits()[:, :BYTE_VOCAB_SIZE]
                if megabyte is not None
                else logits[:, -1, :BYTE_VOCAB_SIZE]
            )
            model_next_logits = next_logits
            forcing_this_byte = False
            if forced_used < len(forced):
                forcing_this_byte = True
                n_high = min(8, len(forced) - forced_used)
                high_bits = 0
                for j in range(n_high):
                    high_bits = (high_bits << 1) | forced[forced_used + j]
                next_logits = _forced_byte_mask(next_logits, high_bits, n_high)
                forced_used += n_high
            if mask_state is not None:
                allowed = HM.get_valid_byte_mask(mask_state)
                if not any(allowed):
                    if trace is not None:
                        trace["mask_failure_reason"] = mask_state.failure_reason
                    _stop("mask_boxed_in")
                    break  # grammar boxed us in -> desync stop
                if trace is not None:
                    raw_argmax = int(torch.argmax(model_next_logits, dim=-1).item())
                    if not allowed[raw_argmax]:
                        mask_argmax_rejected += 1
                        if first_mask_intervention is None:
                            auto = mask_state.automaton
                            first_mask_intervention = {
                                "generated_byte": step,
                                "nal_index": mask_state.nal_index,
                                "model_argmax": raw_argmax,
                                "stage": (
                                    "nal_header"
                                    if mask_state.expect_nal_header
                                    else getattr(auto, "stage", "unknown")
                                ),
                                "syntax": (
                                    "nal_header"
                                    if mask_state.expect_nal_header
                                    else getattr(auto, "ae_tag", "unknown")
                                ),
                            }
                    allowed_on_device = torch.tensor(
                        allowed, dtype=torch.bool, device=model_next_logits.device
                    )
                    log_total = torch.logsumexp(model_next_logits[0], dim=0)
                    log_allowed = torch.logsumexp(
                        model_next_logits[0, allowed_on_device], dim=0
                    )
                    mask_probability_mass_sum += float(
                        torch.exp(log_allowed - log_total).item()
                    )
                    mask_probability_mass_count += 1
                forbidden = [b for b in range(BYTE_VOCAB_SIZE) if not allowed[b]]
                if forbidden:
                    next_logits = next_logits.clone()
                    next_logits[0, forbidden] = float("-inf")
            # A forced syntax codeword and the online syntax mask are two separate
            # restrictions on the same byte.  Detect an empty intersection instead
            # of letting argmax silently select byte 0 from an all--inf vector.  In
            # the legal-branch experiment this explicitly exposes disagreement
            # between the recursive parser that chose the branch and the mask
            # automaton that constrains generation.
            if not torch.isfinite(next_logits).any():
                _stop("forced_mask_conflict" if forcing_this_byte else "mask_boxed_in")
                break
            token = _sample_token(next_logits, temperature, top_k, top_p)
            generated.append(token)
            if mask_state is not None:
                HM.advance(mask_state, token)

            # "Gave-up" stop: with no learned EOS, the model falls into a repetitive
            # padding attractor (e.g. rbsp_trailing) when it runs out of confident
            # content. A long run of one byte = finished; trim the run and stop so it
            # doesn't waste the budget on padding (which also inflates survival).
            if stop_pad_run:
                run_len = run_len + 1 if token == prev_token else 1
                prev_token = token
                if run_len >= stop_pad_run:
                    del generated[-stop_pad_run:]
                    _stop("pad_run")
                    break

            is_start = len(generated) >= 3 and tuple(generated[-3:]) == START_CODE
            if is_start:
                start_codes += 1
                if nal_start_idx is not None:
                    # The NAL that started at nal_start_idx just closed (everything
                    # since then, minus the 3 start-code bytes just appended for the
                    # NEXT NAL). Check whether it opened a new frame.
                    nal_bytes = bytes(generated[nal_start_idx : len(generated) - 3])
                    nal_type = (nal_bytes[0] & 0x1F) if nal_bytes else -1
                    if nal_type in VCL_NAL_TYPES:
                        first_mb = HS.slice_first_mb(nal_bytes)
                        if first_mb is None:
                            # Can't determine the frame boundary on this NAL's bytes
                            # (too short / malformed ue(v) / implausible value) --
                            # treat it as a desync and stop here rather than letting
                            # frame-counting silently drift past corrupted output.
                            generated = generated[:-3]
                            _stop("first_mb_unparseable")
                            break
                        if first_mb == 0:
                            frames_seen += 1
                            # In the AVC-LM layout, closing first_mb==0 only proves
                            # that a new multi-slice frame has started; wait for the
                            # next frame's first slice as before.  With one complete
                            # picture per slice, closing this VCL NAL completes the
                            # frame, so no look-ahead frame is needed.
                            if frames_seen == target_boundaries:
                                generated = generated[:-3]
                                _stop("frame_target")
                                break
                nal_start_idx = len(generated)  # header byte of the NAL about to start
            if step == max_gen - 1:
                _stop("budget")
                break
            position = prompt_len + step

            if is_start:
                # A start code just completed. Its bytes were fed with continuation
                # offsets; re-feed the whole start code with the correct per-NAL offsets
                # 0..sc_len-1 (fix (a)), using the TRUE start-code length -- 4 when a
                # leading zero_byte precedes 00 00 01, matching parse_annexb_nals -- rather
                # than a hard-coded 3 (fix (b)). KVCache.batched_index_copy_ overwrites the
                # cache slots at these (past) input_pos, so the header/mb_type that follow
                # attend to correctly-encoded start-code bytes.
                sc_len = (
                    4
                    if (len(generated) >= 4 and tuple(generated[-4:]) == (0, 0, 0, 1))
                    else 3
                )
                if megabyte is not None:
                    megabyte.append(token, REGION_TARGET, sc_len - 1)
                    megabyte.rewrite_recent_metadata(
                        sc_len,
                        region_ids=[REGION_TARGET] * sc_len,
                        offset_ids=list(range(sc_len)),
                    )
                    gen_offset = sc_len
                    current_region = REGION_TARGET
                    pending_header = True
                    continue
                base_pos = position - (sc_len - 1)
                for j in range(sc_len):
                    tok_j = generated[len(generated) - sc_len + j]
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        logits = raw(
                            torch.tensor([[tok_j]], device=device),
                            input_pos=torch.tensor([base_pos + j], device=device),
                            input_pos_maxp1=base_pos + j + 1,
                            region_ids=torch.full(
                                (1, 1), REGION_TARGET, device=device, dtype=torch.long
                            ),
                            offset_ids=torch.full(
                                (1, 1), j, device=device, dtype=torch.long
                            ),
                        )
                gen_offset = sc_len  # the NAL header sits at offset sc_len
                current_region = REGION_TARGET
                pending_header = True
                continue

            # Non-start byte. The first byte after a start code is the NAL header; read its
            # nal_type to set the NAL's region: META for parameter sets, TARGET for VCL,
            # instead of hard-coding TARGET everywhere (fix (c)).
            if pending_header:
                nal_type = token & 0x1F
                current_region = (
                    REGION_META
                    if nal_type in PARAMETER_SET_NAL_TYPES
                    else REGION_TARGET
                )
                pending_header = False
            token_offset = gen_offset
            gen_offset = token_offset + 1
            if megabyte is not None:
                megabyte.append(token, current_region, token_offset)
                continue
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = raw(
                    torch.tensor([[token]], device=device),
                    input_pos=torch.tensor([position], device=device),
                    input_pos_maxp1=position + 1,
                    region_ids=torch.full(
                        (1, 1), current_region, device=device, dtype=torch.long
                    ),
                    offset_ids=torch.full(
                        (1, 1), token_offset, device=device, dtype=torch.long
                    ),
                )
    finally:
        if megabyte is not None:
            megabyte.close()
        else:
            raw.clear_kv_cache()
    if trace is not None and mask_state is not None:
        trace["slice_layout"] = slice_layout
        trace["mask_calls"] = mask_state.mask_calls
        trace["mask_strict_calls"] = mask_state.strict_mask_calls
        trace["mask_permissive_calls"] = mask_state.permissive_mask_calls
        trace["mask_argmax_rejected"] = mask_argmax_rejected
        trace["mask_argmax_rejection_rate"] = (
            mask_argmax_rejected / mask_probability_mass_count
            if mask_probability_mass_count
            else None
        )
        trace["mask_allowed_probability_mass_mean"] = (
            mask_probability_mass_sum / mask_probability_mass_count
            if mask_probability_mass_count
            else None
        )
        trace["mask_probability_mass_count"] = mask_probability_mass_count
        trace["first_mask_intervention"] = first_mask_intervention
    return bytes(generated), start_codes


@torch.inference_mode()
def megabyte_generate_batch_frames(
    raw: nn.Module,
    sample: FreeRunSample,
    device: torch.device,
    batch_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    cont_frames: int,
    budget_multiplier: float = 2.0,
    slice_layout: str = HM.SLICE_LAYOUT_MACROBLOCK,
) -> list[GeneratedCandidate] | None:
    """Batched free-run generation, each candidate stopping independently once
    it closes ``cont_frames`` target VCL-NAL frames.

    Only valid when the checkpoint's region/offset conditioning is disabled
    (``config.use_region_id`` and ``config.use_offset_id`` both False).
    Region/offset ids are otherwise content-dependent -- they reset at NAL
    boundaries discovered only as bytes are generated -- which would require
    per-row metadata tracking incompatible with sharing one batched loop.
    When disabled, ``GPT._megabyte_global_embed`` never touches them at all
    (see litgpt/model.py), so passing constant placeholders is exact, not an
    approximation. Guarded explicitly below rather than silently producing
    wrong conditioning on a future checkpoint that does use them.

    Frame detection mirrors free_run_rollout's start-code/first_mb_in_slice
    logic (same primitives, same VCL_NAL_TYPES/first_mb==0 frame-boundary
    definition), evaluated independently per row against that row's own
    generated-byte buffer -- do not let this drift from free_run_rollout's
    definition of a frame boundary.
    """
    if raw.config.use_region_id or raw.config.use_offset_id:
        raise ValueError(
            "megabyte_generate_batch_frames only supports checkpoints with "
            "region/offset conditioning disabled (--no-region-id --no-offset-id)"
        )
    patch_size = int(raw.config.byte_patch_size)
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region_ids = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset_ids = sample.prompt_offset_ids.to(device).unsqueeze(0)

    # FreeRunSample.prompt_ids is [BOS, prefix bytes...]. Match
    # MegabyteInference(..., supervision_start=0) -- what free_run_rollout
    # actually uses for this exact sample shape -- rather than
    # megabyte_prompt_patches' whole-prompt left-pad (correct for FIM's
    # reconstruction-sample prompts, which have no such BOS prefix). Only
    # the 1-byte BOS is a left-padded "seed" patch; the rest packs as
    # complete patches with no left-padding, and any remainder becomes the
    # starting in-progress patch buffer generation continues filling. These
    # two conventions produce the SAME patch count (a ceiling-division
    # identity) but DIFFERENT byte-to-patch alignment whenever
    # len(prefix_bytes) % patch_size != 0 -- i.e. almost always -- which is
    # what silently broke this function before this fix (confirmed via
    # --verify: byte-identical divergence exactly at patch boundaries).
    seed_end = 1
    patched_ids, patched_regions, patched_offsets = megabyte_prompt_patches(
        prompt[:, :seed_end], region_ids[:, :seed_end], offset_ids[:, :seed_end], patch_size
    )
    known_ids = prompt[:, seed_end:]
    known_regions = region_ids[:, seed_end:]
    known_offsets = offset_ids[:, seed_end:]
    complete_known = known_ids.size(1) // patch_size
    complete_bytes = complete_known * patch_size
    if complete_known:
        patched_ids = torch.cat(
            (patched_ids, known_ids[:, :complete_bytes].view(1, complete_known, patch_size)),
            dim=1,
        )
        patched_regions = torch.cat(
            (patched_regions, known_regions[:, :complete_bytes].view(1, complete_known, patch_size)),
            dim=1,
        )
        patched_offsets = torch.cat(
            (patched_offsets, known_offsets[:, :complete_bytes].view(1, complete_known, patch_size)),
            dim=1,
        )
    prompt_patches = patched_ids.size(1)
    if prompt_patches > int(raw.max_seq_length):
        return None
    remainder_ids = known_ids[0, complete_bytes:]  # in-progress partial patch, if any

    max_new = min(
        int(budget_multiplier * sample.gt_cont_bytes) + 512,
        megabyte_max_new_bytes(raw, prompt.size(1), supervision_start=0),
    )
    if max_new <= 0:
        return None
    target_boundaries = _target_closed_vcl_nals(cont_frames, slice_layout)

    def _autocast():
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        )

    b = batch_size
    patched_ids_b = patched_ids.expand(b, -1, -1).contiguous()
    patched_regions_b = patched_regions.expand(b, -1, -1).contiguous()
    patched_offsets_b = patched_offsets.expand(b, -1, -1).contiguous()

    needed_patches = min(
        int(raw.max_seq_length),
        prompt_patches + -(-max_new // patch_size),
    )
    cache_dtype = torch.bfloat16 if device.type == "cuda" else next(raw.parameters()).dtype
    try:
        raw.set_kv_cache(
            batch_size=b, max_seq_length=needed_patches, device=device, dtype=cache_dtype
        )
        with _autocast():
            global_output = raw.megabyte_global_forward(
                patched_ids_b,
                input_pos=torch.arange(prompt_patches, device=device, dtype=torch.long),
                input_pos_maxp1=prompt_patches,
                region_ids=patched_regions_b,
                offset_ids=patched_offsets_b,
            )
        global_output = global_output[:, -1]  # (B, n_embd)

        # Region/offset are inert (guarded above), so patch commits during
        # generation use a constant placeholder rather than tracking any
        # real per-row schedule.
        placeholder_meta = torch.zeros((b, 1, patch_size), dtype=torch.long, device=device)

        generated: list[list[int]] = [[] for _ in range(b)]
        active = torch.ones(b, dtype=torch.bool, device=device)
        stopped = torch.zeros(b, dtype=torch.bool, device=device)
        frames_seen = [0] * b
        nal_start_idx: list[int | None] = [None] * b
        # Seed with the remainder bytes left over from the prompt split
        # above (identical across the batch -- same prompt for every row),
        # not an empty patch: matches MegabyteInference's
        # self._current_tokens initialization exactly.
        r = remainder_ids.numel()
        current_tokens = (
            remainder_ids.to(device).view(1, r).expand(b, r).contiguous()
            if r
            else torch.zeros((b, 0), dtype=torch.long, device=device)
        )
        position = prompt_patches
        for step in range(max_new):
            if current_tokens.size(1) == patch_size:
                with _autocast():
                    out = raw.megabyte_global_forward(
                        current_tokens.view(b, 1, patch_size),
                        input_pos=torch.tensor([position], device=device, dtype=torch.long),
                        input_pos_maxp1=position + 1,
                        region_ids=placeholder_meta,
                        offset_ids=placeholder_meta,
                    )
                global_output = out[:, -1]
                position += 1
                current_tokens = torch.zeros((b, 0), dtype=torch.long, device=device)

            with _autocast():
                logits = raw.megabyte_local_next_logits(
                    global_output, current_tokens
                )[:, :BYTE_VOCAB_SIZE]
            tokens = sample_tokens(logits, temperature, top_k, top_p)  # (B,)

            active_indices = active.nonzero(as_tuple=False).flatten()
            for row in active_indices.tolist():
                token = int(tokens[row])
                buf = generated[row]
                buf.append(token)
                if len(buf) >= 3 and tuple(buf[-3:]) == START_CODE:
                    if nal_start_idx[row] is not None:
                        nal_bytes = bytes(buf[nal_start_idx[row] : len(buf) - 3])
                        nal_type = (nal_bytes[0] & 0x1F) if nal_bytes else -1
                        if nal_type in VCL_NAL_TYPES:
                            first_mb = HS.slice_first_mb(nal_bytes)
                            if first_mb is None:
                                # Can't determine the frame boundary -- treat
                                # as a desync and stop, like free_run_rollout.
                                del buf[-3:]
                                active[row] = False
                                continue
                            if first_mb == 0:
                                frames_seen[row] += 1
                                if frames_seen[row] == target_boundaries:
                                    del buf[-3:]
                                    stopped[row] = True
                                    active[row] = False
                                    continue
                    nal_start_idx[row] = len(buf)

            if not bool(active.any()) or step == max_new - 1:
                break

            fed_tokens = torch.where(active, tokens, torch.zeros_like(tokens))
            current_tokens = torch.cat([current_tokens, fed_tokens.unsqueeze(1)], dim=1)
    finally:
        raw.clear_kv_cache()

    return [
        GeneratedCandidate(data=bytes(row), stopped=bool(stopped[i]))
        for i, row in enumerate(generated)
    ]


@torch.inference_mode()
def _generate(
    raw: nn.Module, sample: FreeRunSample, device, config: FreeRunEvalConfig
) -> bytes:
    """Free-run rollout from the clean prefix; thin wrapper over free_run_rollout."""
    prompt = sample.prompt_ids.to(device).unsqueeze(0)
    region = sample.prompt_region_ids.to(device).unsqueeze(0)
    offset = sample.prompt_offset_ids.to(device).unsqueeze(0)
    max_gen = min(
        int(sample.gt_cont_bytes * config.max_gen_multiple) + 512,
        megabyte_max_new_bytes(
            raw,
            sample.prompt_ids.numel(),
            supervision_start=(
                0 if int(raw.config.byte_patch_size) > 1 else None
            ),
        ),
    )
    generated, _start_codes = free_run_rollout(
        raw,
        prompt,
        region,
        offset,
        device,
        sample.cont_frames,
        max_gen,
        config.temperature,
        slice_layout=config.slice_layout,
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
    # ABSOLUTE offset (into prefix+generated) of the first desync, or None when clean.
    # ``survival`` is this minus the prefix length; the absolute form is what a
    # post-hoc analyzer needs to map a desync onto the NAL that contains it.
    first_desync: int | None = None


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


def _ends_with_dangling_start_code(data: bytes) -> int:
    """Length (3 or 4) of a trailing start code at the very end of ``data`` with no
    header byte after it, else 0. This is exactly what free_run_rollout produces
    when ``max_gen`` cuts generation off right after completing a start code, before
    sampling any byte of the next NAL (see the ``step == max_gen - 1`` break in
    free_run_rollout). iter_nals silently drops such a dangling start code (there's
    no NAL to parse), so callers must check for it separately or the truncation goes
    unlogged and looks identical to a clean, error-free stop.
    """
    if len(data) >= 4 and tuple(data[-4:]) == (0, 0, 0, 1):
        return 4
    if len(data) >= 3 and tuple(data[-3:]) == (0, 0, 1):
        return 3
    return 0


def _survival_and_validity(
    prefix_bytes: bytes, generated: bytes, cont_frames: int
) -> SurvivalResult:
    """Parse prefix+generated; return a SurvivalResult.

    survival = generated bytes before the first CAVLC desync at or after the prefix
    boundary. A desync counts even when it lands in a NAL that *began* in the prefix
    (the absorbed-garbage case, no leading start code). If nothing desyncs but the
    model never emitted a generated start code, survival is 0 (it never began a valid
    frame); only a started-and-clean budget gives survival = len(generated).

    If generation was cut off by the max_gen budget right after a start code (no
    header byte sampled yet), that dangling start code is not a real NAL and is
    excluded from survival; desync_reason is set to "truncated_at_budget" so this
    case is distinguishable in logs/metrics from a genuine clean stop, rather than
    silently reporting full survival for bytes that never became valid content.

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

    dangling = _ends_with_dangling_start_code(prefix_bytes + generated)
    if first_desync is not None:
        survival = first_desync - n_prefix
    elif dangling:
        # Budget ran out right after a start code; the dangling bytes aren't a
        # NAL (iter_nals dropped them) so they don't count as survived content.
        survival = len(generated) - dangling
    elif start_codes == 0:
        survival = 0  # model never emitted a generated frame boundary
    else:
        survival = len(generated)

    region, category, reason = _desync_info(desync_nal)
    if first_desync is None and dangling:
        reason = "truncated_at_budget"
    exposure = _exposure_by_category(parsed.all_spans(), n_prefix, n_prefix + survival)
    return SurvivalResult(
        survival,
        min(valid_cont, cont_frames),
        start_codes,
        region,
        category,
        reason,
        exposure,
        first_desync,
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
