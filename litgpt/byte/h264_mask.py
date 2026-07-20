"""Fast constrained-decoding masks for free-run H.264 byte generation.

Layer 1 enforces Annex-B emulation prevention in emitted EBSP byte space. Layer 2
uses the resumable CAVLC automaton in :mod:`h264_automaton` and its byte masks.
The recursive-descent parser is used only when a NAL closes (to retain
SPS/PPS) and once when enough of a VCL slice header has arrived to initialize the
automaton; it is never run once per candidate byte.

V1 is deliberately residual-first: header fields are decoded to maintain state but
receive a permissive mask. Strict trie-derived masks apply to CAVLC residual fields
and chain across all syntax-element boundaries contained in the next byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from litgpt.byte import h264_automaton as HA
from litgpt.byte import h264_syntax as HS

START_CODE = (0, 0, 1)
_DEBUG_DEFAULT = False
_DEBUG_STAGES_DEFAULT = False
_RESIDUAL_TAGS = {
    "coeff_token",
    "signs",
    "level_prefix",
    "level_suffix",
    "total_zeros",
    "run_before",
}


def configure_debug(enabled: bool = True, *, stages: bool = False) -> None:
    """Enable diagnostics for subsequently created ``MaskState`` objects."""
    global _DEBUG_DEFAULT, _DEBUG_STAGES_DEFAULT
    _DEBUG_DEFAULT = enabled
    _DEBUG_STAGES_DEFAULT = stages


@dataclass
class MaskState:
    """Streaming state for Annex-B framing and one open CAVLC NAL."""

    sps_map: dict[int, "HS.SPS"] = field(default_factory=dict)
    pps_map: dict[int, "HS.PPS"] = field(default_factory=dict)
    cur_nal_bytes: bytearray = field(default_factory=bytearray)
    cur_is_vcl: bool = False
    automaton: "HA.MbAutomaton | None" = None
    automaton_unknown: bool = False
    slice_max_mbs: int = 1
    debug: bool = field(default_factory=lambda: _DEBUG_DEFAULT)
    debug_stages: bool = field(default_factory=lambda: _DEBUG_STAGES_DEFAULT)
    debug_every_masks: int = 250
    nal_index: int = field(default=0, init=False)
    mask_calls: int = field(default=0, init=False)
    strict_mask_calls: int = field(default=0, init=False)

    def last_two_raw(self) -> tuple[int, int] | None:
        if len(self.cur_nal_bytes) < 2:
            return None
        return self.cur_nal_bytes[-2], self.cur_nal_bytes[-1]

    @property
    def at_nal_boundary(self) -> bool:
        return self.automaton is not None and self.automaton.stage == "done"


def annexb_forbidden(state: MaskState, *, at_nal_boundary: bool) -> list[int]:
    """Return bytes forbidden after the two most recent emitted EBSP bytes."""
    if state.last_two_raw() != (0x00, 0x00):
        return []
    if at_nal_boundary:
        return [0x02]
    return [0x00, 0x01, 0x02]


def _has_pending_emulation_prevention(state: MaskState) -> bool:
    return (
        len(state.cur_nal_bytes) >= 3
        and tuple(state.cur_nal_bytes[-3:]) == (0x00, 0x00, 0x03)
    )


def _apply_annexb_mask(
    state: MaskState, mask: list[bool], *, at_nal_boundary: bool
) -> None:
    """Translate an RBSP-byte legality mask into emitted EBSP-byte legality.

    Mid-NAL, after raw ``00 00``, emitted ``03`` contributes no RBSP bits and
    announces that the following emitted byte must be in ``00..03``. At a known or
    conservative boundary, keep start-code bytes available instead.
    """
    if at_nal_boundary:
        for byte in annexb_forbidden(state, at_nal_boundary=True):
            mask[byte] = False
        return

    if _has_pending_emulation_prevention(state):
        # The following byte is the actual RBSP byte protected by the EPB.
        for byte in range(4, 256):
            mask[byte] = False
        return

    if state.last_two_raw() == (0x00, 0x00):
        # 00/01/02 cannot be emitted directly. Emitted 03 is legal iff at least
        # one protected RBSP continuation 00..03 is syntactically legal.
        epb_legal = any(mask[:4])
        mask[0] = mask[1] = mask[2] = False
        mask[3] = epb_legal


def get_valid_byte_mask(state: MaskState) -> list[bool]:
    """Return a 256-entry mask for the next emitted byte.

    Unknown/unsupported syntax remains permissive. Definite residual CAVLC
    incompatibilities and Annex-B violations are rejected.
    """
    auto = state.automaton
    state.mask_calls += 1
    strict = False
    if (
        state.cur_is_vcl
        and auto is not None
        and not state.automaton_unknown
        and auto.stage != "done"
        and auto.ae_tag in _RESIDUAL_TAGS
    ):
        mask = HA.compile_byte_mask(auto, residual_only=True)
        strict = True
        state.strict_mask_calls += 1
    else:
        mask = [True] * 256

    boundary = (
        True
        if not state.cur_is_vcl or state.automaton_unknown or auto is None
        else state.at_nal_boundary
    )
    _apply_annexb_mask(state, mask, at_nal_boundary=boundary)
    if state.debug and state.mask_calls % state.debug_every_masks == 0:
        _debug(
            state,
            "mask-summary",
            allowed=sum(mask),
            stage=auto.stage if auto is not None else "uninitialized",
            syntax=auto.ae_tag if auto is not None else "unknown",
            strict=strict,
        )
    return mask


def advance(state: MaskState, byte: int) -> None:
    """Commit one emitted EBSP byte and incrementally advance parser state."""
    if not 0 <= byte <= 255:
        raise ValueError("byte must be in [0, 255]")

    was_empty = not state.cur_nal_bytes
    state.cur_nal_bytes.append(byte)
    if was_empty:
        state.cur_is_vcl = (byte & 0x1F) in HS.VCL_NAL_TYPES

    tail = state.cur_nal_bytes
    if len(tail) >= 3 and tuple(tail[-3:]) == START_CODE:
        sc_len = 4 if len(tail) >= 4 and tuple(tail[-4:]) == (0, 0, 0, 1) else 3
        _close_nal(state, bytes(tail[:-sc_len]))
        state.cur_nal_bytes = bytearray()
        state.cur_is_vcl = False
        state.automaton = None
        state.automaton_unknown = False
        state.nal_index += 1
        return

    if state.cur_is_vcl and not state.automaton_unknown:
        _sync_automaton(state)


def _sync_automaton(state: MaskState) -> None:
    """Initialize when the slice header is complete, then feed newly arrived bits."""
    if len(state.cur_nal_bytes) < 2:
        return
    payload_end = len(state.cur_nal_bytes)
    if _has_pending_emulation_prevention(state):
        # Whether a terminal 00 00 03 is an EPB depends on the following byte. Do
        # not feed the 03 as RBSP data while that byte is still unknown.
        payload_end -= 1
    rbsp, byte_map, _ = HS.unescape_rbsp(bytes(state.cur_nal_bytes), 1, payload_end)
    if not byte_map:
        return

    if state.automaton is None:
        try:
            reader = HS.BitReader(rbsp)
            record = HS._Recorder(byte_map, len(rbsp))
            save = reader.pos
            reader.read_ue()
            reader.read_ue()
            pps = state.pps_map.get(reader.read_ue())
            reader.pos = save
            if pps is None:
                return
            sps = state.sps_map.get(pps.sps_id)
            if sps is None:
                return
            nal = _nal_info(state.cur_nal_bytes)
            header = HS.parse_slice_header(reader, record, nal, sps, pps)
            if header.slice_type not in (HS.SLICE_TYPE_P, HS.SLICE_TYPE_I):
                state.automaton_unknown = True
                _debug(state, "fallback-permissive", reason=f"slice_type={header.slice_type}")
                return
            state.automaton = HA.MbAutomaton(
                pic_width_in_mbs=sps.pic_width_in_mbs,
                pic_height_in_mbs=sps.pic_height_in_mbs,
                slice_type=header.slice_type,
                num_ref_idx_l0_active=header.num_ref_idx_l0_active,
                first_mb_in_slice=header.first_mb_in_slice,
                slice_data_start_bit=reader.pos,
                max_mbs=state.slice_max_mbs,
            )
        except HS.BitReaderError:
            return
        except (ValueError, KeyError, IndexError, HS._DesyncError, HS._Unsupported) as exc:
            state.automaton_unknown = True
            _debug(state, "fallback-permissive", reason=f"{type(exc).__name__}: {exc}")
            return

    auto = state.automaton
    assert auto is not None
    nbits = len(rbsp) * 8
    while auto.pos < nbits and auto.stage != "done":
        pos = auto.pos
        bit = (rbsp[pos >> 3] >> (7 - (pos & 7))) & 1
        before = _syntax_name(auto)
        status = auto.consume_bit(bit)
        if status == HA.INVALID:
            # Conservative fallback: unsupported or already-corrupt prefixes must not
            # turn into an all-false mask and crash generation.
            state.automaton_unknown = True
            _debug(
                state,
                "fallback-permissive",
                reason="automaton-invalid",
                rbsp_bit=pos,
                syntax=auto.ae_tag,
            )
            return
        after = _syntax_name(auto)
        if state.debug_stages and after != before:
            _debug(
                state,
                "syntax-transition",
                rbsp_bit=auto.pos,
                mb=auto.mbs_done,
                block=auto.res_blk,
                previous=before,
                current=after,
            )
        if state.debug_stages and status == HA.COMPLETE_MB:
            _debug(
                state,
                "macroblock-complete",
                mbs=auto.mbs_done,
                rbsp_bit=auto.pos,
                next=after,
            )


def _debug(state: MaskState, event: str, **fields: object) -> None:
    if not state.debug:
        return
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[h264-mask] nal={state.nal_index} event={event}{suffix}", flush=True)


def _syntax_name(auto: "HA.MbAutomaton") -> str:
    if auto.stage == "trailing":
        return "rbsp_trailing_bits"
    if auto.stage == "done":
        return "done"
    return {
        "mb_skip_run": "mb_skip_run",
        "mb_type": "mb_type",
        "sub_mb_type": "prediction.sub_mb_type",
        "ref_idx": "prediction.ref_idx_l0",
        "mvd": "prediction.mvd_l0",
        "prev_intra": "prediction.prev_intra",
        "rem_intra": "prediction.rem_intra",
        "intra_chroma": "prediction.intra_chroma",
        "cbp": "coded_block_pattern",
        "mb_qp_delta": "mb_qp_delta",
        "coeff_token": "residual.coeff_token",
        "signs": "residual.trailing_one_signs",
        "level_prefix": "residual.level_prefix",
        "level_suffix": "residual.level_suffix",
        "total_zeros": "residual.total_zeros",
        "run_before": "residual.run_before",
        "pcm": "prediction.pcm",
    }.get(auto.ae_tag, str(auto.ae_tag))


def _nal_info(payload: bytes | bytearray) -> "HS.NALInfo":
    header = payload[0]
    return HS.NALInfo(
        index=0,
        nal_type=header & 0x1F,
        ref_idc=(header >> 5) & 0x3,
        start_code_start=0,
        start_code_len=3,
        payload_start=0,
        payload_end=len(payload),
    )


def _close_nal(state: MaskState, nal_payload: bytes) -> None:
    """Parse a closed NAL once so newly defined SPS/PPS remain available."""
    if not nal_payload:
        return
    buf = bytes(START_CODE) + nal_payload
    nals = HS.iter_nals(buf)
    if not nals:
        return
    try:
        HS.parse_nal(buf, nals[0], state.sps_map, state.pps_map, parse_slice_data=False)
    except (ValueError, KeyError, IndexError, HS.BitReaderError, HS._DesyncError, HS._Unsupported):
        # Generated non-parameter NALs can be incomplete or unsupported. Parameter
        # maps already populated by earlier valid NALs remain usable.
        return
