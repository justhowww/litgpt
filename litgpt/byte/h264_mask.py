"""Fast constrained-decoding masks for free-run H.264 byte generation.

Layer 1 enforces Annex-B emulation prevention in emitted EBSP byte space. Layer 2
uses the resumable CAVLC automaton in :mod:`h264_automaton` and its byte masks.
The recursive-descent parser is used only when a NAL closes to retain SPS/PPS.
The resumable VCL automaton covers the slice header and slice data without
re-parsing the prefix once per candidate byte.

The default mask covers the complete supported slice-data path: macroblock header,
prediction, CBP, QP delta, CAVLC residual fields, and rbsp_trailing_bits. A legacy
``residual_only`` state option is retained solely for paired ablations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from litgpt.byte import h264_automaton as HA
from litgpt.byte import h264_syntax as HS

START_CODE = (0, 0, 1)
SLICE_LAYOUT_MACROBLOCK = "macroblock"
SLICE_LAYOUT_FRAME = "frame"
SLICE_LAYOUTS = (SLICE_LAYOUT_MACROBLOCK, SLICE_LAYOUT_FRAME)
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


def slice_max_mbs_for_layout(layout: str) -> int | None:
    """Translate the public layout name to the automaton's slice extent.

    ``frame`` means one complete progressive picture per slice, resolved from SPS.
    The user-facing name stays simple while the codec-facing implementation uses
    picture dimensions (a frame and picture coincide for the supported profile).
    """
    if layout == SLICE_LAYOUT_MACROBLOCK:
        return 1
    if layout == SLICE_LAYOUT_FRAME:
        return None
    raise ValueError(f"unknown slice layout {layout!r}; expected one of {SLICE_LAYOUTS}")


@dataclass
class PictureState:
    """Cross-NAL picture state for fixed-MB and one-picture slice layouts."""

    active: bool = False
    picture_complete: bool = False
    picture_mbs: int | None = None
    next_first_mb: int | None = None
    frame_num: int | None = None
    pps_id: int | None = None
    slice_type: int | None = None
    nal_type: int | None = None
    nal_ref_idc: int | None = None
    identity: dict[str, int] = field(default_factory=dict)
    reference_pictures: int = 0
    last_nonidr_nal_type: int | None = None
    last_nonidr_ref_idc: int | None = None
    last_nonidr_slice_type: int | None = None

    def allowed_nal_headers(self) -> list[int]:
        """Return profile-consistent VCL NAL header bytes for the next slice."""
        if self.active and not self.picture_complete:
            return [(self.nal_ref_idc << 5) | self.nal_type]
        if self.last_nonidr_nal_type is not None:
            return [(self.last_nonidr_ref_idc << 5) | self.last_nonidr_nal_type]
        if self.active and self.nal_type == HS.NAL_SLICE_IDR:
            # The first predictive picture after an IDR is non-IDR.  The prefix
            # normally teaches ref_idc=2 before free-run begins; permit all legal
            # reference strengths only for this bootstrap transition.
            return [(ref_idc << 5) | HS.NAL_SLICE_NONIDR for ref_idc in (1, 2, 3)]
        # A phase-profile stream begins with a reference IDR picture.
        return [(3 << 5) | HS.NAL_SLICE_IDR]

    def constraints(
        self, nal_type: int, nal_ref_idc: int, sps_map: dict, pps_map: dict
    ) -> dict:
        same_picture = self.active and not self.picture_complete
        expected_frame = self.frame_num
        if self.active and self.picture_complete:
            # Resolve MaxFrameNum through the active PPS/SPS if still available.
            # Unknown maps are handled by the header automaton itself.
            expected_frame = None
            pps = pps_map.get(self.pps_id)
            sps = None if pps is None else sps_map.get(pps.sps_id)
            if sps is not None:
                expected_frame = (self.frame_num + 1) % (1 << sps.log2_max_frame_num)
            if nal_type == HS.NAL_SLICE_IDR:
                expected_frame = 0
        result = {
            "nal_type": self.nal_type if same_picture else nal_type,
            "nal_ref_idc": self.nal_ref_idc if same_picture else nal_ref_idc,
            "first_mb_in_slice": self.next_first_mb if same_picture else 0,
            "frame_num": expected_frame,
            "pic_parameter_set_id": self.pps_id,
            "slice_type": (
                self.slice_type
                if same_picture
                else (
                    HS.SLICE_TYPE_I
                    if nal_type == HS.NAL_SLICE_IDR
                    else self.last_nonidr_slice_type
                )
            ),
            "available_reference_pictures": self.reference_pictures,
        }
        if same_picture:
            result.update(self.identity)
        return result

    def observe(
        self, header: dict, *, nal_type: int, nal_ref_idc: int, sps, max_mbs: int
    ) -> None:
        """Commit one completely parsed slice header to picture state."""
        first_mb = header["first_mb_in_slice"]
        picture_mbs = sps.pic_width_in_mbs * sps.pic_height_in_mbs
        new_picture = not self.active or self.picture_complete
        if new_picture:
            self.active = True
            self.picture_complete = False
            self.picture_mbs = picture_mbs
            self.frame_num = header["frame_num"]
            self.pps_id = header["pic_parameter_set_id"]
            self.slice_type = header["slice_type"]
            self.nal_type = nal_type
            self.nal_ref_idc = nal_ref_idc
            self.identity = {
                key: header[key]
                for key in (
                    "idr_pic_id",
                    "pic_order_cnt_lsb",
                    "delta_pic_order_cnt_bottom",
                    "delta_pic_order_cnt[0]",
                    "delta_pic_order_cnt[1]",
                    "no_output_of_prior_pics_flag",
                    "long_term_reference_flag",
                )
                if key in header
            }
        self.next_first_mb = first_mb + max_mbs
        if nal_type == HS.NAL_SLICE_NONIDR:
            self.last_nonidr_nal_type = nal_type
            self.last_nonidr_ref_idc = nal_ref_idc
            self.last_nonidr_slice_type = header["slice_type"]
        if self.next_first_mb >= picture_mbs:
            self.picture_complete = True
            if nal_ref_idc != 0:
                self.reference_pictures = min(16, self.reference_pictures + 1)


@dataclass
class MaskState:
    """Streaming state for Annex-B framing and one open CAVLC NAL."""

    sps_map: dict[int, "HS.SPS"] = field(default_factory=dict)
    pps_map: dict[int, "HS.PPS"] = field(default_factory=dict)
    cur_nal_bytes: bytearray = field(default_factory=bytearray)
    cur_is_vcl: bool = False
    automaton: "HA.MbAutomaton | HA.VclAutomaton | None" = None
    automaton_unknown: bool = False
    picture: PictureState = field(default_factory=PictureState)
    expect_nal_header: bool = False
    generation_started: bool = False
    # Positive integer: fixed macroblocks per slice (the AVC-LM corpus uses 1).
    # None: one complete progressive picture per slice; VclAutomaton resolves the
    # concrete extent from the active SPS and requires first_mb_in_slice == 0.
    slice_max_mbs: int | None = 1
    residual_only: bool = False
    fail_closed: bool = True
    debug: bool = field(default_factory=lambda: _DEBUG_DEFAULT)
    debug_stages: bool = field(default_factory=lambda: _DEBUG_STAGES_DEFAULT)
    debug_every_masks: int = 250
    nal_index: int = field(default=0, init=False)
    mask_calls: int = field(default=0, init=False)
    strict_mask_calls: int = field(default=0, init=False)
    permissive_mask_calls: int = field(default=0, init=False)
    failure_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.slice_max_mbs is not None and self.slice_max_mbs <= 0:
            raise ValueError("slice_max_mbs must be positive or None")

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
    return len(state.cur_nal_bytes) >= 3 and tuple(state.cur_nal_bytes[-3:]) == (
        0x00,
        0x00,
        0x03,
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

    In full mode, the NAL header, slice header, and supported slice data are all
    constrained before commitment. Invalid/unsupported state fails closed instead
    of silently reverting to unconstrained generation.
    """
    auto = state.automaton
    state.mask_calls += 1
    strict = False
    if (
        state.expect_nal_header
        and not state.residual_only
        and (state.picture.active or state.generation_started)
    ):
        mask = [False] * 256
        for header in state.picture.allowed_nal_headers():
            mask[header] = True
        state.strict_mask_calls += 1
        if state.debug and state.mask_calls % state.debug_every_masks == 0:
            _debug(
                state,
                "mask-summary",
                allowed=sum(mask),
                stage="nal_header",
                syntax="nal_header",
                strict=True,
            )
        return mask
    active = (
        state.cur_is_vcl
        and auto is not None
        and not state.automaton_unknown
        and auto.stage != "done"
    )
    should_compile = active and (
        not state.residual_only or auto.ae_tag in _RESIDUAL_TAGS
    )
    if should_compile:
        mask = HA.compile_byte_mask(auto, residual_only=state.residual_only)
        strict = True
        state.strict_mask_calls += 1
        if not any(mask) and state.failure_reason is None:
            state.failure_reason = (
                f"no_valid_byte:{auto.stage}:{auto.ae_tag}@{auto.pos}"
            )
            _debug(
                state,
                "mask-state-invalid",
                reason=state.failure_reason,
            )
    elif state.cur_is_vcl and state.automaton_unknown and state.fail_closed:
        mask = [False] * 256
    elif (
        state.cur_is_vcl
        and auto is not None
        and auto.stage == "done"
        and not state.residual_only
    ):
        mask = _nal_boundary_mask(state)
        strict = True
        state.strict_mask_calls += 1
    else:
        mask = [True] * 256
        state.permissive_mask_calls += 1

    boundary = (
        True
        if not state.cur_is_vcl or state.automaton_unknown or auto is None
        else state.at_nal_boundary
    )
    full_boundary = (
        state.cur_is_vcl
        and auto is not None
        and auto.stage == "done"
        and not state.residual_only
    )
    if not full_boundary:
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


def can_append_bytes(
    state: MaskState,
    data: bytes | bytearray,
    *,
    require_complete: bool = False,
) -> bool:
    """Return whether ``data`` can legally continue ``state`` without mutating it.

    FIM generation needs this to decide whether EOS is legal: the generated middle
    may stop only when the fixed orphan/suffix can be consumed from the resulting
    parser state.  Copying the state is intentional--probing EOS must not commit the
    suffix to the live generation state.

    ``require_complete`` additionally rejects a suffix that ends in an unfinished
    VCL NAL or a dangling start code.  This is the mode used by the FIM evaluator at
    the end of the repaired frame.
    """
    probe = deepcopy(state)
    for byte in data:
        allowed = get_valid_byte_mask(probe)
        if not allowed[byte]:
            return False
        advance(probe, byte)
        if probe.automaton_unknown:
            return False

    if not require_complete:
        return True
    if probe.automaton_unknown or probe.expect_nal_header:
        return False
    if probe.cur_is_vcl:
        return probe.automaton is not None and probe.automaton.stage == "done"
    return True


def _nal_boundary_mask(state: MaskState) -> list[bool]:
    """Allow only Annex-B zero bytes and the terminating ``00 00 01`` prefix.

    ``rbsp_trailing_bits`` ends on an RBSP byte boundary. Non-zero payload bytes
    after that point are not part of the completed NAL and must not be accepted as
    if the automaton still described them.
    """
    mask = [False] * 256
    mask[0x00] = True
    if state.last_two_raw() == (0x00, 0x00):
        mask[0x01] = True
    return mask


def advance(state: MaskState, byte: int) -> None:
    """Commit one emitted EBSP byte and incrementally advance parser state."""
    if not 0 <= byte <= 255:
        raise ValueError("byte must be in [0, 255]")

    was_empty = not state.cur_nal_bytes
    state.cur_nal_bytes.append(byte)
    if was_empty:
        state.cur_is_vcl = (byte & 0x1F) in HS.VCL_NAL_TYPES
        state.expect_nal_header = False
        if state.cur_is_vcl and not state.residual_only:
            _init_vcl_automaton(state)

    tail = state.cur_nal_bytes
    if len(tail) >= 3 and tuple(tail[-3:]) == START_CODE:
        sc_len = 4 if len(tail) >= 4 and tuple(tail[-4:]) == (0, 0, 0, 1) else 3
        _close_nal(state, bytes(tail[:-sc_len]))
        state.cur_nal_bytes = bytearray()
        state.cur_is_vcl = False
        state.automaton = None
        state.automaton_unknown = False
        state.expect_nal_header = True
        state.failure_reason = None
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

    if state.automaton is None and not state.residual_only:
        _init_vcl_automaton(state)
        if state.automaton_unknown:
            return

    if state.automaton is None:
        try:
            reader = HS.BitReader(rbsp)
            record = HS._Recorder(byte_map, len(rbsp))
            save = reader.pos
            first_mb = reader.read_ue()
            slice_type_raw = reader.read_ue()
            pps_id = reader.read_ue()
            reader.pos = save
            if slice_type_raw > 9:
                state.automaton_unknown = True
                state.failure_reason = f"slice_type_out_of_range:{slice_type_raw}"
                return
            pps = state.pps_map.get(pps_id)
            if pps is None:
                state.automaton_unknown = True
                state.failure_reason = f"unknown_pps:{pps_id}"
                return
            sps = state.sps_map.get(pps.sps_id)
            if sps is None:
                state.automaton_unknown = True
                state.failure_reason = f"unknown_sps:{pps.sps_id}"
                return
            picture_mbs = sps.pic_width_in_mbs * sps.pic_height_in_mbs
            if first_mb >= picture_mbs:
                state.automaton_unknown = True
                state.failure_reason = f"first_mb_out_of_range:{first_mb}/{picture_mbs}"
                return
            max_mbs = state.slice_max_mbs
            if max_mbs is None:
                if first_mb != 0:
                    state.automaton_unknown = True
                    state.failure_reason = (
                        f"frame_slice_first_mb_not_zero:{first_mb}"
                    )
                    return
                max_mbs = picture_mbs
            nal = _nal_info(state.cur_nal_bytes)
            header = HS.parse_slice_header(reader, record, nal, sps, pps)
            if header.slice_type not in (HS.SLICE_TYPE_P, HS.SLICE_TYPE_I):
                state.automaton_unknown = True
                state.failure_reason = f"unsupported_slice_type:{header.slice_type}"
                _debug(
                    state,
                    "fallback-permissive",
                    reason=f"slice_type={header.slice_type}",
                )
                return
            if (
                nal.nal_type == HS.NAL_SLICE_IDR
                and header.slice_type != HS.SLICE_TYPE_I
            ):
                state.automaton_unknown = True
                state.failure_reason = f"idr_non_i_slice:{header.slice_type}"
                return
            if not 0 <= header.slice_qp <= 51:
                state.automaton_unknown = True
                state.failure_reason = f"slice_qp_out_of_range:{header.slice_qp}"
                return
            if (
                header.slice_type == HS.SLICE_TYPE_P
                and not 1 <= header.num_ref_idx_l0_active <= 16
            ):
                state.automaton_unknown = True
                state.failure_reason = (
                    f"active_refs_out_of_range:{header.num_ref_idx_l0_active}"
                )
                return
            state.automaton = HA.MbAutomaton(
                pic_width_in_mbs=sps.pic_width_in_mbs,
                pic_height_in_mbs=sps.pic_height_in_mbs,
                slice_type=header.slice_type,
                num_ref_idx_l0_active=header.num_ref_idx_l0_active,
                first_mb_in_slice=header.first_mb_in_slice,
                slice_data_start_bit=reader.pos,
                max_mbs=max_mbs,
            )
        except HS.BitReaderError:
            return
        except (
            ValueError,
            KeyError,
            IndexError,
            HS._DesyncError,
            HS._Unsupported,
        ) as exc:
            state.automaton_unknown = True
            state.failure_reason = f"slice_header:{type(exc).__name__}:{exc}"
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
            # The next mask call fails closed by default. Keeping the state transition
            # here (instead of raising) gives rollout a deterministic stop reason.
            state.automaton_unknown = True
            state.failure_reason = f"automaton_invalid:{auto.ae_tag}@{pos}"
            invalid_reason = getattr(auto, "invalid_reason", None)
            if invalid_reason:
                state.failure_reason = invalid_reason
            _debug(
                state,
                "mask-state-invalid" if state.fail_closed else "fallback-permissive",
                reason=state.failure_reason,
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


def _syntax_name(auto) -> str:
    if auto.stage == "header":
        return f"slice_header.{auto.ae_tag}"
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


def _init_vcl_automaton(state: MaskState) -> None:
    if not state.cur_nal_bytes:
        return
    nal = _nal_info(state.cur_nal_bytes)
    try:
        state.automaton = HA.VclAutomaton(
            nal_type=nal.nal_type,
            nal_ref_idc=nal.ref_idc,
            sps_map=state.sps_map,
            pps_map=state.pps_map,
            constraints=state.picture.constraints(
                nal.nal_type, nal.ref_idc, state.sps_map, state.pps_map
            ),
            max_mbs=state.slice_max_mbs,
        )
    except ValueError as exc:
        state.automaton_unknown = True
        state.failure_reason = f"vcl_header:{exc}"
        _debug(
            state,
            "mask-state-invalid" if state.fail_closed else "fallback-permissive",
            reason=state.failure_reason,
        )


def _close_nal(state: MaskState, nal_payload: bytes) -> None:
    """Parse a closed NAL once so newly defined SPS/PPS remain available."""
    if not nal_payload:
        return
    buf = bytes(START_CODE) + nal_payload
    nals = HS.iter_nals(buf)
    if not nals:
        return
    auto = state.automaton
    if (
        isinstance(auto, HA.VclAutomaton)
        and auto.stage == "done"
        and auto.sps is not None
        and auto.max_mbs is not None
    ):
        state.picture.observe(
            auto.header,
            nal_type=nals[0].nal_type,
            nal_ref_idc=nals[0].ref_idc,
            sps=auto.sps,
            max_mbs=auto.max_mbs,
        )
    try:
        HS.parse_nal(buf, nals[0], state.sps_map, state.pps_map, parse_slice_data=False)
    except (
        ValueError,
        KeyError,
        IndexError,
        HS.BitReaderError,
        HS._DesyncError,
        HS._Unsupported,
    ):
        # Generated non-parameter NALs can be incomplete or unsupported. Parameter
        # maps already populated by earlier valid NALs remain usable.
        return
