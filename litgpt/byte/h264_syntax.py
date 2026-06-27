"""Constrained-Baseline H.264 syntax parser (CAVLC), stdlib-only.

Maps every byte of an Annex-B H.264 stream to the syntax element it belongs to:
start code, NAL header, SPS/PPS fields, slice-header fields, and the macroblock
layer (mb_type, prediction, coded_block_pattern, mb_qp_delta, CAVLC residual
coefficients), plus emulation-prevention bytes. This is the byte->syntax map the
diagnoser overlays on the model's per-byte predictions.

Scope is the pinned profile (see 04 - projects/.../h264_tutorial.md): H.264/AVC
**Constrained Baseline** -- CAVLC only (no CABAC), no B-slices, no MBAFF, 4x4
transform only, 4:2:0. Anything outside that is reported as ParseStatus.unsupported
rather than silently mis-parsed.

No third-party deps (no torch): this module is developed and tested without the
training environment so its invariants can be checked anywhere.

Correctness is self-checked by two invariants the diagnoser/tests assert on
ground-truth streams (a desync breaks both):
  A. exact consumption  -- the bit cursor lands on the RBSP trailing-bit boundary.
  B. MB count           -- decoded MBs == PicWidthInMbs * PicHeightInMbs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

try:  # package import (full env) vs flat import (stdlib-only tests)
    from . import h264_cavlc_tables as T
except ImportError:  # pragma: no cover
    import h264_cavlc_tables as T

# ---------------------------------------------------------------------------
# NAL types / slice types
# ---------------------------------------------------------------------------

NAL_SLICE_NONIDR = 1
NAL_SEI = 6
NAL_SPS = 7
NAL_PPS = 8
NAL_SLICE_IDR = 5
VCL_NAL_TYPES = frozenset({1, 5})
PARAMETER_SET_NAL_TYPES = frozenset({7, 8})

SLICE_TYPE_P = 0
SLICE_TYPE_B = 1
SLICE_TYPE_I = 2
SLICE_TYPE_SP = 3
SLICE_TYPE_SI = 4


class Category(str, Enum):
    START_CODE = "start_code"
    NAL_HEADER = "nal_header"
    EMULATION_PREVENTION = "emulation_prevention"
    SPS = "sps"
    PPS = "pps"
    SEI = "sei"
    SLICE_HEADER = "slice_header"
    MB_HEADER = "mb_header"  # mb_type / mb_skip_run / sub_mb_type
    MB_PRED = "mb_pred"  # intra modes, ref_idx, mvd
    CBP = "cbp"  # coded_block_pattern
    MB_QP_DELTA = "mb_qp_delta"
    RESIDUAL_LUMA = "residual_luma"
    RESIDUAL_CHROMA = "residual_chroma"
    RBSP_TRAILING = "rbsp_trailing"
    SLICE_DATA = "slice_data"  # opaque fallback (only if MB parse disabled)
    UNKNOWN = "unknown"


class ParseStatus(str, Enum):
    OK = "ok"
    DESYNC = "desync"
    UNSUPPORTED = "unsupported"


@dataclass
class SyntaxSpan:
    """One syntax element, located in both RBSP bit space and Annex-B byte space."""

    name: str
    category: Category
    bit_start: int  # inclusive, RBSP bit offset
    bit_end: int  # exclusive, RBSP bit offset
    byte_start: int  # inclusive, Annex-B on-disk offset
    byte_end: int  # exclusive, Annex-B on-disk offset
    value: object = None
    mb_addr: int | None = None


@dataclass
class NALInfo:
    index: int
    nal_type: int
    ref_idc: int
    start_code_start: int  # Annex-B offset of the 00 00 01 / 00 00 00 01
    start_code_len: int  # 3 or 4
    payload_start: int  # Annex-B offset of the NAL header byte
    payload_end: int  # exclusive Annex-B offset (next start code / EOF)


@dataclass
class NALParse:
    nal: NALInfo
    status: ParseStatus
    spans: list[SyntaxSpan] = field(default_factory=list)
    desync_bit: int | None = None
    desync_byte: int | None = None
    reason: str | None = None
    mb_count: int | None = None
    # Bits of RBSP consumed when parsing stopped (for invariant A).
    consumed_bits: int | None = None
    rbsp_bits_total: int | None = None


# ---------------------------------------------------------------------------
# Annex-B NAL iteration
# ---------------------------------------------------------------------------


def iter_nals(data: bytes) -> list[NALInfo]:
    """Split an Annex-B byte stream into NAL units (stdlib, mirrors data.py)."""
    starts: list[tuple[int, int]] = []  # (start_code_start, start_code_len)
    i, n = 0, len(data)
    while i + 2 < n:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            # check if it's a 4-byte start code (00 00 00 01)  or 3-byte (00 00 01)
            sc_len = 3
            sc_start = i
            if i >= 1 and data[i - 1] == 0:
                sc_len = 4
                sc_start = i - 1
            starts.append((sc_start, sc_len))
            i += 3
        else:  # not a start code, move forward
            i += 1
    nals: list[NALInfo] = []
    for k, (sc_start, sc_len) in enumerate(starts):
        payload_start = sc_start + sc_len
        payload_end = starts[k + 1][0] if k + 1 < len(starts) else n
        header = data[payload_start]
        nals.append(
            NALInfo(
                index=k,
                nal_type=header
                & 0x1F,  # forbidden_zero_bit | nal_ref_idc | nal_unit_type  with 1 bit    |   2 bits    |    5 bits
                ref_idc=(header >> 5)
                & 0x3,  # forbidden_zero_bit | nal_ref_idc | nal_unit_type  with 1 bit    |   2 bits    |    5 bits
                start_code_start=sc_start,
                start_code_len=sc_len,
                payload_start=payload_start,
                payload_end=payload_end,
            )
        )
    return nals


def unescape_rbsp(
    data: bytes, payload_start: int, payload_end: int
) -> tuple[bytes, list[int], list[int]]:
    """Strip emulation-prevention bytes from a NAL payload (after the header byte).

    Returns (rbsp, byte_map, epb_offsets) where rbsp[i]'s source Annex-B offset is
    byte_map[i], and epb_offsets holds the Annex-B offsets of the removed
    ``0x03`` emulation-prevention bytes.
    """
    rbsp = bytearray()
    byte_map: list[int] = []
    epb_offsets: list[int] = []
    zeros = 0
    j = payload_start
    while j < payload_end:
        b = data[j]
        if zeros >= 2 and b == 0x03 and j + 1 < payload_end and data[j + 1] <= 0x03:
            # Emulation-prevention three byte: drop it, reset the zero run.
            epb_offsets.append(j)
            zeros = 0
            j += 1
            continue
        rbsp.append(b)
        byte_map.append(j)
        zeros = zeros + 1 if b == 0 else 0
        j += 1
    return bytes(rbsp), byte_map, epb_offsets


# ---------------------------------------------------------------------------
# Bit reader + exp-Golomb
# ---------------------------------------------------------------------------


class BitReaderError(Exception):
    pass


class BitReader:
    def __init__(self, rbsp: bytes) -> None:
        self.rbsp = rbsp
        self.nbits = len(rbsp) * 8
        self.pos = 0

    def read_bit(self) -> int:
        if self.pos >= self.nbits:
            raise BitReaderError("read past end of RBSP")
        byte = self.rbsp[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def read_bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_ue(self) -> int:
        leading_zeros = 0
        while self.read_bit() == 0:
            leading_zeros += 1
            if leading_zeros > 31:
                raise BitReaderError("exp-Golomb prefix too long")
        if leading_zeros == 0:
            return 0
        return (1 << leading_zeros) - 1 + self.read_bits(leading_zeros)

    def read_se(self) -> int:
        k = self.read_ue()
        if k == 0:
            return 0
        m = (k + 1) >> 1
        return m if (k & 1) else -m

    def read_te(self, x_max: int) -> int:
        # Truncated exp-Golomb: when the range is 1, a single inverted bit.
        if x_max == 1:
            return 1 - self.read_bit()
        return self.read_ue()

    def byte_aligned(self) -> bool:
        return (self.pos & 7) == 0

    def compute_stop_bit(self) -> int:
        """Locate and cache rbsp_stop_one_bit (the last set bit in the RBSP)."""
        last_one = -1
        for p in range(self.nbits - 1, -1, -1):
            if (self.rbsp[p >> 3] >> (7 - (p & 7))) & 1:
                last_one = p
                break
        self.stop_bit = last_one
        return last_one

    def more_rbsp_data(self) -> bool:
        """True if a syntax element remains before the rbsp_stop_one_bit."""
        if self.pos >= self.nbits:
            return False
        stop = getattr(self, "stop_bit", None)
        if stop is None:
            stop = self.compute_stop_bit()
        return self.pos < stop


# ---------------------------------------------------------------------------
# SPS / PPS
# ---------------------------------------------------------------------------


@dataclass
class SPS:
    sps_id: int
    profile_idc: int
    level_idc: int
    log2_max_frame_num: int
    pic_order_cnt_type: int
    log2_max_pic_order_cnt_lsb: int
    delta_pic_order_always_zero_flag: int
    frame_mbs_only_flag: int
    pic_width_in_mbs: int
    pic_height_in_mbs: int
    chroma_format_idc: int = 1  # 4:2:0 for baseline


@dataclass
class PPS:
    pps_id: int
    sps_id: int
    entropy_coding_mode_flag: int
    bottom_field_pic_order_in_frame_present_flag: int
    num_ref_idx_l0_default_active: int
    num_ref_idx_l1_default_active: int
    weighted_pred_flag: int
    weighted_bipred_idc: int
    pic_init_qp: int
    deblocking_filter_control_present_flag: int
    constrained_intra_pred_flag: int
    redundant_pic_cnt_present_flag: int
    num_slice_groups: int = 1


def parse_sps(reader: BitReader, record: "_Recorder") -> SPS:
    profile_idc = record.u("profile_idc", reader, 8, Category.SPS)
    record.u("constraint_set_flags+reserved", reader, 8, Category.SPS)
    level_idc = record.u("level_idc", reader, 8, Category.SPS)
    sps_id = record.ue("seq_parameter_set_id", reader, Category.SPS)
    chroma_format_idc = 1
    if profile_idc in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        chroma_format_idc = record.ue("chroma_format_idc", reader, Category.SPS)
        if chroma_format_idc == 3:
            record.u("separate_colour_plane_flag", reader, 1, Category.SPS)
        record.ue("bit_depth_luma_minus8", reader, Category.SPS)
        record.ue("bit_depth_chroma_minus8", reader, Category.SPS)
        record.u("qpprime_y_zero_transform_bypass_flag", reader, 1, Category.SPS)
        seq_scaling_matrix_present = record.u(
            "seq_scaling_matrix_present_flag", reader, 1, Category.SPS
        )
        if seq_scaling_matrix_present:
            raise _Unsupported("seq_scaling_matrix_present")
    log2_max_frame_num = (
        record.ue("log2_max_frame_num_minus4", reader, Category.SPS) + 4
    )
    pic_order_cnt_type = record.ue("pic_order_cnt_type", reader, Category.SPS)
    log2_max_poc_lsb = 0
    delta_pic_order_always_zero_flag = 0
    if pic_order_cnt_type == 0:
        log2_max_poc_lsb = (
            record.ue("log2_max_pic_order_cnt_lsb_minus4", reader, Category.SPS) + 4
        )
    elif pic_order_cnt_type == 1:
        delta_pic_order_always_zero_flag = record.u(
            "delta_pic_order_always_zero_flag", reader, 1, Category.SPS
        )
        record.se("offset_for_non_ref_pic", reader, Category.SPS)
        record.se("offset_for_top_to_bottom_field", reader, Category.SPS)
        num_ref_frames_in_poc_cycle = record.ue(
            "num_ref_frames_in_pic_order_cnt_cycle", reader, Category.SPS
        )
        for _ in range(num_ref_frames_in_poc_cycle):
            record.se("offset_for_ref_frame", reader, Category.SPS)
    record.ue("max_num_ref_frames", reader, Category.SPS)
    record.u("gaps_in_frame_num_value_allowed_flag", reader, 1, Category.SPS)
    pic_width_in_mbs = record.ue("pic_width_in_mbs_minus1", reader, Category.SPS) + 1
    pic_height_in_map_units = (
        record.ue("pic_height_in_map_units_minus1", reader, Category.SPS) + 1
    )
    frame_mbs_only_flag = record.u("frame_mbs_only_flag", reader, 1, Category.SPS)
    if not frame_mbs_only_flag:
        raise _Unsupported("interlaced (frame_mbs_only_flag=0)")
    pic_height_in_mbs = (
        pic_height_in_map_units  # frame_mbs_only => map units == MB rows
    )
    record.u("direct_8x8_inference_flag", reader, 1, Category.SPS)
    frame_cropping = record.u("frame_cropping_flag", reader, 1, Category.SPS)
    if frame_cropping:
        record.ue("frame_crop_left_offset", reader, Category.SPS)
        record.ue("frame_crop_right_offset", reader, Category.SPS)
        record.ue("frame_crop_top_offset", reader, Category.SPS)
        record.ue("frame_crop_bottom_offset", reader, Category.SPS)
    vui_present = record.u("vui_parameters_present_flag", reader, 1, Category.SPS)
    # VUI is not needed for slice-data parsing; cover the remaining RBSP bytes as
    # one opaque span so coverage holds without a full VUI parse.
    if vui_present:
        record.fill_to_trailing("vui_parameters", reader, Category.SPS)
    return SPS(
        sps_id=sps_id,
        profile_idc=profile_idc,
        level_idc=level_idc,
        log2_max_frame_num=log2_max_frame_num,
        pic_order_cnt_type=pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb=log2_max_poc_lsb,
        delta_pic_order_always_zero_flag=delta_pic_order_always_zero_flag,
        frame_mbs_only_flag=frame_mbs_only_flag,
        pic_width_in_mbs=pic_width_in_mbs,
        pic_height_in_mbs=pic_height_in_mbs,
        chroma_format_idc=chroma_format_idc,
    )


def parse_pps(reader: BitReader, record: "_Recorder") -> PPS:
    pps_id = record.ue("pic_parameter_set_id", reader, Category.PPS)
    sps_id = record.ue("seq_parameter_set_id", reader, Category.PPS)
    entropy_coding_mode_flag = record.u(
        "entropy_coding_mode_flag", reader, 1, Category.PPS
    )
    if entropy_coding_mode_flag:
        raise _Unsupported("CABAC (entropy_coding_mode_flag=1)")
    bottom_field = record.u(
        "bottom_field_pic_order_in_frame_present_flag", reader, 1, Category.PPS
    )
    num_slice_groups = record.ue("num_slice_groups_minus1", reader, Category.PPS) + 1
    if num_slice_groups > 1:
        raise _Unsupported("multiple slice groups (FMO)")
    num_ref_l0 = (
        record.ue("num_ref_idx_l0_default_active_minus1", reader, Category.PPS) + 1
    )
    num_ref_l1 = (
        record.ue("num_ref_idx_l1_default_active_minus1", reader, Category.PPS) + 1
    )
    weighted_pred = record.u("weighted_pred_flag", reader, 1, Category.PPS)
    weighted_bipred = record.u("weighted_bipred_idc", reader, 2, Category.PPS)
    pic_init_qp = record.se("pic_init_qp_minus26", reader, Category.PPS) + 26
    record.se("pic_init_qs_minus26", reader, Category.PPS)
    record.se("chroma_qp_index_offset", reader, Category.PPS)
    deblocking = record.u(
        "deblocking_filter_control_present_flag", reader, 1, Category.PPS
    )
    constrained_intra = record.u("constrained_intra_pred_flag", reader, 1, Category.PPS)
    redundant_pic_cnt = record.u(
        "redundant_pic_cnt_present_flag", reader, 1, Category.PPS
    )
    if reader.more_rbsp_data():
        # transform_8x8_mode_flag etc. (high profile); not in baseline. Cover it.
        record.fill_to_trailing("pps_extension", reader, Category.PPS)
    return PPS(
        pps_id=pps_id,
        sps_id=sps_id,
        entropy_coding_mode_flag=entropy_coding_mode_flag,
        bottom_field_pic_order_in_frame_present_flag=bottom_field,
        num_ref_idx_l0_default_active=num_ref_l0,
        num_ref_idx_l1_default_active=num_ref_l1,
        weighted_pred_flag=weighted_pred,
        weighted_bipred_idc=weighted_bipred,
        pic_init_qp=pic_init_qp,
        deblocking_filter_control_present_flag=deblocking,
        constrained_intra_pred_flag=constrained_intra,
        redundant_pic_cnt_present_flag=redundant_pic_cnt,
        num_slice_groups=num_slice_groups,
    )


# ---------------------------------------------------------------------------
# Slice header
# ---------------------------------------------------------------------------


@dataclass
class SliceHeader:
    first_mb_in_slice: int
    slice_type: int  # already mod-5
    pps_id: int
    frame_num: int
    field_pic_flag: int
    idr_pic_id: int | None
    slice_qp: int
    num_ref_idx_l0_active: int


def parse_slice_header(
    reader: BitReader, record: "_Recorder", nal: NALInfo, sps: SPS, pps: PPS
) -> SliceHeader:
    first_mb = record.ue("first_mb_in_slice", reader, Category.SLICE_HEADER)
    slice_type_raw = record.ue("slice_type", reader, Category.SLICE_HEADER)
    slice_type = slice_type_raw % 5
    pps_id = record.ue("pic_parameter_set_id", reader, Category.SLICE_HEADER)
    frame_num = record.u(
        "frame_num", reader, sps.log2_max_frame_num, Category.SLICE_HEADER
    )
    field_pic_flag = 0
    if not sps.frame_mbs_only_flag:
        field_pic_flag = record.u("field_pic_flag", reader, 1, Category.SLICE_HEADER)
        if field_pic_flag:
            raise _Unsupported("field coding")
    idr_pic_id = None
    if nal.nal_type == NAL_SLICE_IDR:
        idr_pic_id = record.ue("idr_pic_id", reader, Category.SLICE_HEADER)
    if sps.pic_order_cnt_type == 0:
        record.u(
            "pic_order_cnt_lsb",
            reader,
            sps.log2_max_pic_order_cnt_lsb,
            Category.SLICE_HEADER,
        )
        if pps.bottom_field_pic_order_in_frame_present_flag and not field_pic_flag:
            record.se("delta_pic_order_cnt_bottom", reader, Category.SLICE_HEADER)
    elif sps.pic_order_cnt_type == 1 and not sps.delta_pic_order_always_zero_flag:
        record.se("delta_pic_order_cnt[0]", reader, Category.SLICE_HEADER)
        if pps.bottom_field_pic_order_in_frame_present_flag and not field_pic_flag:
            record.se("delta_pic_order_cnt[1]", reader, Category.SLICE_HEADER)
    if pps.redundant_pic_cnt_present_flag:
        record.ue("redundant_pic_cnt", reader, Category.SLICE_HEADER)
    num_ref_idx_l0_active = pps.num_ref_idx_l0_default_active
    if slice_type == SLICE_TYPE_P or slice_type == SLICE_TYPE_SP:
        override = record.u(
            "num_ref_idx_active_override_flag", reader, 1, Category.SLICE_HEADER
        )
        if override:
            num_ref_idx_l0_active = (
                record.ue("num_ref_idx_l0_active_minus1", reader, Category.SLICE_HEADER)
                + 1
            )
        _parse_ref_pic_list_modification(reader, record)
    if nal.ref_idc != 0:
        _parse_dec_ref_pic_marking(reader, record, nal)
    slice_qp_delta = record.se("slice_qp_delta", reader, Category.SLICE_HEADER)
    slice_qp = pps.pic_init_qp + slice_qp_delta
    if pps.deblocking_filter_control_present_flag:
        idc = record.ue("disable_deblocking_filter_idc", reader, Category.SLICE_HEADER)
        if idc != 1:
            record.se("slice_alpha_c0_offset_div2", reader, Category.SLICE_HEADER)
            record.se("slice_beta_offset_div2", reader, Category.SLICE_HEADER)
    return SliceHeader(
        first_mb_in_slice=first_mb,
        slice_type=slice_type,
        pps_id=pps_id,
        frame_num=frame_num,
        field_pic_flag=field_pic_flag,
        idr_pic_id=idr_pic_id,
        slice_qp=slice_qp,
        num_ref_idx_l0_active=num_ref_idx_l0_active,
    )


def _parse_ref_pic_list_modification(reader: BitReader, record: "_Recorder") -> None:
    flag = record.u(
        "ref_pic_list_modification_flag_l0", reader, 1, Category.SLICE_HEADER
    )
    if flag:
        while True:
            idc = record.ue(
                "modification_of_pic_nums_idc", reader, Category.SLICE_HEADER
            )
            if idc == 3:
                break
            if idc in (0, 1):
                record.ue("abs_diff_pic_num_minus1", reader, Category.SLICE_HEADER)
            elif idc == 2:
                record.ue("long_term_pic_num", reader, Category.SLICE_HEADER)


def _parse_dec_ref_pic_marking(
    reader: BitReader, record: "_Recorder", nal: NALInfo
) -> None:
    if nal.nal_type == NAL_SLICE_IDR:
        record.u("no_output_of_prior_pics_flag", reader, 1, Category.SLICE_HEADER)
        record.u("long_term_reference_flag", reader, 1, Category.SLICE_HEADER)
    else:
        adaptive = record.u(
            "adaptive_ref_pic_marking_mode_flag", reader, 1, Category.SLICE_HEADER
        )
        if adaptive:
            while True:
                op = record.ue(
                    "memory_management_control_operation", reader, Category.SLICE_HEADER
                )
                if op == 0:
                    break
                if op in (1, 3):
                    record.ue(
                        "difference_of_pic_nums_minus1", reader, Category.SLICE_HEADER
                    )
                if op == 2:
                    record.ue("long_term_pic_num", reader, Category.SLICE_HEADER)
                if op in (3, 6):
                    record.ue("long_term_frame_idx", reader, Category.SLICE_HEADER)
                if op == 4:
                    record.ue(
                        "max_long_term_frame_idx_plus1", reader, Category.SLICE_HEADER
                    )


# ---------------------------------------------------------------------------
# Recorder: tracks bit ranges and converts to Annex-B byte spans
# ---------------------------------------------------------------------------


class _Unsupported(Exception):
    pass


class _Recorder:
    """Reads syntax elements through a BitReader and records SyntaxSpans.

    Bit positions are RBSP bit offsets; byte positions are mapped back to the
    Annex-B stream via ``byte_map``.
    """

    def __init__(self, byte_map: list[int], rbsp_len: int) -> None:
        self.byte_map = byte_map
        self.rbsp_len = rbsp_len
        self.spans: list[SyntaxSpan] = []

    # -- low-level element readers that also record a span --
    def _span(
        self, name: str, cat: Category, b0: int, b1: int, value, mb_addr=None
    ) -> None:
        byte_start = self.byte_map[b0 >> 3]
        last_bit = b1 - 1
        byte_end = self.byte_map[last_bit >> 3] + 1
        self.spans.append(
            SyntaxSpan(name, cat, b0, b1, byte_start, byte_end, value, mb_addr)
        )

    def u(self, name, reader: BitReader, n: int, cat: Category, mb_addr=None) -> int:
        b0 = reader.pos
        v = reader.read_bits(n)
        self._span(name, cat, b0, reader.pos, v, mb_addr)
        return v

    def ue(self, name, reader: BitReader, cat: Category, mb_addr=None) -> int:
        b0 = reader.pos
        v = reader.read_ue()
        self._span(name, cat, b0, reader.pos, v, mb_addr)
        return v

    def se(self, name, reader: BitReader, cat: Category, mb_addr=None) -> int:
        b0 = reader.pos
        v = reader.read_se()
        self._span(name, cat, b0, reader.pos, v, mb_addr)
        return v

    def te(
        self, name, reader: BitReader, x_max: int, cat: Category, mb_addr=None
    ) -> int:
        b0 = reader.pos
        v = reader.read_te(x_max)
        self._span(name, cat, b0, reader.pos, v, mb_addr)
        return v

    def raw(self, name, cat: Category, b0: int, b1: int, value, mb_addr=None) -> None:
        """Record a span for bits already consumed elsewhere (e.g. residual)."""
        self._span(name, cat, b0, b1, value, mb_addr)

    def fill_to_trailing(self, name, reader: BitReader, cat: Category) -> None:
        """Consume up to and including rbsp_stop_one_bit as one opaque span."""
        b0 = reader.pos
        # Advance to the rbsp_stop_one_bit boundary.
        last_one = -1
        for p in range(reader.nbits - 1, -1, -1):
            if (reader.rbsp[p >> 3] >> (7 - (p & 7))) & 1:
                last_one = p
                break
        reader.pos = reader.nbits
        if last_one >= b0:
            self._span(name, cat, b0, last_one + 1, None)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass
class StreamParse:
    nals: list[NALParse]
    sps: dict[int, SPS]
    pps: dict[int, PPS]

    def all_spans(self) -> list[SyntaxSpan]:
        out: list[SyntaxSpan] = []
        for np_ in self.nals:
            out.extend(np_.spans)
        return out


def _emit_trailing(record: _Recorder, reader: BitReader) -> None:
    """Record the rbsp_stop_one_bit + zero padding as one span (if present)."""
    if reader.pos >= reader.nbits:
        return
    b0 = reader.pos
    record.raw("rbsp_trailing_bits", Category.RBSP_TRAILING, b0, reader.nbits, None)
    reader.pos = reader.nbits


def parse_nal(
    data: bytes,
    nal: NALInfo,
    sps_map: dict[int, SPS],
    pps_map: dict[int, PPS],
    *,
    parse_slice_data: bool = True,
) -> NALParse:
    spans: list[SyntaxSpan] = []
    # Start code + NAL header live in Annex-B byte space (not RBSP bits).
    spans.append(
        SyntaxSpan(
            "start_code",
            Category.START_CODE,
            -1,
            -1,
            nal.start_code_start,
            nal.payload_start,
            None,
        )
    )
    spans.append(
        SyntaxSpan(
            "nal_header",
            Category.NAL_HEADER,
            -1,
            -1,
            nal.payload_start,
            nal.payload_start + 1,
            {"nal_type": nal.nal_type, "ref_idc": nal.ref_idc},
        )
    )

    rbsp, byte_map, epb_offsets = unescape_rbsp(
        data, nal.payload_start + 1, nal.payload_end
    )
    for off in epb_offsets:
        spans.append(
            SyntaxSpan(
                "emulation_prevention_three_byte",
                Category.EMULATION_PREVENTION,
                -1,
                -1,
                off,
                off + 1,
                0x03,
            )
        )

    result = NALParse(nal=nal, status=ParseStatus.OK, spans=spans)
    if not byte_map:
        return result

    reader = BitReader(rbsp)
    record = _Recorder(byte_map, len(rbsp))

    try:
        if nal.nal_type == NAL_SPS:
            sps = parse_sps(reader, record)
            sps_map[sps.sps_id] = sps
            _emit_trailing(record, reader)
        elif nal.nal_type == NAL_PPS:
            pps = parse_pps(reader, record)
            pps_map[pps.pps_id] = pps
            _emit_trailing(record, reader)
        elif nal.nal_type in VCL_NAL_TYPES:
            _parse_slice(
                data, nal, reader, record, sps_map, pps_map, result, parse_slice_data
            )
        else:
            # SEI / AUD / other: opaque payload span, no bit parse.
            cat = Category.SEI if nal.nal_type == NAL_SEI else Category.UNKNOWN
            record.fill_to_trailing(f"nal_type_{nal.nal_type}_payload", reader, cat)
    except _Unsupported as exc:
        result.status = ParseStatus.UNSUPPORTED
        result.reason = str(exc)
        result.desync_bit = reader.pos
        record.fill_to_trailing("unsupported_remainder", reader, Category.UNKNOWN)
    except (BitReaderError, _DesyncError, ValueError) as exc:
        # ValueError covers a CAVLC VLC-table miss (h264_cavlc_tables.decode_vlc)
        # on an invalid bitstream -- e.g. the model's free-run output. That is a
        # desync, not a crash: report the location like any other desync.
        result.status = ParseStatus.DESYNC
        result.reason = str(exc)
        result.desync_bit = reader.pos
        if reader.pos < reader.nbits and byte_map:
            result.desync_byte = byte_map[min(reader.pos, reader.nbits - 1) >> 3]

    result.spans.extend(record.spans)
    result.consumed_bits = reader.pos
    result.rbsp_bits_total = reader.nbits
    return result


def _parse_slice(
    data: bytes,
    nal: NALInfo,
    reader: BitReader,
    record: _Recorder,
    sps_map: dict[int, SPS],
    pps_map: dict[int, PPS],
    result: NALParse,
    parse_slice_data: bool,
) -> None:
    # Peek pps_id to resolve sps/pps (re-read cleanly inside parse_slice_header).
    save = reader.pos
    _ = reader.read_ue()  # first_mb
    _ = reader.read_ue()  # slice_type
    pps_id = reader.read_ue()
    reader.pos = save
    pps = pps_map.get(pps_id)
    if pps is None:
        raise _DesyncError(f"unknown pps_id {pps_id}")
    sps = sps_map.get(pps.sps_id)
    if sps is None:
        raise _DesyncError(f"unknown sps_id {pps.sps_id}")

    header = parse_slice_header(reader, record, nal, sps, pps)
    if header.slice_type not in (SLICE_TYPE_P, SLICE_TYPE_I):
        raise _Unsupported(f"slice_type {header.slice_type}")

    if not parse_slice_data:
        record.fill_to_trailing("slice_data", reader, Category.SLICE_DATA)
        result.mb_count = None
        return

    from_module = _parse_slice_data_cavlc  # forward ref defined below
    mb_count = from_module(reader, record, sps, pps, header)
    result.mb_count = mb_count
    _emit_trailing(record, reader)


def parse_stream(data: bytes, *, parse_slice_data: bool = True) -> StreamParse:
    sps_map: dict[int, SPS] = {}
    pps_map: dict[int, PPS] = {}
    parses: list[NALParse] = []
    for nal in iter_nals(data):
        parses.append(
            parse_nal(data, nal, sps_map, pps_map, parse_slice_data=parse_slice_data)
        )
    return StreamParse(nals=parses, sps=sps_map, pps=pps_map)


class _DesyncError(Exception):
    pass


# ---------------------------------------------------------------------------
# Macroblock layer + CAVLC residual (Constrained Baseline)
# ---------------------------------------------------------------------------

# P macroblock partition prediction: mb_type -> (num_parts, name)
_P_MB = {
    0: (1, "P_L0_16x16"),
    1: (2, "P_L0_L0_16x8"),
    2: (2, "P_L0_L0_8x16"),
    3: (4, "P_8x8"),
    4: (4, "P_8x8ref0"),
}
_SUB_MB_NUM_PARTS = {0: 1, 1: 2, 2: 2, 3: 4}  # P sub_mb_type -> sub-partitions


class _NNZ:
    """Per-4x4-block non-zero-coefficient grids for nC neighbour prediction."""

    def __init__(self, width_mb: int, height_mb: int) -> None:
        self.lw = width_mb * 4
        self.luma = [[-1] * (width_mb * 4) for _ in range(height_mb * 4)]
        self.cw = width_mb * 2
        self.chroma = [
            [[-1] * (width_mb * 2) for _ in range(height_mb * 2)] for _ in range(2)
        ]

    @staticmethod
    def _predict(grid, x: int, y: int) -> int:
        n_a = grid[y][x - 1] if x > 0 else -1
        n_b = grid[y - 1][x] if y > 0 else -1
        avail_a, avail_b = n_a >= 0, n_b >= 0
        if avail_a and avail_b:
            return (n_a + n_b + 1) >> 1
        if avail_a:
            return n_a
        if avail_b:
            return n_b
        return 0

    def luma_coords(self, mbx: int, mby: int, blk: int) -> tuple[int, int]:
        q, r = blk >> 2, blk & 3
        return mbx * 4 + (q & 1) * 2 + (r & 1), mby * 4 + (q >> 1) * 2 + (r >> 1)

    def chroma_coords(self, mbx: int, mby: int, blk: int) -> tuple[int, int]:
        return mbx * 2 + (blk & 1), mby * 2 + (blk >> 1)

    def predict_luma(self, mbx, mby, blk):
        x, y = self.luma_coords(mbx, mby, blk)
        return self._predict(self.luma, x, y)

    def set_luma(self, mbx, mby, blk, val):
        x, y = self.luma_coords(mbx, mby, blk)
        self.luma[y][x] = val

    def predict_chroma(self, comp, mbx, mby, blk):
        x, y = self.chroma_coords(mbx, mby, blk)
        return self._predict(self.chroma[comp], x, y)

    def set_chroma(self, comp, mbx, mby, blk, val):
        x, y = self.chroma_coords(mbx, mby, blk)
        self.chroma[comp][y][x] = val

    def set_mb(self, mbx, mby, luma_val, chroma_val):
        for b in range(16):
            self.set_luma(mbx, mby, b, luma_val)
        for c in range(2):
            for b in range(4):
                self.set_chroma(c, mbx, mby, b, chroma_val)


def _residual_block(
    reader: BitReader,
    record: _Recorder,
    nc: int,
    max_coeff: int,
    mb_addr: int,
    cat: Category,
    name: str,
) -> int:
    """Parse one CAVLC residual block; record sub-element spans; return TotalCoeff."""
    label = T.coeff_token_label(nc)
    b0 = reader.pos
    (total_coeff, trailing_ones), _ = T.decode_vlc(
        reader.read_bit, T.code_map(label), label
    )
    record.raw(
        f"{name}.coeff_token",
        cat,
        b0,
        reader.pos,
        {"total_coeff": total_coeff, "trailing_ones": trailing_ones, "nC": nc},
        mb_addr,
    )
    if total_coeff == 0:
        return 0
    if trailing_ones > 0:
        s0 = reader.pos
        reader.read_bits(trailing_ones)
        record.raw(
            f"{name}.trailing_ones_sign_flag",
            cat,
            s0,
            reader.pos,
            trailing_ones,
            mb_addr,
        )
    suffix_length = 1 if (total_coeff > 10 and trailing_ones < 3) else 0
    for i in range(total_coeff - trailing_ones):
        ls = reader.pos
        level_prefix = 0
        while reader.read_bit() == 0:
            level_prefix += 1
            if level_prefix > 60:
                raise _DesyncError("level_prefix overflow")
        if level_prefix == 14 and suffix_length == 0:
            suffix_size = 4
        elif level_prefix >= 15:
            suffix_size = level_prefix - 3
        else:
            suffix_size = suffix_length
        level_suffix = reader.read_bits(suffix_size) if suffix_size > 0 else 0
        level_code = (min(15, level_prefix) << suffix_length) + level_suffix
        if level_prefix >= 15 and suffix_length == 0:
            level_code += 15
        if level_prefix >= 16:
            level_code += (1 << (level_prefix - 3)) - 4096
        if i == 0 and trailing_ones < 3:
            level_code += 2
        level = (level_code + 2) >> 1 if level_code % 2 == 0 else (-level_code - 1) >> 1
        if suffix_length == 0:
            suffix_length = 1
        if abs(level) > (3 << (suffix_length - 1)) and suffix_length < 6:
            suffix_length += 1
        record.raw(f"{name}.level[{i}]", cat, ls, reader.pos, level, mb_addr)
    total_zeros = 0
    if total_coeff < max_coeff:
        tz0 = reader.pos
        tz_label = (
            f"total_zeros_cdc_{total_coeff}"
            if max_coeff == 4
            else f"total_zeros_4x4_{total_coeff}"
        )
        total_zeros, _ = T.decode_vlc(reader.read_bit, T.code_map(tz_label), tz_label)
        record.raw(f"{name}.total_zeros", cat, tz0, reader.pos, total_zeros, mb_addr)
    zeros_left = total_zeros
    for i in range(total_coeff - 1):
        if zeros_left <= 0:
            break
        rb0 = reader.pos
        rb_label = f"run_before_{min(zeros_left, 7)}"
        run, _ = T.decode_vlc(reader.read_bit, T.code_map(rb_label), rb_label)
        record.raw(f"{name}.run_before[{i}]", cat, rb0, reader.pos, run, mb_addr)
        zeros_left -= run
    return total_coeff


def _parse_residual(
    reader,
    record,
    nnz: _NNZ,
    mbx,
    mby,
    mb_addr,
    *,
    intra16x16: bool,
    cbp_luma: int,
    cbp_chroma: int,
) -> None:
    lcat, ccat = Category.RESIDUAL_LUMA, Category.RESIDUAL_CHROMA
    if intra16x16:
        nc = nnz.predict_luma(mbx, mby, 0)
        _residual_block(reader, record, nc, 16, mb_addr, lcat, "luma_dc")
        for blk in range(16):
            if cbp_luma & (1 << (blk >> 2)):
                nc = nnz.predict_luma(mbx, mby, blk)
                tc = _residual_block(
                    reader, record, nc, 15, mb_addr, lcat, f"luma_ac[{blk}]"
                )
            else:
                tc = 0
            nnz.set_luma(mbx, mby, blk, tc)
    else:
        for blk in range(16):
            if cbp_luma & (1 << (blk >> 2)):
                nc = nnz.predict_luma(mbx, mby, blk)
                tc = _residual_block(
                    reader, record, nc, 16, mb_addr, lcat, f"luma[{blk}]"
                )
            else:
                tc = 0
            nnz.set_luma(mbx, mby, blk, tc)
    # chroma 4:2:0: cbp_chroma 0=none, 1=DC, 2=DC+AC
    if cbp_chroma in (1, 2):
        for comp in range(2):
            _residual_block(reader, record, -1, 4, mb_addr, ccat, f"chroma_dc[{comp}]")
    if cbp_chroma == 2:
        for comp in range(2):
            for blk in range(4):
                nc = nnz.predict_chroma(comp, mbx, mby, blk)
                tc = _residual_block(
                    reader, record, nc, 15, mb_addr, ccat, f"chroma_ac[{comp}][{blk}]"
                )
                nnz.set_chroma(comp, mbx, mby, blk, tc)
    else:
        for comp in range(2):
            for blk in range(4):
                nnz.set_chroma(comp, mbx, mby, blk, 0)


def _parse_macroblock(
    reader, record, sps: SPS, pps: PPS, header: SliceHeader, nnz: _NNZ, mb_addr: int
) -> None:
    mbx, mby = mb_addr % sps.pic_width_in_mbs, mb_addr // sps.pic_width_in_mbs
    mb_type = record.ue("mb_type", reader, Category.MB_HEADER, mb_addr)

    is_p_slice = header.slice_type == SLICE_TYPE_P
    inter = False
    intra16x16 = False
    i16_pred_cbp = None
    if is_p_slice and mb_type < 5:
        inter = True
        num_parts, _name = _P_MB[mb_type]
    else:
        i_mb_type = mb_type - 5 if is_p_slice else mb_type
        if i_mb_type == 0:
            mb_mode = "I_NxN"
        elif i_mb_type == 25:
            mb_mode = "I_PCM"
        elif 1 <= i_mb_type <= 24:
            mb_mode = "I_16x16"
            intra16x16 = True
            idx = i_mb_type - 1
            i16_pred_cbp = (idx % 4, (idx // 4) % 3, 15 if idx >= 12 else 0)
        else:
            raise _DesyncError(f"bad intra mb_type {i_mb_type}")

    if not inter and mb_mode == "I_PCM":
        while not reader.byte_aligned():
            reader.read_bit()
        p0 = reader.pos
        reader.read_bits(256 * 8 + 2 * 64 * 8)  # 4:2:0 8-bit samples
        record.raw("pcm_samples", Category.MB_PRED, p0, reader.pos, None, mb_addr)
        nnz.set_mb(mbx, mby, 16, 16)
        return

    if inter:
        _parse_inter_pred(reader, record, header, mb_type, mb_addr)
        cbp = T.GOLOMB_TO_INTER_CBP[
            record.ue("coded_block_pattern", reader, Category.CBP, mb_addr)
        ]
        cbp_luma, cbp_chroma = cbp & 15, cbp >> 4
    elif mb_mode == "I_NxN":
        for blk in range(16):
            prev = record.u(
                f"prev_intra4x4_pred_mode_flag[{blk}]",
                reader,
                1,
                Category.MB_PRED,
                mb_addr,
            )
            if not prev:
                record.u(
                    f"rem_intra4x4_pred_mode[{blk}]",
                    reader,
                    3,
                    Category.MB_PRED,
                    mb_addr,
                )
        record.ue("intra_chroma_pred_mode", reader, Category.MB_PRED, mb_addr)
        cbp = T.GOLOMB_TO_INTRA_CBP[
            record.ue("coded_block_pattern", reader, Category.CBP, mb_addr)
        ]
        cbp_luma, cbp_chroma = cbp & 15, cbp >> 4
    else:  # I_16x16
        record.ue("intra_chroma_pred_mode", reader, Category.MB_PRED, mb_addr)
        _, cbp_chroma, cbp_luma = i16_pred_cbp

    if cbp_luma > 0 or cbp_chroma > 0 or intra16x16:
        record.se("mb_qp_delta", reader, Category.MB_QP_DELTA, mb_addr)
        _parse_residual(
            reader,
            record,
            nnz,
            mbx,
            mby,
            mb_addr,
            intra16x16=intra16x16,
            cbp_luma=cbp_luma,
            cbp_chroma=cbp_chroma,
        )
    else:
        nnz.set_mb(mbx, mby, 0, 0)


def _parse_inter_pred(
    reader, record, header: SliceHeader, mb_type: int, mb_addr: int
) -> None:
    num_parts, name = _P_MB[mb_type]
    num_ref = header.num_ref_idx_l0_active
    if mb_type in (3, 4):  # P_8x8 / P_8x8ref0
        sub_types = []
        for p in range(4):
            st = record.ue(f"sub_mb_type[{p}]", reader, Category.MB_HEADER, mb_addr)
            sub_types.append(st)
        # ref_idx per 8x8 (not for ref0), then mvd per sub-partition.
        if mb_type == 3 and num_ref > 1:
            for p in range(4):
                record.te(
                    f"ref_idx_l0[{p}]", reader, num_ref - 1, Category.MB_PRED, mb_addr
                )
        for p in range(4):
            for _ in range(_SUB_MB_NUM_PARTS[sub_types[p]]):
                record.se(f"mvd_l0[{p}].x", reader, Category.MB_PRED, mb_addr)
                record.se(f"mvd_l0[{p}].y", reader, Category.MB_PRED, mb_addr)
        return
    if num_ref > 1:
        for p in range(num_parts):
            record.te(
                f"ref_idx_l0[{p}]", reader, num_ref - 1, Category.MB_PRED, mb_addr
            )
    for p in range(num_parts):
        record.se(f"mvd_l0[{p}].x", reader, Category.MB_PRED, mb_addr)
        record.se(f"mvd_l0[{p}].y", reader, Category.MB_PRED, mb_addr)


def _parse_slice_data_cavlc(
    reader: BitReader, record: _Recorder, sps: SPS, pps: PPS, header: SliceHeader
) -> int:
    pic_size = sps.pic_width_in_mbs * sps.pic_height_in_mbs
    nnz = _NNZ(sps.pic_width_in_mbs, sps.pic_height_in_mbs)
    reader.compute_stop_bit()
    curr = header.first_mb_in_slice
    is_p = header.slice_type == SLICE_TYPE_P
    more = True
    while more and curr < pic_size:
        if is_p:
            run = record.ue("mb_skip_run", reader, Category.MB_HEADER, curr)
            for _ in range(run):
                if curr >= pic_size:
                    break
                mbx, mby = curr % sps.pic_width_in_mbs, curr // sps.pic_width_in_mbs
                nnz.set_mb(mbx, mby, 0, 0)  # P_Skip: all nnz 0
                curr += 1
            if run > 0:
                more = reader.more_rbsp_data()
            if not more or curr >= pic_size:
                break
        _parse_macroblock(reader, record, sps, pps, header, nnz, curr)
        curr += 1
        more = reader.more_rbsp_data()
    return curr - header.first_mb_in_slice
