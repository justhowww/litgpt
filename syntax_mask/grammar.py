"""Step 1 -- the next-syntax-field oracle for one NAL unit's RBSP.

The parser state is always "positioned before (or inside) one physical syntax
field".  ``_expect(name, code, handler)`` is the oracle's output: the field id,
its code (coding + legal-value set, see ``codes.py``), and the handler -- the
value-dependent transition that applies the O(1) state update and selects the
next field (absent fields are skipped inside handlers, i.e. epsilon edges are
collapsed).  Handlers are plain functions ``h(parser, value)`` so a parser can be
cloned by copying its attribute dict.

Two optional annotations let the mask builder share work across values:

* ``vindep=True``  -- the handler's future does not depend on the value
  (e.g. mvd, trailing-one signs, PCM samples, SEI payload bytes);
* ``vclass=f``     -- ``f(parser, value)`` names an equivalence class of values
  that lead to the same future (e.g. Intra16x16 mb_types with the same CBP).

Scope: Baseline/Constrained-Baseline, CAVLC, progressive, I/P slices, 4:2:0
8-bit, one slice group.  Stream-level context (parameter sets, picture tracking)
lives in :class:`Ctx` and is only mutated when a NAL unit is committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import codes as C
from . import cavlc_tables as T
from .codes import iv, ivset, u, ue, ue_range, se, me, te, vlc, unary

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

NAL_SLICE, NAL_IDR, NAL_SEI, NAL_SPS, NAL_PPS = 1, 5, 6, 7, 8
NAL_AUD, NAL_EOSEQ, NAL_EOSTREAM, NAL_FILLER = 9, 10, 11, 12
VCL = (NAL_SLICE, NAL_IDR)
DEFAULT_NAL_TYPES = frozenset({1, 5, 6, 7, 8, 9, 10, 11, 12})

# Table A-1: level_idc -> (MaxFS, MaxDpbMbs).  level_idc 9/11 (1b) use the
# larger of the two interpretations, which keeps the check sound.
LEVELS = {
    9: (99, 396), 10: (99, 396), 11: (396, 900), 12: (396, 2376),
    13: (396, 2376), 20: (396, 2376), 21: (792, 4752), 22: (1620, 8100),
    30: (1620, 8100), 31: (3600, 18000), 32: (5120, 20480), 40: (8192, 32768),
    41: (8192, 32768), 42: (8704, 34816), 50: (22080, 110400),
    51: (36864, 184320), 52: (36864, 184320), 60: (139264, 696320),
    61: (139264, 696320), 62: (139264, 696320),
}

P_NUM_PARTS = (1, 2, 2)  # P_L0_16x16, P_L0_L0_16x8, P_L0_L0_8x16
SUB_NUM_PARTS = (1, 2, 2, 4)
MVD_LO, MVD_HI = -8192 * 4, 8192 * 4 - 1  # quarter-sample range, 7.4.5.1
LEVEL_PREFIX_MAX = 15  # Baseline/Main/Extended, 9.2.2.1

# MB status codes stored per macroblock of the current slice
ST_INTER, ST_I16, ST_I4, ST_PCM = 0, 1, 2, 3

# Intra sample requirements: bit0 = left, bit1 = top, bit2 = top-left
_NEED_I4 = (2, 1, 0, 2, 7, 7, 7, 2, 1)  # Intra4x4 modes 0..8
_NEED_I16 = (2, 1, 0, 7)  # V, H, DC, Plane
_NEED_CHROMA = (0, 1, 2, 7)  # DC, H, V, Plane


@dataclass
class Profile:
    """Project restrictions layered on top of H.264 Baseline legality."""

    slice_layout: str = "frame"  # "frame" (1 slice/picture) | "mb" (1 MB/slice)
    nal_types: frozenset = DEFAULT_NAL_TYPES
    # None = any payloadType (types are framed/opaque).  Type 128 is never
    # accepted after a first message: its 0x80 byte is read as rbsp trailing bits.
    sei_types: frozenset | None = None

    def __post_init__(self):
        if self.slice_layout not in ("frame", "mb"):
            raise ValueError("slice_layout must be 'frame' or 'mb'")


class Illegal(Exception):
    """A committed byte violates the grammar."""


# --------------------------------------------------------------------------
# stream context (committed state only)
# --------------------------------------------------------------------------


@dataclass
class Ctx:
    profile: Profile = field(default_factory=Profile)
    sps: dict = field(default_factory=dict)
    pps: dict = field(default_factory=dict)
    seen_idr: bool = False
    pic: dict | None = None  # open (incomplete) picture, "mb" layout only
    prev_ref_frame_num: int = 0
    prev_idr_pic_id: int | None = None  # set iff the previous picture was IDR
    need_idr: bool = False  # an SPS changed content: a new sequence must start
    active_sps: int | None = None  # SPS of the current coded video sequence
    # Reference frames in the DPB (sliding window since the last IDR); None
    # once adaptive marking (MMCO) makes the count unknown.
    refs: int | None = None

    def commit_sps(self, rec: dict) -> None:
        old = self.sps.get(rec["id"])
        if old is not None and _sps_core(old) != _sps_core(rec):
            self.need_idr = True
        self.sps[rec["id"]] = rec

    def slice_pps_ids(self, nal_type: int) -> tuple:
        """PPS ids a new slice of this NAL type may reference.

        The PPS's SPS must exist; a non-IDR slice must stay on the SPS that is
        active for the current coded video sequence (7.4.1.2.1).
        """
        ids = [i for i, p in self.pps.items() if p["sps_id"] in self.sps]
        if nal_type != NAL_IDR and self.active_sps is not None:
            ids = [i for i in ids if self.pps[i]["sps_id"] == self.active_sps]
        return tuple(sorted(ids))

    def allowed_nal_headers(self) -> tuple:
        vcl_ok = {t: bool(self.slice_pps_ids(t)) for t in VCL}
        pic = self.pic
        out = []
        for ref in range(4):
            for t in self.profile.nal_types:
                if ref and t in (6, 9, 10, 11, 12):
                    continue
                if t == NAL_IDR and not ref:
                    continue
                if t in VCL:
                    if not vcl_ok[t] or (
                        t == NAL_SLICE and (self.need_idr or not self.seen_idr)
                    ):
                        continue
                    if pic is not None and (
                        t != pic["nal_type"] or bool(ref) != pic["ref_nz"]
                    ):
                        continue
                # Project profile: parameter sets are only sent between pictures
                # (x264 repeats them before IDRs), never inside an open picture.
                elif pic is not None and t != NAL_FILLER:
                    continue
                out.append((ref << 5) | t)
        return ivset(out)

    def commit_slice(self, rec: dict) -> None:
        pic = self.pic
        if pic is None:
            pic = {
                "nal_type": rec["nal_type"],
                "ref_nz": rec["ref_idc"] != 0,
                "types": set(),
                "fixed_type": None,
                "marking": rec["marking"],
            }
            for key in ("pps_id", "frame_num", "idr_pic_id", "poc_lsb",
                        "poc_bottom", "poc0", "poc1"):
                if key in rec:
                    pic[key] = rec[key]
        base = rec["slice_type"] % 5
        pic["types"].add(base)
        if rec["slice_type"] >= 5:
            pic["fixed_type"] = base
        pic["next_mb"] = rec["end_mb"]
        if rec["end_mb"] >= rec["pic_size"]:
            if pic["ref_nz"]:
                self.prev_ref_frame_num = 0 if rec.get("mmco5") else pic["frame_num"]
            is_idr = pic["nal_type"] == NAL_IDR
            self.prev_idr_pic_id = pic.get("idr_pic_id") if is_idr else None
            self.seen_idr = self.seen_idr or is_idr
            self.need_idr = self.need_idr and not is_idr
            if is_idr:
                self.refs = 1
            elif pic["ref_nz"]:
                if pic["marking"][:1] == (1,):
                    self.refs = None
                elif self.refs is not None:
                    cap = max(1, self.sps[rec["sps_id"]]["max_num_ref_frames"])
                    self.refs = min(self.refs + 1, cap)
            self.active_sps = rec["sps_id"]
            self.pic = None
        else:
            self.pic = pic


_SPS_CORE = ("profile_idc", "level_idc", "log2_max_frame_num", "poc_type",
             "log2_max_poc_lsb", "delta_poc_always_zero", "max_num_ref_frames",
             "gaps", "W", "H")


def _sps_core(rec):
    return tuple(rec.get(k) for k in _SPS_CORE)


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


class Parser:
    """Clonable field-level RBSP parser for one NAL unit (header byte included)."""

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.pos = 0  # RBSP bits consumed (including the NAL header byte)
        self.done = False
        self.dead = False
        self.rec = {}  # copy-on-write field record (header values)
        self.ov = {}  # copy-on-write grid overlay (slice data)
        self.g = None  # committed per-slice grids
        self.q = ()  # immutable work queue (inter prediction, MMCO pinning)
        self.field = None
        self._expect("nal_header", u(8, ctx.allowed_nal_headers()), _nal_header)

    # ---- oracle plumbing -------------------------------------------------
    def _expect(self, name, code, handler, vindep=False, vclass=None):
        self.field = name
        self.code = code
        self.cs = code.init
        self.handler = handler
        self.vindep = vindep
        self.vclass = vclass
        if code.empty:
            self.dead = True

    def clone(self) -> "Parser":
        c = Parser.__new__(Parser)
        c.__dict__ = self.__dict__.copy()
        return c

    def finish(self, value, length: int) -> None:
        """Apply a completed field (``length`` bits of it read in this call)."""
        self.pos += length
        self.handler(self, value)

    def feed_bit(self, b: int) -> None:
        """Committed-path transition (no branching): one RBSP bit."""
        if self.done or self.dead:
            raise Illegal(f"bit after end/dead at field {self.field}")
        st, x = self.code.bit(self.cs, b)
        if st == C.MORE:
            self.pos += 1
            self.cs = x
        elif st == C.DONE:
            self.finish(x, 1)
            if self.dead:
                raise Illegal(f"value {x!r} rejected at field {self.field}")
        else:
            raise Illegal(f"bit {b} illegal in field {self.field}")

    def feed_byte(self, byte: int) -> None:
        for i in range(7, -1, -1):
            self.feed_bit((byte >> i) & 1)
        self.flush()

    def put(self, key, value) -> None:
        rec = dict(self.rec)
        rec[key] = value
        self.rec = rec

    def kill(self) -> None:
        self.dead = True

    # ---- grid (per-slice context for nC / intra prediction) -------------
    def gget(self, kind: int, idx: int) -> int:
        v = self.ov.get((kind, idx))
        if v is not None:
            return v
        g = self.g
        if g is None:
            return -1 if kind == 4 else 0
        return g[kind][idx]

    def gset(self, kind: int, idx: int, value: int) -> None:
        ov = dict(self.ov)
        ov[(kind, idx)] = value
        self.ov = ov

    def flush(self) -> None:
        if not self.ov:
            return
        g = self.g
        if g is None:
            n4 = 16 * self.W * self.H
            n2 = 4 * self.W * self.H
            g = self.g = [[0] * n4, [0] * n2, [0] * n2, [0] * n4, [-1] * (self.W * self.H)]
        for (kind, idx), v in self.ov.items():
            g[kind][idx] = v
        self.ov = {}


# ==========================================================================
# NAL header + trailing bits
# ==========================================================================


def _nal_header(p, v):
    p.nal_type = t = v & 31
    p.ref_idc = v >> 5
    if t == NAL_SPS:
        _sps_begin(p)
    elif t == NAL_PPS:
        _pps_begin(p)
    elif t == NAL_SEI:
        _sei_type(p, True)
    elif t == NAL_AUD:
        p._expect("primary_pic_type", u(3), _then_trailing, vindep=True)
    elif t in (NAL_EOSEQ, NAL_EOSTREAM):
        p.done = True
    elif t == NAL_FILLER:
        p._expect("ff_byte", u(8, ivset((0x80, 0xFF))), _filler)
    elif t in VCL:
        _slice_begin(p)
    else:  # not reachable: header legal set excludes other types
        p.kill()


def _trailing(p):
    n = 8 - (p.pos & 7)
    p._expect("rbsp_trailing_bits", u(n, iv(1 << (n - 1), 1 << (n - 1))), _end)


def _then_trailing(p, v):
    _trailing(p)


def _end(p, v):
    p.done = True


def _align_zeros(p):
    """After an explicitly read rbsp_stop_one_bit."""
    n = (8 - (p.pos & 7)) & 7
    if n == 0:
        p.done = True
    else:
        p._expect("rbsp_alignment_zero_bit", u(n, iv(0, 0)), _end)


def _filler(p, v):
    if v == 0x80:
        p.done = True
    else:
        p._expect("ff_byte", u(8, ivset((0x80, 0xFF))), _filler)


# ==========================================================================
# SPS (profile_idc 66 only)
# ==========================================================================

def _sps_begin(p):
    p.rec = {}
    p._expect("profile_idc", u(8, iv(66, 66)), _sps_after_profile)


def _sps_after_profile(p, v):
    p.put("profile_idc", v)
    p.put("csf_left", 6)
    p._expect("constraint_set_flag", u(1), _sps_csf, vindep=True)


def _sps_csf(p, v):
    left = p.rec["csf_left"] - 1
    p.put("csf_left", left)
    if left:
        p._expect("constraint_set_flag", u(1), _sps_csf, vindep=True)
    else:
        p._expect("reserved_zero_2bits", u(2, iv(0, 0)), _sps_reserved)


def _sps_reserved(p, v):
    p._expect("level_idc", u(8, ivset(LEVELS)), _sps_level)


def _sps_level(p, v):
    p.put("level_idc", v)
    p._expect("seq_parameter_set_id", ue_range(0, 31), _sps_id)


def _sps_id(p, v):
    p.put("id", v)
    p._expect("log2_max_frame_num_minus4", ue_range(0, 12), _sps_lmfn)


def _sps_lmfn(p, v):
    p.put("log2_max_frame_num", v + 4)
    p._expect("pic_order_cnt_type", ue_range(0, 2), _sps_poc_type)


def _sps_poc_type(p, v):
    p.put("poc_type", v)
    if v == 0:
        p._expect("log2_max_pic_order_cnt_lsb_minus4", ue_range(0, 12), _sps_poc0)
    elif v == 1:
        p._expect("delta_pic_order_always_zero_flag", u(1), _sps_poc1_flag)
    else:
        _sps_refs(p)


def _sps_poc0(p, v):
    p.put("log2_max_poc_lsb", v + 4)
    _sps_refs(p)


def _sps_poc1_flag(p, v):
    p.put("delta_poc_always_zero", v)
    p._expect("offset_for_non_ref_pic", se(), _sps_poc1_nonref, vindep=True)


def _sps_poc1_nonref(p, v):
    p._expect("offset_for_top_to_bottom_field", se(), _sps_poc1_t2b, vindep=True)


def _sps_poc1_t2b(p, v):
    p._expect("num_ref_frames_in_pic_order_cnt_cycle", ue_range(0, 255), _sps_poc1_n)


def _sps_poc1_n(p, v):
    p.put("poc_cycle_left", v)
    _sps_poc1_next(p)


def _sps_poc1_next(p):
    left = p.rec["poc_cycle_left"]
    if left:
        p.put("poc_cycle_left", left - 1)
        p._expect("offset_for_ref_frame", se(), _sps_poc1_off, vindep=True)
    else:
        _sps_refs(p)


def _sps_poc1_off(p, v):
    _sps_poc1_next(p)


def _sps_refs(p):
    p._expect("max_num_ref_frames", ue_range(0, 16), _sps_max_refs)


def _sps_max_refs(p, v):
    p.put("max_num_ref_frames", v)
    p._expect("gaps_in_frame_num_value_allowed_flag", u(1), _sps_gaps)


def _sps_gaps(p, v):
    p.put("gaps", v)
    max_fs, max_dpb_mbs = LEVELS[p.rec["level_idc"]]
    side = int((8 * max_fs) ** 0.5)
    nref = p.rec["max_num_ref_frames"]
    # A.3.1: width bounded by sqrt(8*MaxFS); DPB must hold max_num_ref_frames
    w_max = min(side, max_fs, max_dpb_mbs // nref if nref else side)
    p._expect("pic_width_in_mbs_minus1", ue_range(0, w_max - 1), _sps_width)


def _sps_width(p, v):
    w = v + 1
    p.put("W", w)
    max_fs, max_dpb_mbs = LEVELS[p.rec["level_idc"]]
    side = int((8 * max_fs) ** 0.5)
    nref = p.rec["max_num_ref_frames"]
    h_max = min(side, max_fs // w, max_dpb_mbs // (w * nref) if nref else side)
    p._expect("pic_height_in_map_units_minus1", ue_range(0, h_max - 1), _sps_height)


def _sps_height(p, v):
    h = v + 1
    p.put("H", h)
    max_dpb = min(LEVELS[p.rec["level_idc"]][1] // (p.rec["W"] * h), 16)
    p.put("max_dpb", max_dpb)
    p._expect("frame_mbs_only_flag", u(1, iv(1, 1)), _sps_fmo)


def _sps_fmo(p, v):
    p._expect("direct_8x8_inference_flag", u(1), _sps_d8)


def _sps_d8(p, v):
    p._expect("frame_cropping_flag", u(1), _sps_crop)


def _sps_crop(p, v):
    if v:
        p._expect("frame_crop_left_offset", ue_range(0, 8 * p.rec["W"] - 1), _sps_crop_l)
    else:
        p._expect("vui_parameters_present_flag", u(1), _sps_vui)


def _sps_crop_l(p, v):
    p._expect("frame_crop_right_offset", ue_range(0, 8 * p.rec["W"] - 1 - v), _sps_crop_r)


def _sps_crop_r(p, v):
    p._expect("frame_crop_top_offset", ue_range(0, 8 * p.rec["H"] - 1), _sps_crop_t)


def _sps_crop_t(p, v):
    p._expect("frame_crop_bottom_offset", ue_range(0, 8 * p.rec["H"] - 1 - v), _sps_crop_b)


def _sps_crop_b(p, v):
    p._expect("vui_parameters_present_flag", u(1), _sps_vui)


# ---- VUI (Annex E) --------------------------------------------------------


def _sps_vui(p, v):
    if v:
        p._expect("aspect_ratio_info_present_flag", u(1), _vui_ar)
    else:
        _trailing(p)


def _vui_ar(p, v):
    if v:
        p._expect("aspect_ratio_idc", u(8), _vui_ar_idc)
    else:
        _vui_overscan(p)


def _vui_ar_idc(p, v):
    if v == 255:
        p._expect("sar_width", u(16), _vui_sar_w, vindep=True)
    else:
        _vui_overscan(p)


def _vui_sar_w(p, v):
    p._expect("sar_height", u(16), lambda p, v: _vui_overscan(p), vindep=True)


def _vui_overscan(p):
    p._expect("overscan_info_present_flag", u(1), _vui_overscan_f)


def _vui_overscan_f(p, v):
    if v:
        p._expect("overscan_appropriate_flag", u(1), lambda p, v: _vui_vst(p), vindep=True)
    else:
        _vui_vst(p)


def _vui_vst(p):
    p._expect("video_signal_type_present_flag", u(1), _vui_vst_f)


def _vui_vst_f(p, v):
    if v:
        p._expect("video_format", u(3), _vui_vf, vindep=True)
    else:
        _vui_chroma_loc(p)


def _vui_vf(p, v):
    p._expect("video_full_range_flag", u(1), _vui_fr, vindep=True)


def _vui_fr(p, v):
    p._expect("colour_description_present_flag", u(1), _vui_cd)


def _vui_cd(p, v):
    if v:
        p._expect("colour_primaries", u(8), _vui_cp, vindep=True)
    else:
        _vui_chroma_loc(p)


def _vui_cp(p, v):
    p._expect("transfer_characteristics", u(8), _vui_tc, vindep=True)


def _vui_tc(p, v):
    p._expect("matrix_coefficients", u(8), lambda p, v: _vui_chroma_loc(p), vindep=True)


def _vui_chroma_loc(p):
    p._expect("chroma_loc_info_present_flag", u(1), _vui_cl)


def _vui_cl(p, v):
    if v:
        p._expect("chroma_sample_loc_type_top_field", ue_range(0, 5), _vui_cl_t, vindep=True)
    else:
        _vui_timing(p)


def _vui_cl_t(p, v):
    p._expect(
        "chroma_sample_loc_type_bottom_field", ue_range(0, 5),
        lambda p, v: _vui_timing(p), vindep=True,
    )


def _vui_timing(p):
    p._expect("timing_info_present_flag", u(1), _vui_ti)


def _vui_ti(p, v):
    if v:
        p._expect("num_units_in_tick", u(32, iv(1, (1 << 32) - 1)), _vui_nuit, vindep=True)
    else:
        _vui_nal_hrd(p)


def _vui_nuit(p, v):
    p._expect("time_scale", u(32, iv(1, (1 << 32) - 1)), _vui_ts, vindep=True)


def _vui_ts(p, v):
    p._expect("fixed_frame_rate_flag", u(1), lambda p, v: _vui_nal_hrd(p), vindep=True)


def _vui_nal_hrd(p):
    p.put("any_hrd", 0)
    p._expect("nal_hrd_parameters_present_flag", u(1), _vui_nal_hrd_f)


def _vui_nal_hrd_f(p, v):
    if v:
        p.put("any_hrd", 1)
        _hrd_begin(p, _vui_vcl_hrd)
    else:
        _vui_vcl_hrd(p)


def _vui_vcl_hrd(p):
    p._expect("vcl_hrd_parameters_present_flag", u(1), _vui_vcl_hrd_f)


def _vui_vcl_hrd_f(p, v):
    if v:
        p.put("any_hrd", 1)
        _hrd_begin(p, _vui_low_delay)
    else:
        _vui_low_delay(p)


def _vui_low_delay(p):
    if p.rec["any_hrd"]:
        p._expect("low_delay_hrd_flag", u(1), lambda p, v: _vui_pic_struct(p), vindep=True)
    else:
        _vui_pic_struct(p)


def _vui_pic_struct(p):
    p._expect("pic_struct_present_flag", u(1), _vui_ps, vindep=True)


def _vui_ps(p, v):
    p._expect("bitstream_restriction_flag", u(1), _vui_br)


def _vui_br(p, v):
    if v:
        p._expect("motion_vectors_over_pic_boundaries_flag", u(1), _vui_br1, vindep=True)
    else:
        _trailing(p)


def _vui_br1(p, v):
    p._expect("max_bytes_per_pic_denom", ue_range(0, 16), _vui_br2, vindep=True)


def _vui_br2(p, v):
    p._expect("max_bits_per_mb_denom", ue_range(0, 16), _vui_br3, vindep=True)


def _vui_br3(p, v):
    p._expect("log2_max_mv_length_horizontal", ue_range(0, 16), _vui_br4, vindep=True)


def _vui_br4(p, v):
    p._expect("log2_max_mv_length_vertical", ue_range(0, 16), _vui_br5, vindep=True)


def _vui_br5(p, v):
    p._expect("max_num_reorder_frames", ue_range(0, p.rec["max_dpb"]), _vui_br6)


def _vui_br6(p, v):
    lo = max(v, p.rec["max_num_ref_frames"])
    p._expect(
        "max_dec_frame_buffering", ue_range(lo, p.rec["max_dpb"]),
        _then_trailing, vindep=True,
    )


def _hrd_begin(p, after):
    p.put("hrd_after", after)
    p._expect("cpb_cnt_minus1", ue_range(0, 31), _hrd_cnt)


def _hrd_cnt(p, v):
    p.put("cpb_left", v + 1)
    p._expect("bit_rate_scale", u(4), _hrd_brs, vindep=True)


def _hrd_brs(p, v):
    p._expect("cpb_size_scale", u(4), lambda p, v: _hrd_next(p), vindep=True)


def _hrd_next(p):
    if p.rec["cpb_left"]:
        p.put("cpb_left", p.rec["cpb_left"] - 1)
        p._expect("bit_rate_value_minus1", ue(), _hrd_br, vindep=True)
    else:
        p._expect("initial_cpb_removal_delay_length_minus1", u(5), _hrd_d1, vindep=True)


def _hrd_br(p, v):
    p._expect("cpb_size_value_minus1", ue(), _hrd_cs, vindep=True)


def _hrd_cs(p, v):
    p._expect("cbr_flag", u(1), lambda p, v: _hrd_next(p), vindep=True)


def _hrd_d1(p, v):
    p._expect("cpb_removal_delay_length_minus1", u(5), _hrd_d2, vindep=True)


def _hrd_d2(p, v):
    p._expect("dpb_output_delay_length_minus1", u(5), _hrd_d3, vindep=True)


def _hrd_d3(p, v):
    p._expect("time_offset_length", u(5), lambda p, v: p.rec["hrd_after"](p), vindep=True)


# ==========================================================================
# PPS
# ==========================================================================


def _pps_begin(p):
    p.rec = {}
    p._expect("pic_parameter_set_id", ue_range(0, 255), _pps_id)


def _pps_id(p, v):
    p.put("id", v)
    p._expect("seq_parameter_set_id", ue_range(0, 31), _pps_sps)


def _pps_sps(p, v):
    p.put("sps_id", v)
    p._expect("entropy_coding_mode_flag", u(1, iv(0, 0)), _pps_ecm)


def _pps_ecm(p, v):
    p._expect("bottom_field_pic_order_in_frame_present_flag", u(1), _pps_bfpo)


def _pps_bfpo(p, v):
    p.put("bottom_field_poc", v)
    p._expect("num_slice_groups_minus1", ue_range(0, 0), _pps_nsg)


def _pps_nsg(p, v):
    p._expect("num_ref_idx_l0_default_active_minus1", ue_range(0, 31), _pps_ref0)


def _pps_ref0(p, v):
    p.put("num_ref_l0", v + 1)
    p._expect("num_ref_idx_l1_default_active_minus1", ue_range(0, 31), _pps_ref1)


def _pps_ref1(p, v):
    p._expect("weighted_pred_flag", u(1, iv(0, 0)), _pps_wp)


def _pps_wp(p, v):
    p._expect("weighted_bipred_idc", u(2, iv(0, 0)), _pps_wb)


def _pps_wb(p, v):
    p._expect("pic_init_qp_minus26", se(-26, 25), _pps_qp)


def _pps_qp(p, v):
    p.put("init_qp", 26 + v)
    p._expect("pic_init_qs_minus26", se(-26, 25), _pps_qs)


def _pps_qs(p, v):
    p._expect("chroma_qp_index_offset", se(-12, 12), _pps_cqo)


def _pps_cqo(p, v):
    p._expect("deblocking_filter_control_present_flag", u(1), _pps_dfc)


def _pps_dfc(p, v):
    p.put("deblocking", v)
    p._expect("constrained_intra_pred_flag", u(1), _pps_cip)


def _pps_cip(p, v):
    p.put("constrained_intra", v)
    p._expect("redundant_pic_cnt_present_flag", u(1), _pps_rpc)


def _pps_rpc(p, v):
    p.put("redundant_pic_cnt", v)
    # more_rbsp_data(): a 1 here is rbsp_stop_one_bit, a 0 is the (Baseline:
    # zero) transform_8x8_mode_flag that opens the optional extension.
    p._expect("rbsp_stop_one_bit|transform_8x8_mode_flag", u(1), _pps_more)


def _pps_more(p, v):
    if v:
        _align_zeros(p)
    else:
        p._expect("pic_scaling_matrix_present_flag", u(1, iv(0, 0)), _pps_psm)


def _pps_psm(p, v):
    p._expect("second_chroma_qp_index_offset", se(-12, 12), _then_trailing)


# ==========================================================================
# SEI (payloads are byte-framed and opaque)
# ==========================================================================


def _sei_type_legal(ctx, first):
    types = ctx.profile.sei_types
    if types is None:
        vals = set(range(256))
    else:
        vals = {t for t in types if t < 255}
        if any(t >= 255 for t in types):
            vals.add(0xFF)
    if not first:
        vals.add(0x80)  # read as rbsp_trailing_bits by _sei_type_next
    return ivset(vals)


def _sei_type(p, first):
    p.put("sei_acc", 0)
    p._expect(
        "sei_payload_type" if first else "sei_payload_type|rbsp_trailing_bits",
        u(8, _sei_type_legal(p.ctx, first)),
        _sei_type_first if first else _sei_type_next,
    )


def _sei_type_next(p, v):
    if v == 0x80:
        p.done = True
    else:
        _sei_type_first(p, v)


def _sei_type_first(p, v):
    if v == 0xFF:
        p.put("sei_acc", p.rec["sei_acc"] + 255)
        p._expect("sei_payload_type_ff", u(8), _sei_type_first)
        return
    p.put("sei_acc", 0)
    p._expect("sei_payload_size", u(8), _sei_size)


def _sei_size(p, v):
    if v == 0xFF:
        p.put("sei_acc", p.rec["sei_acc"] + 255)
        p._expect("sei_payload_size", u(8), _sei_size)
        return
    size = p.rec["sei_acc"] + v
    p.put("sei_left", size)
    _sei_payload_next(p)


def _sei_payload_next(p):
    left = p.rec["sei_left"]
    if left:
        p.put("sei_left", left - 1)
        p._expect("sei_payload_byte", u(8), _sei_payload_byte, vindep=True)
    else:
        _sei_type(p, False)


def _sei_payload_byte(p, v):
    _sei_payload_next(p)


# ==========================================================================
# slice header (7.3.3)
# ==========================================================================


def _pinned(p, key, legal):
    """Slices after the first of an open picture must repeat picture fields."""
    pic = p.ctx.pic
    if pic is not None and key in pic:
        x = pic[key]
        return iv(x, x) if C._hits(legal, x, x) else ()
    return legal


def _slice_begin(p):
    p.rec = {"nal_type": p.nal_type, "ref_idc": p.ref_idc, "marking": ()}
    pic = p.ctx.pic
    first = pic["next_mb"] if pic is not None else 0
    p._expect("first_mb_in_slice", ue_range(first, first), _sh_first_mb)


def _sh_first_mb(p, v):
    p.put("first_mb", v)
    pic = p.ctx.pic
    if p.nal_type == NAL_IDR:
        bases = (2,)
    else:
        bases = (0, 2)
    vals = []
    for t in bases:
        if pic is not None and pic["fixed_type"] is not None:
            if t == pic["fixed_type"]:
                vals += [t, t + 5]
            continue
        vals.append(t)
        if pic is None or pic["types"] <= {t}:
            vals.append(t + 5)
    p._expect("slice_type", ue(ivset(vals)), _sh_slice_type)


def _sh_slice_type(p, v):
    p.put("slice_type", v)
    ids = p.ctx.slice_pps_ids(p.nal_type)
    legal = _pinned(p, "pps_id", ivset(ids))
    p._expect("pic_parameter_set_id", ue(legal), _sh_pps)


def _sh_pps(p, v):
    ctx = p.ctx
    pps = ctx.pps[v]
    sps = ctx.sps[pps["sps_id"]]
    p.put("pps_id", v)
    p.put("sps_id", pps["sps_id"])
    p.P = pps
    p.S = sps
    p.W, p.H = sps["W"], sps["H"]
    size = p.W * p.H
    first = p.rec["first_mb"]
    if first >= size:
        p.kill()
        return
    if ctx.profile.slice_layout == "frame":
        end = size
    else:
        end = first + 1
    p.put("pic_size", size)
    p.put("end_mb", end)
    n = sps["log2_max_frame_num"]
    max_fn = 1 << n
    if p.nal_type == NAL_IDR:
        legal = iv(0, 0)
    elif sps["gaps"]:
        legal = iv(0, max_fn - 1)
    else:
        f = (ctx.prev_ref_frame_num + 1) % max_fn
        legal = iv(f, f)
    p._expect("frame_num", u(n, _pinned(p, "frame_num", legal)), _sh_frame_num)


def _sh_frame_num(p, v):
    p.put("frame_num", v)
    if p.nal_type == NAL_IDR:
        legal = iv(0, 65535)
        if p.ctx.pic is None and p.ctx.prev_idr_pic_id is not None:
            legal = C.iv_minus_point(legal, p.ctx.prev_idr_pic_id)
        p._expect("idr_pic_id", ue(_pinned(p, "idr_pic_id", legal)), _sh_idr_id)
    else:
        _sh_poc(p)


def _sh_idr_id(p, v):
    p.put("idr_pic_id", v)
    _sh_poc(p)


def _sh_poc(p):
    S = p.S
    if S["poc_type"] == 0:
        n = S["log2_max_poc_lsb"]
        legal = _pinned(p, "poc_lsb", iv(0, (1 << n) - 1))
        p._expect("pic_order_cnt_lsb", u(n, legal), _sh_poc_lsb)
    elif S["poc_type"] == 1 and not S["delta_poc_always_zero"]:
        p._expect("delta_pic_order_cnt[0]", _pinned_se(p, "poc0"), _sh_poc0)
    else:
        _sh_redundant(p)


def _pinned_se(p, key):
    pic = p.ctx.pic
    if pic is not None and key in pic:
        return se(pic[key], pic[key])
    return se()


def _sh_poc_lsb(p, v):
    p.put("poc_lsb", v)
    if p.P["bottom_field_poc"]:
        p._expect("delta_pic_order_cnt_bottom", _pinned_se(p, "poc_bottom"), _sh_poc_bottom)
    else:
        _sh_redundant(p)


def _sh_poc_bottom(p, v):
    p.put("poc_bottom", v)
    _sh_redundant(p)


def _sh_poc0(p, v):
    p.put("poc0", v)
    if p.P["bottom_field_poc"]:
        p._expect("delta_pic_order_cnt[1]", _pinned_se(p, "poc1"), _sh_poc1)
    else:
        _sh_redundant(p)


def _sh_poc1(p, v):
    p.put("poc1", v)
    _sh_redundant(p)


def _sh_redundant(p):
    if p.P["redundant_pic_cnt"]:
        p._expect("redundant_pic_cnt", ue_range(0, 127), lambda p, v: _sh_refs(p))
    else:
        _sh_refs(p)


def _sh_refs(p):
    if p.rec["slice_type"] % 5 == 0:
        p._expect("num_ref_idx_active_override_flag", u(1), _sh_override)
    else:
        p.nref = 0
        _sh_marking(p)


def _sh_override(p, v):
    if v:
        p._expect("num_ref_idx_l0_active_minus1", ue_range(0, 15), _sh_nref)
    else:
        _sh_nref(p, p.P["num_ref_l0"] - 1)


def _sh_nref(p, v):
    p.nref = v + 1
    # ref_idx_l0 must not select a "no reference picture" list entry (8.2.4.2)
    p.avail = p.ctx.refs
    p.put("mods", 0)
    p._expect("ref_pic_list_modification_flag_l0", u(1), _sh_rplm_flag)


def _sh_rplm_flag(p, v):
    if v:
        _sh_rplm_idc(p)
    else:
        _sh_marking(p)


def _sh_rplm_idc(p):
    legal = iv(3, 3) if p.rec["mods"] >= p.nref else iv(0, 3)
    p._expect("modification_of_pic_nums_idc", ue(legal), _sh_rplm_op)


def _sh_rplm_op(p, v):
    if v == 3:
        _sh_marking(p)
        return
    p.put("mods", p.rec["mods"] + 1)
    p.avail = None  # modified lists may repeat pictures: no count bound
    if v in (0, 1):
        max_pic_num = 1 << p.S["log2_max_frame_num"]
        p._expect("abs_diff_pic_num_minus1", ue_range(0, max_pic_num - 1),
                  lambda p, v: _sh_rplm_idc(p), vindep=True)
    else:
        p._expect("long_term_pic_num", ue(), lambda p, v: _sh_rplm_idc(p), vindep=True)


def _mk(p, name, code_legal, handler):
    """Reference-marking field; pinned to the picture's first slice."""
    pic = p.ctx.pic
    idx = len(p.rec["marking"])
    if pic is not None:
        pinned = pic["marking"]
        if idx >= len(pinned):
            p.kill()
            return
        x = pinned[idx]
        code_legal = iv(x, x) if C._hits(code_legal, x, x) else ()
    code = ue(code_legal) if name.startswith("ue:") else u(1, code_legal)
    p._expect(name[3:], code, handler)


def _mark(p, v):
    p.put("marking", p.rec["marking"] + (v,))


def _sh_marking(p):
    if p.ref_idc == 0:
        _sh_qp(p)
    elif p.nal_type == NAL_IDR:
        _mk(p, "u1:no_output_of_prior_pics_flag", iv(0, 1), _sh_noout)
    else:
        _mk(p, "u1:adaptive_ref_pic_marking_mode_flag", iv(0, 1), _sh_adaptive)


def _sh_noout(p, v):
    _mark(p, v)
    _mk(p, "u1:long_term_reference_flag", iv(0, 1), _sh_ltr)


def _sh_ltr(p, v):
    _mark(p, v)
    _sh_qp(p)


def _sh_adaptive(p, v):
    _mark(p, v)
    if v:
        _mk(p, "ue:memory_management_control_operation", iv(0, 6), _sh_mmco)
    else:
        _sh_qp(p)


def _sh_mmco(p, v):
    _mark(p, v)
    if v == 0:
        _sh_qp(p)
        return
    if v == 5:
        p.put("mmco5", 1)
    args = {1: ("difference_of_pic_nums_minus1",), 2: ("long_term_pic_num",),
            3: ("difference_of_pic_nums_minus1", "long_term_frame_idx"),
            4: ("max_long_term_frame_idx_plus1",), 5: (), 6: ("long_term_frame_idx",)}[v]
    p.q = args
    _sh_mmco_arg(p)


def _sh_mmco_arg(p):
    if not p.q:
        _mk(p, "ue:memory_management_control_operation", iv(0, 6), _sh_mmco)
        return
    name, p.q = p.q[0], p.q[1:]
    _mk(p, "ue:" + name, C.ANY_CN, _sh_mmco_argv)


def _sh_mmco_argv(p, v):
    _mark(p, v)
    _sh_mmco_arg(p)


def _sh_qp(p):
    qp0 = p.P["init_qp"]
    p._expect("slice_qp_delta", se(-qp0, 51 - qp0), _sh_qp_done, vindep=True)


def _sh_qp_done(p, v):
    if p.P["deblocking"]:
        p._expect("disable_deblocking_filter_idc", ue_range(0, 2), _sh_dbk)
    else:
        _sd_begin(p)


def _sh_dbk(p, v):
    if v != 1:
        p._expect("slice_alpha_c0_offset_div2", se(-6, 6), _sh_alpha, vindep=True)
    else:
        _sd_begin(p)


def _sh_alpha(p, v):
    p._expect("slice_beta_offset_div2", se(-6, 6), lambda p, v: _sd_begin(p), vindep=True)


# ==========================================================================
# slice data + macroblock layer (CAVLC)
# ==========================================================================


def _sd_begin(p):
    rec = p.rec
    p.first = rec["first_mb"]
    p.end = rec["end_mb"]
    p.cur = p.first
    p.is_p = rec["slice_type"] % 5 == 0
    p.cip = p.P["constrained_intra"]
    p.g = None
    p.ov = {}
    _mb_next(p)


def _mb_next(p):
    if p.is_p:
        p._expect("mb_skip_run", ue_range(0, p.end - p.cur), _on_skip)
    else:
        _mb_begin(p)


def _on_skip(p, run):
    p.cur += run
    if run and p.cur == p.end:
        _trailing(p)
    else:
        _mb_begin(p)


def _mb_status(p, addr) -> int:
    """-2: unavailable; -1: skipped (P_Skip); else ST_* of a coded MB."""
    if addr < p.first or addr >= p.cur:
        return -2
    return p.gget(4, addr)


def _mb_avail_intra(p, addr) -> bool:
    st = _mb_status(p, addr)
    if st == -2:
        return False
    if p.cip and st in (-1, ST_INTER):
        return False
    return True


def _mb_neighbours(p):
    """Intra sample availability bits (left, top, top-left) at MB level."""
    W = p.W
    x, y = p.cur % W, p.cur // W
    a = 0
    if x > 0 and _mb_avail_intra(p, p.cur - 1):
        a |= 1
    if y > 0 and _mb_avail_intra(p, p.cur - W):
        a |= 2
    if x > 0 and y > 0 and _mb_avail_intra(p, p.cur - W - 1):
        a |= 4
    return a


def _mb_type_class(p, v):
    i = v - 5 if p.is_p else v
    if i < 0:
        return v  # inter types differ in partition syntax
    if 1 <= i <= 24:
        k = i - 1
        return ("I16", (k // 4) % 3, 15 if k >= 12 else 0)
    return ("I", i)


def _mb_begin(p):
    p.mbx, p.mby = p.cur % p.W, p.cur // p.W
    avail = _mb_neighbours(p)
    p.nb = avail
    code = _MBT_CODES.get((p.is_p, avail))
    if code is None:
        intra = [0, 25] + [1 + k for k in range(24) if (_NEED_I16[k % 4] & ~avail) == 0]
        vals = list(range(5)) + [5 + i for i in intra] if p.is_p else intra
        code = _MBT_CODES[(p.is_p, avail)] = ue(ivset(vals))
    p._expect("mb_type", code, _on_mb_type, vclass=_mb_type_class)


_MBT_CODES = {}


def _on_mb_type(p, v):
    p.mbt_is_ref0 = p.is_p and v == 4  # P_8x8ref0 infers ref_idx_l0 = 0
    if p.is_p and v < 5:
        p.gset(4, p.cur, ST_INTER)
        p.inter = True
        p.i16 = False
        if v >= 3:
            p.sub = ()
            p._expect("sub_mb_type", ue_range(0, 3), _on_sub_mb_type)
            return
        parts = P_NUM_PARTS[v]
        q = ("r",) * parts if p.nref > 1 else ()
        p.q = q + ("m",) * (2 * parts)
        _ip_next(p)
        return
    i = v - 5 if p.is_p else v
    p.inter = False
    if i == 0:
        p.gset(4, p.cur, ST_I4)
        p.i16 = False
        p.blk = 0
        _i4_begin(p)
    elif i == 25:
        p.gset(4, p.cur, ST_PCM)
        n = (8 - (p.pos & 7)) & 7
        p.pcm_left = 384
        if n:
            p._expect("pcm_alignment_zero_bit", u(n, iv(0, 0)), lambda p, v: _pcm_next(p))
        else:
            _pcm_next(p)
    else:
        p.gset(4, p.cur, ST_I16)
        p.i16 = True
        k = i - 1
        p.cbpc = (k // 4) % 3
        p.cbpl = 15 if k >= 12 else 0
        _chroma_mode(p)


def _pcm_next(p):
    if p.pcm_left:
        p.pcm_left -= 1
        p._expect("pcm_sample", u(8), lambda p, v: _pcm_next(p), vindep=True)
    else:
        _mb_finish(p)


def _on_sub_mb_type(p, v):
    p.sub = p.sub + (v,)
    if len(p.sub) < 4:
        p._expect("sub_mb_type", ue_range(0, 3), _on_sub_mb_type)
        return
    q = ("r",) * 4 if (not p.mbt_is_ref0 and p.nref > 1) else ()
    for st in p.sub:
        q += ("m",) * (2 * SUB_NUM_PARTS[st])
    p.q = q
    _ip_next(p)


def _ip_next(p):
    if not p.q:
        p._expect("coded_block_pattern", me(False), _on_cbp)
        return
    op, p.q = p.q[0], p.q[1:]
    if op == "r":
        vmax = None if p.avail is None else max(1, p.avail) - 1
        p._expect("ref_idx_l0", te(p.nref - 1, vmax), _ip_step, vindep=True)
    else:
        p._expect("mvd_l0", se(MVD_LO, MVD_HI), _ip_step, vindep=True)


def _ip_step(p, v):
    _ip_next(p)


# ---- intra prediction ------------------------------------------------------


def _l4(p, blk):
    """Picture 4x4-luma coordinates of block ``blk`` of the current MB."""
    q, r = blk >> 2, blk & 3
    return p.mbx * 4 + (q & 1) * 2 + (r & 1), p.mby * 4 + (q >> 1) * 2 + (r >> 1)


def _cell_mb(p, x, y):
    """MB address owning 4x4 cell (x, y), or -1 outside the picture."""
    if x < 0 or y < 0 or x >= 4 * p.W:
        return -1
    return (y >> 2) * p.W + (x >> 2)


def _i4_neighbour(p, x, y):
    """(sample available, predicted-mode contribution) for 4x4 cell (x, y).

    Mode contribution: -1 forces DC (dcPredModePredictedFlag), else the mode.
    """
    addr = _cell_mb(p, x, y)
    if addr == p.cur:
        return True, p.gget(3, y * 4 * p.W + x)
    if addr < 0:
        return False, -1
    st = _mb_status(p, addr)
    if st == -2:
        return False, -1
    if st in (-1, ST_INTER):
        return (False, -1) if p.cip else (True, 2)
    if st == ST_I4:
        return True, p.gget(3, y * 4 * p.W + x)
    return True, 2


def _i4_begin(p):
    x, y = _l4(p, p.blk)
    la, ma = _i4_neighbour(p, x - 1, y)
    ta, mb = _i4_neighbour(p, x, y - 1)
    tla, _ = _i4_neighbour(p, x - 1, y - 1)
    avail = (1 if la else 0) | (2 if ta else 0) | (4 if tla else 0)
    pred = 2 if ma < 0 or mb < 0 else min(ma, mb)
    p.i4_pred = pred
    p.i4_avail = avail
    p._expect("prev_intra4x4_pred_mode_flag", _i4_codes(pred, avail)[0], _on_prev_flag)


_I4_CODES = {}


def _i4_codes(pred, avail):
    """(prev_intra4x4_pred_mode_flag code, rem_intra4x4_pred_mode code)."""
    key = (pred, avail)
    c = _I4_CODES.get(key)
    if c is None:
        ok = [(_NEED_I4[m] & ~avail) == 0 for m in range(9)]
        flags = ([1] if ok[pred] else []) + (
            [0] if any(ok[m] for m in range(9) if m != pred) else []
        )
        rems = [r for r in range(8) if ok[r if r < pred else r + 1]]
        c = _I4_CODES[key] = (u(1, ivset(flags)), u(3, ivset(rems)))
    return c


def _on_prev_flag(p, v):
    if v:
        _i4_set(p, p.i4_pred)
        return
    code = _i4_codes(p.i4_pred, p.i4_avail)[1]
    p._expect("rem_intra4x4_pred_mode", code, _on_rem)


def _on_rem(p, v):
    _i4_set(p, v if v < p.i4_pred else v + 1)


def _i4_set(p, mode):
    x, y = _l4(p, p.blk)
    p.gset(3, y * 4 * p.W + x, mode)
    p.blk += 1
    if p.blk < 16:
        _i4_begin(p)
    else:
        _chroma_mode(p)


def _chroma_mode(p):
    code = _CHROMA_CODES.get(p.nb)
    if code is None:
        legal = ivset(m for m in range(4) if (_NEED_CHROMA[m] & ~p.nb) == 0)
        code = _CHROMA_CODES[p.nb] = ue(legal)
    p._expect("intra_chroma_pred_mode", code, _on_chroma_mode, vindep=True)


_CHROMA_CODES = {}


def _on_chroma_mode(p, v):
    if p.i16:
        _residual_gate(p)
    else:
        p._expect("coded_block_pattern", me(True), _on_cbp)


def _on_cbp(p, cbp):
    p.cbpl, p.cbpc = cbp & 15, cbp >> 4
    _residual_gate(p)


def _residual_gate(p):
    if p.cbpl or p.cbpc or p.i16:
        p._expect("mb_qp_delta", se(-26, 25), _res_begin, vindep=True)
    else:
        _mb_finish(p)


def _mb_finish(p):
    p.cur += 1
    if p.cur == p.end:
        _trailing(p)
    else:
        _mb_next(p)


# ---- residual (7.3.5.3 / 9.2) -----------------------------------------------


def _nc_luma(p, blk):
    x, y = _l4(p, blk)
    return _nc(p, _nnz_luma(p, x - 1, y), _nnz_luma(p, x, y - 1))


def _nnz_luma(p, x, y):
    addr = _cell_mb(p, x, y)
    if addr < 0:
        return -1
    if addr != p.cur:
        st = _mb_status(p, addr)
        if st == -2:
            return -1
        if st == -1:
            return 0
        if st == ST_PCM:
            return 16
    return p.gget(0, y * 4 * p.W + x)


def _nnz_chroma(p, comp, x, y):
    if x < 0 or y < 0 or x >= 2 * p.W:
        return -1
    addr = (y >> 1) * p.W + (x >> 1)
    if addr != p.cur:
        st = _mb_status(p, addr)
        if st == -2:
            return -1
        if st == -1:
            return 0
        if st == ST_PCM:
            return 16
    return p.gget(1 + comp, y * 2 * p.W + x)


def _nc(p, na, nb):
    if na >= 0 and nb >= 0:
        return (na + nb + 1) >> 1
    if na >= 0:
        return na
    if nb >= 0:
        return nb
    return 0


def _res_begin(p, v=None):
    p.rphase = 0 if p.i16 else 1  # 0 luma DC, 1 luma, 2 chroma DC, 3 chroma AC
    p.rblk = 0
    _res_next(p)


def _res_next(p):
    while True:
        ph = p.rphase
        if ph == 0:
            p.rphase = 1
            p.rblk = 0
            _rb_start(p, _nc_luma(p, 0), 16, None)
            return
        if ph == 1:
            while p.rblk < 16:
                blk = p.rblk
                p.rblk += 1
                if p.cbpl & (1 << (blk >> 2)):
                    _rb_start(p, _nc_luma(p, blk), 15 if p.i16 else 16, (0, blk))
                    return
            p.rphase = 2
            p.rblk = 0
            continue
        if ph == 2:
            if p.cbpc and p.rblk < 2:
                p.rblk += 1
                _rb_start(p, -1, 4, None)
                return
            p.rphase = 3
            p.rblk = 0
            continue
        if ph == 3:
            if p.cbpc == 2 and p.rblk < 8:
                comp, blk = divmod(p.rblk, 4)
                p.rblk += 1
                x = p.mbx * 2 + (blk & 1)
                y = p.mby * 2 + (blk >> 1)
                nc = _nc(p, _nnz_chroma(p, comp, x - 1, y), _nnz_chroma(p, comp, x, y - 1))
                _rb_start(p, nc, 15, (1 + comp, y * 2 * p.W + x))
                return
            _mb_finish(p)
            return


def _rb_start(p, nc, maxc, target):
    p.rtarget = target
    p.rmax = maxc
    p._expect("coeff_token", _coeff_token_code(T.coeff_token_label(nc), maxc), _rb_coeff_token)


_CT_ALLOWED = {}


def _coeff_token_code(label, maxc):
    key = (label, maxc)
    code = _CT_ALLOWED.get(key)
    if code is None:
        allowed = frozenset(v for v in T.code_map(label).values() if v[0] <= maxc)
        code = _CT_ALLOWED[key] = vlc(label, allowed)
    return code


def _rb_write(p, tc):
    t = p.rtarget
    if t is None or tc == 0:
        return
    kind, idx = t
    if kind == 0:
        x, y = _l4(p, idx)
        idx = y * 4 * p.W + x
    p.gset(kind, idx, tc)


def _rb_coeff_token(p, v):
    tc, t1 = v
    p.tc, p.t1 = tc, t1
    if tc == 0:
        _res_next(p)
    elif t1:
        p._expect("trailing_ones_sign_flag", u(t1), _rb_levels, vindep=True)
    else:
        _rb_levels(p)


def _rb_levels(p, v=None):
    tc, t1 = p.tc, p.t1
    p.sl = 1 if (tc > 10 and t1 < 3) else 0
    p.li = 0
    if tc == t1:
        _rb_total_zeros(p)
    else:
        p._expect("level_prefix", unary(LEVEL_PREFIX_MAX), _rb_prefix, vclass=_prefix_class)


def _suffix_size(lp, sl):
    if lp == 14 and sl == 0:
        return 4
    if lp >= 15:
        return lp - 3
    return sl


def _next_sl(p, lp, suffix):
    sl = p.sl
    code = (min(15, lp) << sl) + suffix
    if lp >= 15 and sl == 0:
        code += 15
    if lp >= 16:
        code += (1 << (lp - 3)) - 4096
    if p.li == 0 and p.t1 < 3:
        code += 2
    level = (code + 2) >> 1 if code % 2 == 0 else (-code - 1) >> 1
    if sl == 0:
        sl = 1
    if abs(level) > (3 << (sl - 1)) and sl < 6:
        sl += 1
    return sl


def _prefix_class(p, lp):
    if _suffix_size(lp, p.sl) == 0:
        return ("sl", _next_sl(p, lp, 0))
    return ("lp", lp)


def _rb_prefix(p, lp):
    p.lp = lp
    ss = _suffix_size(lp, p.sl)
    if ss:
        p._expect("level_suffix", u(ss), _rb_suffix, vclass=_suffix_class)
    else:
        _rb_suffix(p, 0)


def _suffix_class(p, s):
    return _next_sl(p, p.lp, s)


def _rb_suffix(p, s):
    p.sl = _next_sl(p, p.lp, s)
    p.li += 1
    if p.li < p.tc - p.t1:
        p._expect("level_prefix", unary(LEVEL_PREFIX_MAX), _rb_prefix, vclass=_prefix_class)
    else:
        _rb_total_zeros(p)


def _rb_total_zeros(p):
    tc, maxc = p.tc, p.rmax
    if tc < maxc:
        code = _TZ_CODES.get((tc, maxc))
        if code is None:
            label = f"total_zeros_cdc_{tc}" if maxc == 4 else f"total_zeros_4x4_{tc}"
            code = _TZ_CODES[(tc, maxc)] = vlc(label, range(maxc - tc + 1))
        p._expect("total_zeros", code, _rb_tz)
    else:
        _rb_tz(p, 0)


def _rb_tz(p, tz):
    p.zl = tz
    p.ri = 0
    _rb_run_next(p)


def _rb_run_next(p):
    if p.ri < p.tc - 1 and p.zl > 0:
        code = _RUN_CODES.get(p.zl)
        if code is None:
            code = _RUN_CODES[p.zl] = vlc(f"run_before_{min(p.zl, 7)}", range(p.zl + 1))
        p._expect("run_before", code, _rb_run)
    else:
        _rb_write(p, p.tc)
        _res_next(p)


_TZ_CODES = {}
_RUN_CODES = {}


def _rb_run(p, r):
    p.zl -= r
    p.ri += 1
    _rb_run_next(p)

