"""Integration tests for the automaton-backed emitted-byte mask."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_BYTE_DIR = Path(__file__).resolve().parents[2] / "litgpt" / "byte"
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "baseline_qcif.h264"


def _load_stack():
    # Avoid importing litgpt.__init__ (and its optional training dependencies).
    old = {name: sys.modules.get(name) for name in ("litgpt", "litgpt.byte")}
    sys.modules["litgpt"] = types.ModuleType("litgpt")
    sys.modules["litgpt.byte"] = types.ModuleType("litgpt.byte")

    def load(short):
        name = f"litgpt.byte.{short}"
        spec = importlib.util.spec_from_file_location(name, _BYTE_DIR / f"{short}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(sys.modules["litgpt.byte"], short, module)
        spec.loader.exec_module(module)
        return module

    load("h264_cavlc_tables")
    syntax = load("h264_syntax")
    automaton = load("h264_automaton")
    mask = load("h264_mask")
    return syntax, automaton, mask, old


def test_wrapper_never_rejects_fixture_prefix_and_uses_strict_masks():
    if not _FIXTURE.exists():
        return
    syntax, _, mask_module, _ = _load_stack()
    data = _FIXTURE.read_bytes()
    vcl = next(n for n in syntax.iter_nals(data) if n.nal_type in syntax.VCL_NAL_TYPES)
    stop = min(len(data), vcl.payload_start + 1 + 500)
    state = mask_module.MaskState(slice_max_mbs=99)
    strict_masks = 0
    for offset, byte in enumerate(data[:stop]):
        allowed = mask_module.get_valid_byte_mask(state)
        assert allowed[byte], f"rejected fixture byte {byte:#04x} at {offset}"
        strict_masks += sum(allowed) < 256
        mask_module.advance(state, byte)
    assert strict_masks > 0


def test_wrapper_memoized_compiler_matches_legacy_masks():
    if not _FIXTURE.exists():
        return
    syntax, automaton, mask_module, _ = _load_stack()
    data = _FIXTURE.read_bytes()
    vcl = next(n for n in syntax.iter_nals(data) if n.nal_type in syntax.VCL_NAL_TYPES)
    stop = min(len(data), vcl.payload_start + 1 + 256)
    legacy_state = mask_module.MaskState(slice_max_mbs=99)
    memoized_state = mask_module.MaskState(slice_max_mbs=99)
    compiler = automaton.MemoizedByteMaskCompiler(max_cache_entries=500_000)

    for offset, byte in enumerate(data[:stop]):
        compiler.record_corpus_byte()
        legacy = mask_module.get_valid_byte_mask(legacy_state)
        memoized = mask_module.get_valid_byte_mask(
            memoized_state, byte_mask_compiler=compiler
        )
        assert memoized == legacy, f"memoized wrapper mismatch at byte {offset}"
        mask_module.advance(legacy_state, byte)
        mask_module.advance(memoized_state, byte)

    assert compiler.statistics()["root_requests"] > 0


def test_frame_layout_never_rejects_full_picture_fixture():
    """The dynamic layout must derive the fixture's 99-MB extent from its SPS."""
    if not _FIXTURE.exists():
        return
    syntax, _, mask_module, _ = _load_stack()
    data = _FIXTURE.read_bytes()
    first_vcl = next(
        nal for nal in syntax.iter_nals(data) if nal.nal_type in syntax.VCL_NAL_TYPES
    )
    state = mask_module.MaskState(
        slice_max_mbs=mask_module.slice_max_mbs_for_layout("frame")
    )
    strict_masks = 0
    # Exhaustively check candidate-byte membership through the first VCL header,
    # where the dynamic SPS-derived extent is established. Replaying the remainder
    # with advance() still exercises all 99 macroblocks without multiplying this
    # regression test by 256 candidate branches at every residual byte.
    exhaustive_end = first_vcl.payload_start + 16
    for offset, byte in enumerate(data[: first_vcl.payload_end]):
        if offset < exhaustive_end:
            allowed = mask_module.get_valid_byte_mask(state)
            assert allowed[byte], (
                f"rejected frame-layout GT byte {byte:#04x} at {offset}"
            )
            strict_masks += sum(allowed) < 256
        mask_module.advance(state, byte)
        assert not state.automaton_unknown, (
            f"frame-layout automaton rejected GT byte {byte:#04x} at {offset}: "
            f"{state.failure_reason}"
        )
    assert strict_masks > 0
    assert state.automaton.stage == "done"
    assert state.automaton.mbs_done == state.automaton.max_mbs == 99


def test_frame_layout_resolves_sps_extent_and_rejects_partial_picture_start():
    syntax, automaton, _, _ = _load_stack()
    sps, pps = _phase_parameter_sets(syntax)

    full = automaton.VclAutomaton(
        nal_type=syntax.NAL_SLICE_NONIDR,
        nal_ref_idc=2,
        sps_map={0: sps},
        pps_map={0: pps},
        constraints={},
        max_mbs=None,
    )
    status = automaton.MORE
    for bit in "111":  # ue(0) first_mb, ue(0) P slice, ue(0) PPS
        status = full.consume_bit(int(bit))
    assert status != automaton.INVALID
    assert full.max_mbs == 16 * 9

    partial = automaton.VclAutomaton(
        nal_type=syntax.NAL_SLICE_NONIDR,
        nal_ref_idc=2,
        sps_map={0: sps},
        pps_map={0: pps},
        constraints={},
        max_mbs=None,
    )
    status = automaton.MORE
    for bit in "01011":  # ue(1) first_mb, ue(0) P slice, ue(0) PPS
        status = partial.consume_bit(int(bit))
    assert status == automaton.INVALID
    assert partial.invalid_reason == "frame_slice_first_mb_not_zero:1"


def test_mask_lookup_does_not_call_recursive_parser():
    if not _FIXTURE.exists():
        return
    syntax, _, mask_module, _ = _load_stack()
    data = _FIXTURE.read_bytes()
    vcl = next(n for n in syntax.iter_nals(data) if n.nal_type in syntax.VCL_NAL_TYPES)
    stop = min(len(data), vcl.payload_start + 1 + 300)
    state = mask_module.MaskState(slice_max_mbs=99)
    for byte in data[:stop]:
        mask_module.advance(state, byte)
    assert state.automaton is not None

    original = syntax.parse_nal
    syntax.parse_nal = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("mask lookup reparsed the NAL")
    )
    try:
        mask_module.get_valid_byte_mask(state)
    finally:
        syntax.parse_nal = original


def test_unknown_automaton_fails_closed_by_default():
    _, _, mask_module, _ = _load_stack()
    state = mask_module.MaskState(
        cur_nal_bytes=bytearray((0x61, 0x00, 0x00)),
        cur_is_vcl=True,
        automaton_unknown=True,
    )
    allowed = mask_module.get_valid_byte_mask(state)
    assert not any(allowed)


def test_unknown_automaton_can_be_explicitly_fail_open():
    _, _, mask_module, _ = _load_stack()
    state = mask_module.MaskState(
        cur_nal_bytes=bytearray((0x61, 0x00, 0x00)),
        cur_is_vcl=True,
        automaton_unknown=True,
        fail_closed=False,
    )
    allowed = mask_module.get_valid_byte_mask(state)
    assert allowed[0x00]
    assert allowed[0x01]
    assert not allowed[0x02]


def test_completed_header_with_unknown_pps_fails_closed():
    _, _, mask_module, _ = _load_stack()
    state = mask_module.MaskState()
    mask_module.advance(state, 0x61)  # non-IDR VCL NAL header
    mask_module.advance(state, 0xE0)  # ue(0) first_mb, ue(0) P, ue(0) PPS
    assert state.automaton_unknown
    assert state.failure_reason == "unknown_pps:0"
    assert not any(mask_module.get_valid_byte_mask(state))


def _phase_parameter_sets(syntax):
    sps = syntax.SPS(
        sps_id=0,
        profile_idc=66,
        level_idc=10,
        log2_max_frame_num=4,
        pic_order_cnt_type=2,
        log2_max_pic_order_cnt_lsb=0,
        delta_pic_order_always_zero_flag=0,
        frame_mbs_only_flag=1,
        pic_width_in_mbs=16,
        pic_height_in_mbs=9,
    )
    pps = syntax.PPS(
        pps_id=0,
        sps_id=0,
        entropy_coding_mode_flag=0,
        bottom_field_pic_order_in_frame_present_flag=0,
        num_ref_idx_l0_default_active=3,
        num_ref_idx_l1_default_active=1,
        weighted_pred_flag=0,
        weighted_bipred_idc=0,
        pic_init_qp=26,
        deblocking_filter_control_present_flag=1,
        constrained_intra_pred_flag=0,
        redundant_pic_cnt_present_flag=0,
    )
    return sps, pps


def test_picture_state_constrains_nal_header_and_slice_identity():
    syntax, _, mask_module, _ = _load_stack()
    sps, _ = _phase_parameter_sets(syntax)
    picture = mask_module.PictureState()
    picture.observe(
        {
            "first_mb_in_slice": 0,
            "slice_type": syntax.SLICE_TYPE_P,
            "pic_parameter_set_id": 0,
            "frame_num": 4,
        },
        nal_type=syntax.NAL_SLICE_NONIDR,
        nal_ref_idc=2,
        sps=sps,
        max_mbs=1,
    )
    state = mask_module.MaskState(picture=picture, expect_nal_header=True)
    allowed = mask_module.get_valid_byte_mask(state)
    assert [i for i, ok in enumerate(allowed) if ok] == [0x41]

    constraints = picture.constraints(1, 2, {0: sps}, {})
    assert constraints["first_mb_in_slice"] == 1
    assert constraints["frame_num"] == 4
    assert constraints["pic_parameter_set_id"] == 0
    assert constraints["slice_type"] == syntax.SLICE_TYPE_P


def test_syntax_only_nal_header_policy_accepts_repeated_parameter_sets():
    """Corpus replay must not mistake legal SPS/PPS insertion for a bad slice."""
    _, _, mask_module, _ = _load_stack()
    picture = mask_module.PictureState(active=True, picture_complete=True)
    state = mask_module.MaskState(
        picture=picture,
        expect_nal_header=True,
        nal_header_policy=mask_module.NAL_HEADER_POLICY_SYNTAX_ONLY,
    )

    allowed = mask_module.get_valid_byte_mask(state)

    assert allowed[0x67]  # SPS
    assert allowed[0x68]  # PPS
    assert allowed[0x06]  # SEI
    assert allowed[0x65]  # IDR slice
    assert sum(allowed) == 128
    assert not allowed[0xE7]  # forbidden_zero_bit is set


def test_generation_without_prior_picture_starts_with_idr():
    _, _, mask_module, _ = _load_stack()
    state = mask_module.MaskState(
        expect_nal_header=True,
        generation_started=True,
    )
    allowed = mask_module.get_valid_byte_mask(state)
    assert [i for i, ok in enumerate(allowed) if ok] == [0x65]


def test_slice_header_mask_rejects_wrong_first_mb_before_commit():
    syntax, automaton, _, _ = _load_stack()
    sps, pps = _phase_parameter_sets(syntax)
    auto = automaton.VclAutomaton(
        nal_type=syntax.NAL_SLICE_NONIDR,
        nal_ref_idc=2,
        sps_map={0: sps},
        pps_map={0: pps},
        constraints={
            "first_mb_in_slice": 1,
            "frame_num": 4,
            "pic_parameter_set_id": 0,
            "slice_type": syntax.SLICE_TYPE_P,
            "available_reference_pictures": 3,
        },
        max_mbs=1,
    )
    allowed = automaton.compile_byte_mask(auto)
    # ue(0) begins with one, while the required ue(1) begins with 010.  Reject
    # every byte whose first bit commits the wrong macroblock address.
    assert not any(allowed[0x80:])


def test_slice_header_mask_prunes_impossible_partial_first_mb_prefixes():
    syntax, automaton, _, _ = _load_stack()
    sps, pps = _phase_parameter_sets(syntax)

    def allowed_for(first_mb):
        auto = automaton.VclAutomaton(
            nal_type=syntax.NAL_SLICE_NONIDR,
            nal_ref_idc=2,
            sps_map={0: sps},
            pps_map={0: pps},
            constraints={
                "first_mb_in_slice": first_mb,
                "frame_num": 4,
                "pic_parameter_set_id": 0,
                "slice_type": syntax.SLICE_TYPE_P,
                "available_reference_pictures": 3,
            },
            max_mbs=1,
        )
        return automaton.compile_byte_mask(auto)

    # These bytes caused the previous mask to get boxed in one byte later.  Each
    # byte is already incompatible with the required ue(v) codeword prefix.
    assert not allowed_for(7)[0x0A]
    assert not allowed_for(140)[0x03]
    assert not allowed_for(125)[0x01]


def test_macroblock_mask_prunes_impossible_partial_cbp_prefix():
    syntax, automaton, _, _ = _load_stack()
    auto = automaton.MbAutomaton(
        pic_width_in_mbs=16,
        pic_height_in_mbs=9,
        slice_type=syntax.SLICE_TYPE_P,
        num_ref_idx_l0_active=3,
        first_mb_in_slice=0,
        slice_data_start_bit=0,
        max_mbs=1,
    )
    auto._start("ue", "cbp")
    allowed = automaton.compile_byte_mask(auto)
    # cbp is bounded to 0..47, whose longest ue(v) codeword has five leading
    # zeroes.  Eight zero bits cannot be completed into a valid cbp.
    assert not allowed[0x00]


def test_picture_state_advances_frame_after_last_macroblock():
    syntax, _, mask_module, _ = _load_stack()
    sps, pps = _phase_parameter_sets(syntax)
    picture = mask_module.PictureState()
    common = {
        "slice_type": syntax.SLICE_TYPE_P,
        "pic_parameter_set_id": 0,
        "frame_num": 4,
    }
    picture.observe(
        {**common, "first_mb_in_slice": 0},
        nal_type=1,
        nal_ref_idc=2,
        sps=sps,
        max_mbs=1,
    )
    for first_mb in range(1, 144):
        picture.observe(
            {**common, "first_mb_in_slice": first_mb},
            nal_type=1,
            nal_ref_idc=2,
            sps=sps,
            max_mbs=1,
        )
    constraints = picture.constraints(1, 2, {0: sps}, {0: pps})
    assert picture.picture_complete
    assert constraints["first_mb_in_slice"] == 0
    assert constraints["frame_num"] == 5


def test_annexb_mid_nal_rules():
    _, _, mask_module, _ = _load_stack()
    state = mask_module.MaskState(cur_nal_bytes=bytearray((0, 0)))
    assert mask_module.annexb_forbidden(state, at_nal_boundary=False) == [0, 1, 2]
    assert mask_module.annexb_forbidden(state, at_nal_boundary=True) == [2]


def test_emulation_prevention_translates_rbsp_mask_to_ebsp():
    _, _, mask_module, _ = _load_stack()
    state = mask_module.MaskState(cur_nal_bytes=bytearray((0x61, 0x00, 0x00)))
    rbsp_mask = [False] * 256
    rbsp_mask[0x01] = True
    mask_module._apply_annexb_mask(state, rbsp_mask, at_nal_boundary=False)
    assert rbsp_mask[0x03]  # emit the zero-bit EPB first
    assert not rbsp_mask[0x01]  # protected byte cannot be emitted directly

    state.cur_nal_bytes.append(0x03)
    protected_mask = [False] * 256
    protected_mask[0x01] = True
    mask_module._apply_annexb_mask(state, protected_mask, at_nal_boundary=False)
    assert protected_mask[0x01]
    assert not any(protected_mask[4:])


def test_pending_emulation_prevention_byte_is_not_fed_to_rbsp():
    syntax, _, mask_module, _ = _load_stack()
    state = mask_module.MaskState(
        cur_nal_bytes=bytearray((0x61, 0x00, 0x00, 0x03)), cur_is_vcl=True
    )
    seen = []
    original = syntax.unescape_rbsp

    def capture(data, payload_start, payload_end):
        seen.append(payload_end)
        return b"", [], []

    syntax.unescape_rbsp = capture
    try:
        mask_module._sync_automaton(state)
    finally:
        syntax.unescape_rbsp = original
    assert seen == [len(state.cur_nal_bytes) - 1]


def test_completed_vcl_nal_only_allows_annexb_boundary_bytes():
    _, automaton, mask_module, _ = _load_stack()
    state = mask_module.MaskState(
        cur_nal_bytes=bytearray((0x61, 0x80)), cur_is_vcl=True
    )
    auto = automaton.MbAutomaton(
        pic_width_in_mbs=1,
        pic_height_in_mbs=1,
        slice_type=0,
        num_ref_idx_l0_active=1,
        first_mb_in_slice=0,
        slice_data_start_bit=0,
        max_mbs=1,
    )
    auto.stage = "done"
    state.automaton = auto

    allowed = mask_module.get_valid_byte_mask(state)
    assert [i for i, ok in enumerate(allowed) if ok] == [0x00]
    state.cur_nal_bytes.extend((0x00, 0x00))
    allowed = mask_module.get_valid_byte_mask(state)
    assert [i for i, ok in enumerate(allowed) if ok] == [0x00, 0x01]


def test_can_append_bytes_is_non_mutating_and_checks_completion():
    _, automaton, mask_module, _ = _load_stack()

    def state(stage):
        auto = automaton.MbAutomaton(
            pic_width_in_mbs=1,
            pic_height_in_mbs=1,
            slice_type=0,
            num_ref_idx_l0_active=1,
            first_mb_in_slice=0,
            slice_data_start_bit=0,
            max_mbs=1,
        )
        auto.stage = stage
        return mask_module.MaskState(
            cur_nal_bytes=bytearray((0x61, 0x80)),
            cur_is_vcl=True,
            automaton=auto,
        )

    complete = state("done")
    assert mask_module.can_append_bytes(complete, b"", require_complete=True)
    assert mask_module.can_append_bytes(complete, b"\x00\x00\x01")
    assert not mask_module.can_append_bytes(
        complete, b"\x00\x00\x01", require_complete=True
    )  # dangling next-NAL start code
    assert not mask_module.can_append_bytes(complete, b"\x12")
    assert complete.mask_calls == 0
    assert complete.cur_nal_bytes == bytearray((0x61, 0x80))

    incomplete = state("slice")
    assert not mask_module.can_append_bytes(incomplete, b"", require_complete=True)


def test_default_wrapper_masks_header_but_legacy_residual_mode_does_not():
    syntax, automaton, mask_module, _ = _load_stack()

    def state(residual_only):
        auto = automaton.MbAutomaton(
            pic_width_in_mbs=16,
            pic_height_in_mbs=9,
            slice_type=syntax.SLICE_TYPE_P,
            num_ref_idx_l0_active=3,
            first_mb_in_slice=0,
            slice_data_start_bit=0,
            max_mbs=1,
        )
        return mask_module.MaskState(
            cur_nal_bytes=bytearray((0x61,)),
            cur_is_vcl=True,
            automaton=auto,
            residual_only=residual_only,
            fail_closed=not residual_only,
        )

    full = mask_module.get_valid_byte_mask(state(False))
    legacy = mask_module.get_valid_byte_mask(state(True))
    # The byte prefix 011 completes mb_skip_run=2, invalid for max_mbs=1.
    assert not any(full[0b01100000:0b10000000])
    assert all(legacy)
