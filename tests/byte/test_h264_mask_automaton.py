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
