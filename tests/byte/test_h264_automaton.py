"""Differential + soundness tests for the incremental CAVLC automaton.

Loads modules by file path (stdlib-only, no torch) like test_h264_syntax.py. The
existing recursive-descent parser (h264_syntax) is the oracle:

  Differential -- drive the automaton bit-by-bit over each fixture VCL NAL's
     slice-data RBSP; it must decode the same number of macroblocks and reach a
     clean rbsp_trailing_bits DONE with no INVALID.
  Soundness    -- at every RBSP byte boundary, compile_byte_mask must mark the byte
     actually present as legal. The mask may never reject a real decodable byte.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BYTE_DIR = Path(__file__).resolve().parents[2] / "litgpt" / "byte"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_FILES = ["baseline_qcif.h264", "baseline_qcif_lowqp.h264"]


def _load_module(name: str):
    if str(_BYTE_DIR) not in sys.path:
        sys.path.insert(0, str(_BYTE_DIR))
    spec = importlib.util.spec_from_file_location(name, _BYTE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load_module("h264_syntax")
A = _load_module("h264_automaton")


def _bit(rbsp: bytes, p: int) -> int:
    return (rbsp[p >> 3] >> (7 - (p & 7))) & 1


def _slice_context(data: bytes, p, sps_map, pps_map):
    """Return (rbsp, slice_data_start_bit, head, sps) for a VCL NAL, reusing the
    real slice-header parser (the automaton only covers slice data)."""
    rbsp, byte_map, _ = H.unescape_rbsp(data, p.nal.payload_start + 1, p.nal.payload_end)
    reader = H.BitReader(rbsp)
    record = H._Recorder(byte_map, len(rbsp))
    save = reader.pos
    reader.read_ue()  # first_mb_in_slice
    reader.read_ue()  # slice_type
    pps = pps_map[reader.read_ue()]
    reader.pos = save
    sps = sps_map[pps.sps_id]
    head = H.parse_slice_header(reader, record, p.nal, sps, pps)
    return rbsp, reader.pos, head, sps


def _make_automaton(head, sps, start_bit):
    pic = sps.pic_width_in_mbs * sps.pic_height_in_mbs
    return A.MbAutomaton(
        pic_width_in_mbs=sps.pic_width_in_mbs,
        pic_height_in_mbs=sps.pic_height_in_mbs,
        slice_type=head.slice_type,
        num_ref_idx_l0_active=head.num_ref_idx_l0_active,
        first_mb_in_slice=head.first_mb_in_slice,
        slice_data_start_bit=start_bit,
        max_mbs=pic,
    )


def _vcl_parses(data):
    parsed = H.parse_stream(data, parse_slice_data=True)
    vcl = [
        p for p in parsed.nals
        if p.nal.nal_type in (1, 5) and p.status == H.ParseStatus.OK
    ]
    return parsed, vcl


def _assert_automaton_matches_parser(name):
    path = _FIXTURES / name
    if not path.exists():
        return
    data = path.read_bytes()
    parsed, vcl = _vcl_parses(data)
    assert vcl, "no VCL slices parsed"
    for p in vcl:
        rbsp, start, head, sps = _slice_context(data, p, parsed.sps, parsed.pps)
        pic = sps.pic_width_in_mbs * sps.pic_height_in_mbs
        auto = _make_automaton(head, sps, start)
        nbits = len(rbsp) * 8
        done_at = None
        for bp in range(start, nbits):
            st = auto.consume_bit(_bit(rbsp, bp))
            assert st != A.INVALID, (
                f"{name} nal {p.nal.index}: INVALID at bit {bp} "
                f"(mbs_done={auto.mbs_done}/{pic})"
            )
            if st == A.DONE:
                done_at = bp + 1
                break
        assert auto.mbs_done == p.mb_count == pic, (
            f"{name} nal {p.nal.index}: mbs_done={auto.mbs_done} "
            f"parser={p.mb_count} pic={pic}"
        )
        assert done_at is not None, f"{name} nal {p.nal.index}: never reached DONE"


def _assert_mask_never_rejects_real_byte(name):
    path = _FIXTURES / name
    if not path.exists():
        return
    data = path.read_bytes()
    parsed, vcl = _vcl_parses(data)
    assert vcl, "no VCL slices parsed"
    checked = 0
    for p in vcl:
        rbsp, start, head, sps = _slice_context(data, p, parsed.sps, parsed.pps)
        auto = _make_automaton(head, sps, start)
        nbits = len(rbsp) * 8
        bp = start
        while bp < nbits:
            if auto.stage != "done" and auto.pos % 8 == 0 and bp + 8 <= nbits:
                mask = A.compile_byte_mask(auto)
                real = rbsp[auto.pos >> 3]
                assert mask[real], (
                    f"{name} nal {p.nal.index}: mask rejects real byte "
                    f"{real:#04x} at bit {bp}"
                )
                checked += 1
            st = auto.consume_bit(_bit(rbsp, bp))
            assert st != A.INVALID
            bp += 1
            if st == A.DONE:
                break
    assert checked > 0, "no byte-aligned states were checked"


def test_automaton_matches_parser():
    for name in _FIXTURE_FILES:
        _assert_automaton_matches_parser(name)


def test_mask_never_rejects_real_byte():
    for name in _FIXTURE_FILES:
        _assert_mask_never_rejects_real_byte(name)


if __name__ == "__main__":  # stdlib runner (no pytest/torch)
    for _name in _FIXTURE_FILES:
        _path = _FIXTURES / _name
        if not _path.exists():
            print(f"skip {_name} (missing)")
            continue
        _assert_automaton_matches_parser(_name)
        _assert_mask_never_rejects_real_byte(_name)
        print(f"ok {_name}")
    print("all passed")
