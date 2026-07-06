"""Tests for the H.264 encoding helpers used by the free-run rescue test.

Both `h264_encode` and `h264_syntax` are stdlib-only, so we load them by file
path (independent of the torch/lightning env the repo conftest imports).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BYTE_DIR = Path(__file__).resolve().parents[1] / "litgpt" / "byte"


def _load(name: str):
    if str(_BYTE_DIR) not in sys.path:
        sys.path.insert(0, str(_BYTE_DIR))
    spec = importlib.util.spec_from_file_location(name, _BYTE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("h264_encode")
H = _load("h264_syntax")


def _bits_to_bytes(bits: list[int]) -> bytes:
    padded = bits + [0] * ((-len(bits)) % 8)
    out = bytearray(len(padded) // 8)
    for i, b in enumerate(padded):
        if b:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def test_encode_ue_roundtrip():
    for v in range(0, 1001):
        bits = E.encode_ue(v)
        assert len(bits) == E.ue_length(v)
        reader = H.BitReader(_bits_to_bytes(bits))
        assert reader.read_ue() == v, v
        assert reader.pos == len(bits), (v, reader.pos, len(bits))


def test_encode_se_roundtrip():
    for v in range(-500, 501):
        bits = E.encode_se(v)
        reader = H.BitReader(_bits_to_bytes(bits))
        assert reader.read_se() == v, v
        assert reader.pos == len(bits), (v, reader.pos, len(bits))


def test_encode_ue_known_codewords():
    # Spec examples: 0->"1", 1->"010", 2->"011", 5->"00110".
    assert E.encode_ue(0) == [1]
    assert E.encode_ue(1) == [0, 1, 0]
    assert E.encode_ue(2) == [0, 1, 1]
    assert E.encode_ue(5) == [0, 0, 1, 1, 0]


def test_legal_mb_type_ranges():
    # P: 0..30 (0-4 inter, 5-30 intra); I: 0..25. Matches h264_syntax reject at
    # _parse_macroblock (mb_type>=31 for P, i_mb_type>25 for I -> _DesyncError).
    assert E.legal_mb_type(E.SLICE_TYPE_P) == list(range(31))
    assert E.legal_mb_type(E.SLICE_TYPE_I) == list(range(26))
    assert E.legal_sub_mb_type() == [0, 1, 2, 3]


def test_splice_bits_overwrite_and_readback():
    data = bytes([0xFF, 0x00, 0xAA])
    # Overwrite 3 bits starting at bit 6 with [0,1,0].
    out = E.splice_bits(data, 6, [0, 1, 0])
    assert E.read_bits_int(out, 6, 3) == 0b010
    # Bits before 6 and after 9 are untouched.
    assert E.read_bits_int(out, 0, 6) == 0b111111
    assert E.read_bits_int(out, 9, 7) == E.read_bits_int(data, 9, 7)


def test_splice_bits_extends_buffer():
    out = E.splice_bits(b"\x00", 12, [1, 1, 1, 1])  # writes into a new second byte
    assert len(out) == 2
    assert E.read_bits_int(out, 12, 4) == 0b1111


def test_legal_prob_mass():
    # Byte-aligned field (bit_offset=0), legal mb_type in {0,1,2}.
    #   value 0 -> "1"   -> bytes 128..255
    #   value 1 -> "010" -> bytes 64..95
    #   value 2 -> "011" -> bytes 96..127
    # bytes 0..63 begin "00..." (mb_type >= 3) -> illegal for this legal set.
    probs = [0.0] * 256
    probs[200] = 0.5  # top bit 1 -> value 0 (legal)
    probs[70] = 0.2   # 010xxxxx -> value 1 (legal)
    probs[10] = 0.3   # 000xxxxx -> illegal
    r = E.legal_prob_mass(probs, bit_offset=0, fixed_high=0, legal_values=[0, 1, 2], emitted_byte=10)
    assert abs(r["p_legal"] - 0.7) < 1e-9, r
    assert r["best_value"] == 0 and abs(r["best_prob"] - 0.5) < 1e-9, r
    assert r["best_legal_rank"] == 0, r  # byte 200 (0.5) is the top byte overall
    assert abs(r["illegal_prob"] - 0.3) < 1e-9, r


def test_legal_prob_mass_respects_fixed_high():
    # bit_offset=2, fixed_high=0b11 -> only bytes 0b11xxxxxx (192..255) considered.
    probs = [0.0] * 256
    probs[192] = 0.4  # 11 000000 -> tail 000000 begins "0..." not in legal {0}
    probs[240] = 0.6  # 11 110000 -> tail 110000 begins "1" -> value 0 legal
    probs[10] = 0.9   # inconsistent high bits -> ignored entirely
    r = E.legal_prob_mass(probs, bit_offset=2, fixed_high=0b11, legal_values=[0], emitted_byte=192)
    assert abs(r["p_legal"] - 0.6) < 1e-9, r  # only byte 240
    assert r["best_value"] == 0


def test_splice_legal_mb_type_reparses():
    # Take a real CAVLC fixture, find the first P/I slice mb_type span, splice a
    # different *legal* mb_type at its bit_start, and confirm the parser reads it.
    data = (Path(__file__).resolve().parent / "byte" / "fixtures" / "baseline_qcif.h264").read_bytes()
    spans = H.parse_stream(data, parse_slice_data=True).all_spans()
    mb_type_spans = [s for s in spans if s.name == "mb_type"]
    assert mb_type_spans, "fixture has no mb_type spans"
    span = mb_type_spans[0]
    # bit_start is an RBSP bit offset; byte_start is the raw offset. For a span with
    # no emulation-prevention before it, splice at the raw bit position.
    # Use a legal value different from the decoded one where possible.
    target = 0 if span.value != 0 else 1
    raw_bit = span.byte_start * 8 + (span.bit_start & 7)
    spliced = E.splice_bits(data, raw_bit, E.encode_ue(target))
    reparsed = H.parse_stream(spliced, parse_slice_data=True).all_spans()
    hit = [s for s in reparsed if s.name == "mb_type" and s.byte_start == span.byte_start]
    assert hit and hit[0].value == target, (target, hit[0].value if hit else None)
