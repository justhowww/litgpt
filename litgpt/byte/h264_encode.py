# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""H.264 syntax *encoding* helpers for the free-run rescue test.

The parser in ``h264_syntax.py`` only decodes. The rescue experiment needs the
inverse: synthesize the bits of a chosen legal value (to splice a legal ``mb_type``
into a desynced stream), enumerate the legal value set for a field, and overwrite
bits in a byte stream. Stdlib-only (no torch), mirroring ``h264_syntax`` so tests
can load it by path.

Bit order matches ``BitReader`` (MSB-first within each byte): bit ``p`` of a stream
is ``(data[p >> 3] >> (7 - (p & 7))) & 1``.
"""

from __future__ import annotations

# Slice types (mirror h264_syntax constants; kept local to stay import-light).
SLICE_TYPE_P = 0
SLICE_TYPE_I = 2

# Legal mb_type ranges (see h264_syntax._parse_macroblock, lines ~1031-1052):
#   P slice: mb_type 0..4 are inter (P modes); 5..30 are intra (i_mb_type 0..25).
#   I slice: mb_type 0..25 (i_mb_type == mb_type). Anything larger desyncs.
_P_MB_TYPE_MAX = 30
_I_MB_TYPE_MAX = 25
_SUB_MB_TYPE = (0, 1, 2, 3)  # P sub_mb_type legal set (_SUB_MB_NUM_PARTS keys)


# -----------------------------------------------------------------------------
# Exp-Golomb encoders (inverse of BitReader.read_ue / read_se)
# -----------------------------------------------------------------------------


def encode_ue(value: int) -> list[int]:
    """Bits (MSB-first) of the ``ue(v)`` codeword for a non-negative ``value``.

    Inverse of ``BitReader.read_ue``: codeNum = value+1, written as its binary
    digits with (bit_length-1) leading zeros. E.g. 0->[1], 1->[0,1,0], 2->[0,1,1],
    5->[0,0,1,1,0].
    """
    if value < 0:
        raise ValueError(f"ue(v) requires value >= 0, got {value}")
    code_num = value + 1
    digits = [int(b) for b in bin(code_num)[2:]]  # MSB-first, length = bit_length
    leading_zeros = len(digits) - 1
    return [0] * leading_zeros + digits


def encode_se(value: int) -> list[int]:
    """Bits of the ``se(v)`` codeword (inverse of ``BitReader.read_se``).

    se maps to ue codeNum k: 0->0, +m->2m-1, -m->2m. So k = 2|v| - (v > 0).
    """
    if value == 0:
        k = 0
    elif value > 0:
        k = 2 * value - 1
    else:
        k = -2 * value
    return encode_ue(k)


def ue_length(value: int) -> int:
    """Bit length of ``encode_ue(value)`` without materializing it."""
    return 2 * ((value + 1).bit_length() - 1) + 1


# -----------------------------------------------------------------------------
# Legal-value enumeration (which values a field may legally take)
# -----------------------------------------------------------------------------


def legal_mb_type(slice_type: int) -> list[int]:
    """The legal ``mb_type`` values for a slice type (P inter+intra, or I)."""
    if slice_type == SLICE_TYPE_P:
        return list(range(_P_MB_TYPE_MAX + 1))  # 0..30
    return list(range(_I_MB_TYPE_MAX + 1))  # 0..25


def legal_sub_mb_type() -> list[int]:
    """Legal P ``sub_mb_type`` values."""
    return list(_SUB_MB_TYPE)


# -----------------------------------------------------------------------------
# Bit splicing (overwrite bits in a byte stream at an absolute bit offset)
# -----------------------------------------------------------------------------


def splice_bits(data: bytes, bit_pos: int, bits: list[int]) -> bytes:
    """Return ``data`` with ``bits`` written starting at absolute bit ``bit_pos``.

    MSB-first within each byte (matching ``BitReader``). Extends the buffer (zero-
    padded) if the write runs past the end. Does not touch bits outside the range.
    """
    if bit_pos < 0:
        raise ValueError(f"bit_pos must be >= 0, got {bit_pos}")
    end_bit = bit_pos + len(bits)
    out = bytearray(data)
    needed = (end_bit + 7) >> 3
    if needed > len(out):
        out.extend(b"\x00" * (needed - len(out)))
    for i, b in enumerate(bits):
        p = bit_pos + i
        byte_i = p >> 3
        shift = 7 - (p & 7)
        if b:
            out[byte_i] |= 1 << shift
        else:
            out[byte_i] &= ~(1 << shift) & 0xFF
    return bytes(out)


def legal_prob_mass(
    byte_probs, bit_offset: int, fixed_high: int, legal_values: list[int], emitted_byte: int | None = None
) -> dict:
    """P_legal for a sub-byte field starting at ``bit_offset`` within a boundary byte.

    ``byte_probs`` is the model's distribution over that byte (length-256 sequence).
    Only bytes whose top ``bit_offset`` bits equal ``fixed_high`` (the already-fixed
    prefix bits) are considered; among those, a byte is *legal* if its bits from
    ``bit_offset`` begin (a prefix of) some legal ``ue`` codeword. Returns a dict:
    ``{p_legal, best_value, best_prob, best_legal_rank, illegal_prob}``.

    Byte-level approximation: codewords longer than the byte tail are matched on the
    in-byte prefix only (the remainder falls in the next byte, unobserved here).
    ``best_legal_rank`` = 0-indexed rank of the most-probable legal byte among all 256.
    """
    tail_len = 8 - bit_offset
    prefixes = {v: tuple(encode_ue(v)[: min(ue_length(v), tail_len)]) for v in legal_values}

    per_value = {v: 0.0 for v in legal_values}
    legal_bytes: list[int] = []
    for b in range(256):
        if (b >> tail_len) != fixed_high:
            continue  # inconsistent with the fixed prefix bits
        tail = [(b >> (tail_len - 1 - i)) & 1 for i in range(tail_len)]
        best_v, best_k = None, -1
        for v, pref in prefixes.items():
            k = len(pref)
            if tuple(tail[:k]) == pref and k > best_k:
                best_v, best_k = v, k
        if best_v is not None:
            per_value[best_v] += byte_probs[b]
            legal_bytes.append(b)

    p_legal = sum(byte_probs[b] for b in legal_bytes)
    best_value = max(per_value, key=lambda v: per_value[v]) if per_value else None
    best_prob = per_value[best_value] if best_value is not None else 0.0
    order = sorted(range(256), key=lambda b: byte_probs[b], reverse=True)
    rank_of = {b: i for i, b in enumerate(order)}
    best_legal_rank = min((rank_of[b] for b in legal_bytes), default=None)
    illegal_prob = byte_probs[emitted_byte] if emitted_byte is not None else None
    return {
        "p_legal": p_legal,
        "best_value": best_value,
        "best_prob": best_prob,
        "best_legal_rank": best_legal_rank,
        "illegal_prob": illegal_prob,
    }


def int_to_bits(value: int, n: int) -> list[int]:
    """The ``n`` bits of ``value`` (MSB-first). Inverse of ``read_bits_int`` semantics."""
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def read_bits_int(data: bytes, bit_pos: int, n: int) -> int:
    """Read ``n`` bits (MSB-first) starting at absolute bit ``bit_pos`` as an int.

    Helper for tests / P_legal marginalization: the leading bits of a byte value.
    """
    v = 0
    for i in range(n):
        p = bit_pos + i
        bit = (data[p >> 3] >> (7 - (p & 7))) & 1
        v = (v << 1) | bit
    return v
