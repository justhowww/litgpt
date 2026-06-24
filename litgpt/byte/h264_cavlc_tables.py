"""CAVLC variable-length-code tables for H.264 (Constrained Baseline, 4:2:0).

The bit codes are generated from the authoritative ``(length, bits)`` integer
arrays used by FFmpeg's ``libavcodec/h264_cavlc.c`` (Tables 9-5, 9-7/9-8, 9-9a,
9-10 of the H.264/AVC spec) -- so no codeword is hand-transcribed. A
module-level validator asserts every generated table is prefix-free with no
duplicate codes, turning any slip into an import-time error instead of a silent
parser desync.

Index convention for the coeff_token arrays: ``idx = total_coeff*4 +
trailing_ones``; a ``length`` of 0 marks an invalid (T1>TC) combination.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Raw FFmpeg (length, bits) arrays
# ---------------------------------------------------------------------------

# coeff_token, 4 tables: nC in [0,2), [2,4), [4,8), and >=8 (FLC). 4*17 entries.
_COEFF_TOKEN_LEN = [
    [1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8, 13, 13, 11, 9, 13, 13, 13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15, 14, 16, 15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16],
    [2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9, 9, 6, 11, 11, 11, 7, 12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13, 13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14],
    [4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6, 6, 4, 8, 7, 7, 5, 8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    [6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
]
_COEFF_TOKEN_BITS = [
    [1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11, 14, 5, 4, 8, 10, 13, 4, 15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1, 9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8],
    [3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15, 6, 5, 4, 11, 14, 13, 4, 15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4],
    [15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9, 14, 13, 9, 8, 10, 9, 8, 15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8, 10, 13, 8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2],
    [3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
]

# chroma DC coeff_token (nC == -1), 4*5 entries, idx = total_coeff*4 + trailing_ones.
_CHROMA_DC_COEFF_TOKEN_LEN = [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7]
_CHROMA_DC_COEFF_TOKEN_BITS = [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0]

# total_zeros, 4x4 (Tables 9-7/9-8). Row r (0-based) is tzVlcIndex = r+1; the j-th
# entry is total_zeros = j.
_TOTAL_ZEROS_LEN = [
    [1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6],
    [4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6],
    [5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5],
    [4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5],
    [6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6],
    [6, 5, 3, 3, 3, 2, 3, 4, 3, 6],
    [6, 4, 5, 3, 2, 2, 3, 3, 6],
    [6, 6, 4, 2, 2, 3, 2, 5],
    [5, 5, 3, 2, 2, 2, 4],
    [4, 4, 3, 3, 1, 3],
    [4, 4, 2, 1, 3],
    [3, 3, 1, 2],
    [2, 2, 1],
    [1, 1],
]
_TOTAL_ZEROS_BITS = [
    [1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
    [7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0],
    [5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0],
    [3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0],
    [5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0],
    [1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0],
    [1, 1, 5, 4, 3, 3, 2, 1, 1, 0],
    [1, 1, 1, 3, 3, 2, 2, 1, 0],
    [1, 0, 1, 3, 2, 1, 1, 1],
    [1, 0, 1, 3, 2, 1, 1],
    [0, 1, 1, 2, 1, 3],
    [0, 1, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 1],
    [0, 1],
]

# total_zeros, chroma DC 2x2 (Table 9-9a). Row r is tzVlcIndex = r+1.
_CHROMA_DC_TOTAL_ZEROS_LEN = [[1, 2, 3, 3], [1, 2, 2], [1, 1]]
_CHROMA_DC_TOTAL_ZEROS_BITS = [[1, 1, 1, 0], [1, 1, 0], [1, 0]]

# run_before (Table 9-10). Row r is zerosLeft = r+1 (row 6 covers zerosLeft >= 7).
_RUN_LEN = [
    [1, 1],
    [1, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 3, 3],
    [2, 2, 3, 3, 3, 3],
    [2, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11],
]
_RUN_BITS = [
    [1, 0],
    [1, 1, 0],
    [3, 2, 1, 0],
    [3, 2, 1, 1, 0],
    [3, 2, 3, 2, 1, 0],
    [3, 0, 1, 3, 2, 5, 4],
    [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


# ---------------------------------------------------------------------------
# Build {value: bitstring} maps from the (len, bits) arrays
# ---------------------------------------------------------------------------


def _bits_to_code(value: int, length: int) -> str:
    code = format(value, "b").zfill(length)
    if len(code) != length:
        raise ValueError(f"bits {value} do not fit length {length}")
    return code


def _build_coeff_token(lens: list[int], bits: list[int]) -> dict[tuple[int, int], str]:
    table: dict[tuple[int, int], str] = {}
    for idx in range(len(lens)):
        length = lens[idx]
        if length == 0:
            continue
        tc, t1 = divmod(idx, 4)
        table[(tc, t1)] = _bits_to_code(bits[idx], length)
    return table


def _build_rows(lens: list[list[int]], bits: list[list[int]]) -> dict[int, dict[int, str]]:
    table: dict[int, dict[int, str]] = {}
    for r, (lrow, brow) in enumerate(zip(lens, bits), start=1):
        table[r] = {j: _bits_to_code(brow[j], lrow[j]) for j in range(len(lrow)) if lrow[j] > 0}
    return table


COEFF_TOKEN_BINS = [_build_coeff_token(_COEFF_TOKEN_LEN[i], _COEFF_TOKEN_BITS[i]) for i in range(4)]
COEFF_TOKEN_CHROMA_DC = _build_coeff_token(_CHROMA_DC_COEFF_TOKEN_LEN, _CHROMA_DC_COEFF_TOKEN_BITS)
TOTAL_ZEROS_4x4 = _build_rows(_TOTAL_ZEROS_LEN, _TOTAL_ZEROS_BITS)
TOTAL_ZEROS_CHROMA_DC = _build_rows(_CHROMA_DC_TOTAL_ZEROS_LEN, _CHROMA_DC_TOTAL_ZEROS_BITS)
RUN_BEFORE = _build_rows(_RUN_LEN, _RUN_BITS)


# ---------------------------------------------------------------------------
# Decoders + structural validation
# ---------------------------------------------------------------------------

_CODE_MAPS: dict[str, dict[str, object]] = {}


def _invert(table: dict, label: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for value, code in table.items():
        if code in out:
            raise ValueError(f"{label}: duplicate code {code!r}")
        out[code] = value
    return out


def _assert_prefix_free(codes: list[str], label: str) -> None:
    ordered = sorted(codes, key=len)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.startswith(a):
                raise ValueError(f"{label}: code {a!r} is a prefix of {b!r}")


def _register(label: str, table: dict) -> dict[str, object]:
    code_map = _invert(table, label)
    _assert_prefix_free(list(code_map.keys()), label)
    _CODE_MAPS[label] = code_map
    return code_map


def code_map(label: str) -> dict[str, object]:
    return _CODE_MAPS[label]


def decode_vlc(read_bit, cmap: dict[str, object], label: str = "", max_len: int = 32):
    """Read bits until a code in ``cmap`` matches; return (value, num_bits)."""
    bits = ""
    for _ in range(max_len):
        bits += "1" if read_bit() else "0"
        if bits in cmap:
            return cmap[bits], len(bits)
    raise ValueError(f"no VLC match ({label}) for {bits!r}")


# coded_block_pattern me(v) mapping, Table 9-4 (ChromaArrayType 1/2): codeNum -> cbp.
GOLOMB_TO_INTRA_CBP = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10,
    12, 19, 21, 26, 28, 35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25,
    32, 33, 34, 36, 40, 38, 41,
]
GOLOMB_TO_INTER_CBP = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37,
    42, 44, 33, 34, 36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23,
    27, 29, 30, 22, 25, 38, 41,
]


def coeff_token_label(nc: int) -> str:
    if nc < 0:
        return "coeff_token_cdc"
    if nc < 2:
        return "coeff_token_0"
    if nc < 4:
        return "coeff_token_1"
    if nc < 8:
        return "coeff_token_2"
    return "coeff_token_flc"


def _validate() -> None:
    for i in range(4):
        _register(f"coeff_token_{i}" if i < 3 else "coeff_token_flc", COEFF_TOKEN_BINS[i])
    _register("coeff_token_cdc", COEFF_TOKEN_CHROMA_DC)
    for idx, t in TOTAL_ZEROS_4x4.items():
        _register(f"total_zeros_4x4_{idx}", t)
        if set(t.keys()) != set(range(17 - idx)):
            raise ValueError(f"total_zeros_4x4[{idx}] coverage {sorted(t)} != 0..{16 - idx}")
    for idx, t in TOTAL_ZEROS_CHROMA_DC.items():
        _register(f"total_zeros_cdc_{idx}", t)
        if set(t.keys()) != set(range(5 - idx)):
            raise ValueError(f"total_zeros_cdc[{idx}] coverage {sorted(t)} != 0..{4 - idx}")
    for zl, t in RUN_BEFORE.items():
        _register(f"run_before_{zl}", t)


_validate()
