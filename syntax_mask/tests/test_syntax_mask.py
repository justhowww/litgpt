"""Tests for the syntax_mask package (pytest, or ``python -m syntax_mask.tests``)."""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from syntax_mask import cavlc_tables as T
from syntax_mask import codes as C
from syntax_mask.grammar import Illegal, Profile
from syntax_mask.scan import _probe
from syntax_mask.stream import MaskStream, iter_masks

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "byte" / "fixtures"


# ---------------------------------------------------------------------------
# Step 1b: code DFAs against an independent encoder
# ---------------------------------------------------------------------------


def _ue_bits(k):
    s = f"{k + 1:b}"
    return "0" * (len(s) - 1) + s


def _decode(code, bits):
    """Feed a bit string through the DFA; return value or raise on INV/short."""
    cs = code.init
    for i, ch in enumerate(bits):
        st, x = code.bit(cs, int(ch))
        if st == C.INV:
            raise ValueError("invalid")
        if st == C.DONE:
            if i != len(bits) - 1:
                raise ValueError("finished early")
            return x
        cs = x
    raise ValueError("unfinished")


def _check_code(code, domain, encode, legal):
    for v in domain:
        bits = encode(v)
        if legal(v):
            assert _decode(code, bits) == v, (code.key, v)
        else:
            try:
                got = _decode(code, bits)
            except ValueError:
                continue
            raise AssertionError(f"{code.key}: illegal {v} decoded as {got}")


def test_code_dfas_match_encoder():
    for n, lo, hi in [(3, 2, 5), (5, 0, 31), (8, 0x80, 0x80), (4, 1, 15)]:
        _check_code(C.u(n, C.iv(lo, hi)), range(1 << n),
                    lambda v, n=n: f"{v:0{n}b}", lambda v: lo <= v <= hi)
    for lo, hi in [(0, 0), (0, 3), (3, 3), (2, 40), (0, 300)]:
        _check_code(C.ue_range(lo, hi), range(400), _ue_bits, lambda v: lo <= v <= hi)
    for lo, hi in [(-26, 25), (-6, 6), (-3, 10), (-51, 0), (4, 9)]:
        _check_code(C.se(lo, hi), range(-60, 61),
                    lambda v: _ue_bits(C.se_to_codenum(v)), lambda v: lo <= v <= hi)
    for intra in (True, False):
        table = T.GOLOMB_TO_INTRA_CBP if intra else T.GOLOMB_TO_INTER_CBP
        code = C.me(intra)
        for k, cbp in enumerate(table):
            assert _decode(code, _ue_bits(k)) == cbp
    assert _decode(C.te(1), "1") == 0 and _decode(C.te(1), "0") == 1
    assert _decode(C.te(1, 0), "1") == 0
    try:
        _decode(C.te(1, 0), "0")
        raise AssertionError("te(1, vmax=0) accepted 1")
    except ValueError:
        pass
    for label in ("coeff_token_0", "coeff_token_cdc", "total_zeros_4x4_3", "run_before_7"):
        values = list(set(T.code_map(label).values()))
        allowed = frozenset(random.Random(0).sample(values, max(1, len(values) // 2)))
        code = C.vlc(label, allowed)
        for bits, value in T.code_map(label).items():
            if value in allowed:
                assert _decode(code, bits) == value
            else:
                try:
                    _decode(code, bits)
                    raise AssertionError(f"{label}: {value} accepted")
                except ValueError:
                    pass
    code = C.unary(15)
    assert _decode(code, "0" * 15 + "1") == 15
    try:
        _decode(code, "0" * 16 + "1")
        raise AssertionError("level_prefix 16 accepted")
    except ValueError:
        pass


def test_step_tables_match_bitwise_walk():
    rng = random.Random(1)
    codes = [C.u(12, C.iv(100, 3000)), C.u(16), C.ue_range(5, 700), C.se(-40, 9),
             C.se(), C.vlc("coeff_token_1"), C.unary(15), C.me(False)]
    for code in codes:
        for _ in range(30):
            cs = code.init
            for _ in range(rng.randrange(0, 6)):  # random legal prefix
                opts = [(b, code.bit(cs, b)) for b in (0, 1)]
                opts = [(b, x) for b, (st, x) in opts if st == C.MORE]
                if not opts:
                    break
                cs = rng.choice(opts)[1]
            for r in range(1, 9):
                step = code.step(cs, r)
                partial, comps = 0, set()
                for chunk in range(1 << r):
                    s = cs
                    for i in range(r):
                        b = (chunk >> (r - 1 - i)) & 1
                        st, x = code.bit(s, b)
                        if st == C.INV:
                            break
                        if st == C.DONE:
                            v = x
                            comps.add((v, i + 1, chunk >> (r - 1 - i)))
                            break
                        s = x
                    else:
                        partial |= 1 << chunk
                got = set()
                for v, L, ch in step.comps:
                    if code.rel:
                        v = (cs[1] << L) | v
                    got.add((v, L, ch))
                assert step.partial == partial, (code.key, cs, r)
                assert got == comps, (code.key, cs, r)


# ---------------------------------------------------------------------------
# Steps 2 + 3 on real streams
# ---------------------------------------------------------------------------


def _fixture(name="baseline_qcif.h264"):
    path = FIXTURES / name
    return path.read_bytes() if path.exists() else None


def test_fixture_gt_accepted_and_ebsp_invariants():
    for name in ("baseline_qcif.h264", "baseline_qcif_lowqp.h264"):
        data = _fixture(name)
        if data is None:
            continue
        ms = MaskStream()
        for i, byte in enumerate(data):
            m = ms.mask()
            assert (m >> byte) & 1, f"{name}@{i} {ms.where()}"
            if ms.p is not None and ms.zr >= 2 and not ms.epb:
                assert m & 0b111 == 0  # 00/01/02 never follow 00 00 in a payload
            if ms.epb:
                assert m >> 4 == 0  # after an EPB only 00..03
            ms.advance(byte)


def test_field_mask_equals_bitwise_reference():
    data = _fixture()
    if data is None:
        return
    ms = MaskStream()
    for i, byte in enumerate(data[:9000]):
        if i % 3 == 0:
            assert ms.mask() == ms.mask_reference(), f"offset {i} {ms.where()}"
        ms.advance(byte)


def test_random_probes_have_no_dead_ends():
    data = _fixture()
    if data is None:
        return
    rng = random.Random(3)
    ms = MaskStream()
    for i, byte in enumerate(data[:12000]):
        if i % 97 == 0:
            err = _probe(ms, rng, 48, byte)
            assert err is None, f"offset {i}: {err}"
        ms.advance(byte)


def test_corrupted_byte_is_rejected_somewhere():
    """Flipping a syntax byte (not opaque SEI payload) is caught within 400 bytes."""
    data = _fixture()
    if data is None:
        return
    candidates = [i for i, _, where in iter_masks(data[:6000])
                  if where not in ("sei_payload_byte", "start_code")]
    rng = random.Random(4)
    rejected = 0
    for pos in rng.sample(candidates, 20):
        bad = bytearray(data[: pos + 400])
        bad[pos] ^= 0xFF
        try:
            for _ in iter_masks(bytes(bad)):
                pass
        except Illegal:
            rejected += 1
    assert rejected >= 18, rejected


def _encode(tmp, name, x264_params, size="176x144", refs=3, qp=30):
    out = Path(tmp) / name
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc2=size={size}:rate=10,noise=alls=20:allf=t", "-t", "1.2", "-an",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", "-profile:v", "baseline",
           "-bf", "0", "-g", "8", "-threads", "1", "-refs", str(refs), "-qp", str(qp),
           "-x264-params", x264_params, "-f", "h264", str(out)]
    subprocess.run(cmd, check=True)
    return out.read_bytes()


def test_encoded_profiles_roundtrip():
    if shutil.which("ffmpeg") is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            ("frame", _encode(tmp, "frame.h264", "constrained-intra=1")),
            ("mb", _encode(tmp, "mb.h264", "slice-max-mbs=1", refs=3)),
        ]
        for layout, data in cases:
            n = sum(1 for _ in iter_masks(data[:20000], Profile(slice_layout=layout)))
            assert n == min(len(data), 20000)
