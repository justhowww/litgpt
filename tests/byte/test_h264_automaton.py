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


def _small_automaton(*, is_p=True, max_mbs=1, num_ref=3):
    return A.MbAutomaton(
        pic_width_in_mbs=16,
        pic_height_in_mbs=9,
        slice_type=H.SLICE_TYPE_P if is_p else H.SLICE_TYPE_I,
        num_ref_idx_l0_active=num_ref,
        first_mb_in_slice=0,
        slice_data_start_bit=0,
        max_mbs=max_mbs,
    )


def _ue(value):
    code_num = value + 1
    suffix = f"{code_num:b}"
    return "0" * (len(suffix) - 1) + suffix


def _se(value):
    code_num = 2 * abs(value) - (1 if value > 0 else 0)
    return _ue(code_num)


def _consume(auto, bits):
    status = A.MORE
    for bit in bits:
        status = auto.consume_bit(int(bit))
    return status


def _vlc(label, value):
    return next(
        code for code, decoded in A.T.code_map(label).items() if decoded == value
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


def test_skip_run_cannot_exceed_profile_slice_size():
    assert _consume(_small_automaton(), _ue(1)) == A.COMPLETE_MB
    assert _consume(_small_automaton(), _ue(2)) == A.INVALID


def test_standard_macroblock_type_boundaries_keep_p_slice_intra_valid():
    p_30 = _small_automaton()
    p_30._start("ue", "mb_type")
    assert _consume(p_30, _ue(30)) != A.INVALID
    p_31 = _small_automaton()
    p_31._start("ue", "mb_type")
    assert _consume(p_31, _ue(31)) == A.INVALID
    assert _consume(_small_automaton(is_p=False), _ue(25)) != A.INVALID
    assert _consume(_small_automaton(is_p=False), _ue(26)) == A.INVALID


def test_truncated_exp_golomb_reference_index_checks_xmax():
    accepted = _small_automaton()
    accepted._start("te", "ref_idx", xmax=2)
    assert _consume(accepted, _ue(2)) != A.INVALID
    rejected = _small_automaton()
    rejected._start("te", "ref_idx", xmax=2)
    assert _consume(rejected, _ue(3)) == A.INVALID


def test_intra_chroma_prediction_mode_boundaries():
    accepted = _small_automaton(is_p=False)
    accepted.mb_mode = "I_NxN"
    accepted._start("ue", "intra_chroma")
    assert _consume(accepted, _ue(3)) != A.INVALID
    rejected = _small_automaton(is_p=False)
    rejected.mb_mode = "I_NxN"
    rejected._start("ue", "intra_chroma")
    assert _consume(rejected, _ue(4)) == A.INVALID


def test_baseline_mb_qp_delta_boundaries():
    for value in (-26, 25):
        accepted = _small_automaton()
        accepted.cbp_luma = 0
        accepted.cbp_chroma = 0
        accepted._start("se", "mb_qp_delta")
        assert _consume(accepted, _se(value)) != A.INVALID
    for value in (-27, 26):
        rejected = _small_automaton()
        rejected._start("se", "mb_qp_delta")
        assert _consume(rejected, _se(value)) == A.INVALID


def test_coeff_token_cannot_exceed_block_capacity():
    rejected = _small_automaton()
    rejected._rb_start(nc=0, max_coeff=15, set_target=None)
    assert _consume(rejected, _vlc("coeff_token_0", (16, 0))) == A.INVALID


def test_level_prefix_cannot_exceed_baseline_limit():
    accepted = _small_automaton()
    accepted.rb_total_coeff = 1
    accepted.rb_trailing_ones = 0
    accepted.rb_suffix_length = 0
    accepted.rb_i = 0
    accepted._start("unary", "level_prefix")
    assert _consume(accepted, "0" * 28 + "1") != A.INVALID

    rejected = _small_automaton()
    rejected.rb_total_coeff = 1
    rejected.rb_trailing_ones = 0
    rejected.rb_suffix_length = 0
    rejected.rb_i = 0
    rejected._start("unary", "level_prefix")
    assert _consume(rejected, "0" * 29) == A.INVALID


def test_total_zeros_cannot_exceed_remaining_block_capacity():
    rejected = _small_automaton()
    rejected.rb_max = 15
    rejected.rb_total_coeff = 1
    rejected._start("vlc", "total_zeros", label="total_zeros_4x4_1")
    assert _consume(rejected, _vlc("total_zeros_4x4_1", 15)) == A.INVALID


def test_run_before_cannot_exceed_zeros_left():
    rejected = _small_automaton()
    rejected.rb_total_coeff = 2
    rejected.rb_zeros_left = 7
    rejected.rb_run_i = 0
    rejected.res_phase = "luma"
    rejected.res_blk = 16
    rejected.cbp_chroma = 0
    rejected._start("vlc", "run_before", label="run_before_7")
    assert _consume(rejected, _vlc("run_before_7", 14)) == A.INVALID

    # The invalid run has a long code. Its impossible first byte must be pruned
    # now, rather than accepted until every next-byte choice becomes invalid.
    rejected = _small_automaton()
    rejected.rb_total_coeff = 2
    rejected.rb_zeros_left = 7
    rejected.rb_run_i = 0
    rejected.res_phase = "luma"
    rejected.res_blk = 16
    rejected.cbp_chroma = 0
    rejected._start("vlc", "run_before", label="run_before_7")
    invalid = _vlc("run_before_7", 14)
    first_byte = int((invalid + "0" * 8)[:8], 2)
    assert not A.compile_byte_mask(rejected)[first_byte]


def test_full_byte_mask_rejects_oversized_skip_run_prefix():
    mask = A.compile_byte_mask(_small_automaton(), residual_only=False)
    # ue(2) == 011, so every byte with that high-bit prefix must be rejected.
    assert not any(mask[0b01100000:0b10000000])


def test_memoized_byte_mask_is_exact_and_reuses_root():
    compiler = A.MemoizedByteMaskCompiler(
        max_cache_entries=100_000,
        collect_field_cardinality=True,
    )
    auto = _small_automaton()
    compiler.record_corpus_byte()
    assert compiler.compile_byte_mask(auto) == A.compile_byte_mask(auto)

    before = compiler.statistics()
    compiler.record_corpus_byte()
    assert compiler.compile_byte_mask(auto) == A.compile_byte_mask(auto)
    after = compiler.statistics()
    assert after["root_hits"] == before["root_hits"] + 1
    assert (
        after["transition_computations"]
        == before["transition_computations"]
    )
    assert after["transition_computations_per_byte"] is not None
    assert after["signature_field_cardinality"]


def test_memoized_cache_key_changes_after_committed_grid_write():
    compiler = A.MemoizedByteMaskCompiler(max_cache_entries=100_000)
    auto = _small_automaton()
    compiler.record_corpus_byte()
    compiler.compile_byte_mask(auto)
    before = compiler.statistics()

    auto._set_luma(0, 0, 0, 3)
    compiler.record_corpus_byte()
    assert compiler.compile_byte_mask(auto) == A.compile_byte_mask(auto)
    after = compiler.statistics()
    assert after["root_misses"] == before["root_misses"] + 1


def test_memoized_compiler_exercises_committed_grid_coeff_context():
    auto = _small_automaton()
    auto.cur_mbx = 1
    auto.cur_mby = 1
    auto.res_phase = "luma"
    auto.res_blk = 1
    auto.cbp_luma = 1
    auto.cbp_chroma = 0
    auto._rb_start(nc=0, max_coeff=16, set_target=("L", 0))

    compiler = A.MemoizedByteMaskCompiler(max_cache_entries=100_000)
    compiler.record_corpus_byte()
    assert compiler.compile_byte_mask(auto) == A.compile_byte_mask(auto)
    statistics = compiler.statistics()
    assert statistics["coeff_context_crossings"] > 0
    assert statistics["committed_grid_bucket_crossings"] > 0
    assert statistics["bucket_crossings_by_coeff_token_label"]


def test_memoized_compiler_aborts_instead_of_evicting():
    compiler = A.MemoizedByteMaskCompiler(max_cache_entries=1)
    try:
        compiler.compile_byte_mask(_small_automaton())
    except A.MaskCacheLimitError as exc:
        assert "no entries were evicted" in str(exc)
    else:
        raise AssertionError("hard cache limit did not abort")


def test_automaton_rejects_slice_extent_outside_picture():
    try:
        A.MbAutomaton(
            pic_width_in_mbs=16,
            pic_height_in_mbs=9,
            slice_type=H.SLICE_TYPE_P,
            num_ref_idx_l0_active=3,
            first_mb_in_slice=144,
            slice_data_start_bit=0,
            max_mbs=1,
        )
    except ValueError as exc:
        assert "first_mb_in_slice" in str(exc)
    else:
        raise AssertionError("out-of-picture first_mb_in_slice was accepted")


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
