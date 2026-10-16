"""Parser-state probe labels agree with the parser's own syntax spans."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from litgpt.byte import h264_syntax as HS
from litgpt.byte.probe_labels import (
    CLASS_ID,
    MAX_BITS_INTO,
    element_class,
    parser_state_labels,
)


def _encode(num_frames: int = 6) -> bytes:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=10:duration={num_frames / 10}",
        "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
        "-g", str(num_frames), "-bf", "0", "-refs", "2",
        "-x264-params", "slices=1:sliced-threads=0",
        "-f", "h264", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        pytest.skip(f"libx264 encode failed: {result.stderr[:200]!r}")
    return result.stdout


def test_labels_cover_every_boundary_and_match_spans():
    stream = _encode()
    labels = parser_state_labels(stream)
    assert len(labels.element) == len(stream) + 1

    parse = HS.parse_stream(stream, parse_slice_data=True)
    checked = 0
    for nal_parse in parse.nals:
        nal = nal_parse.nal
        assert labels.element[nal.start_code_start] == CLASS_ID["start_code"]
        assert labels.element[nal.payload_start] == CLASS_ID["nal_header"]
        _, byte_map, epb = HS.unescape_rbsp(stream, nal.payload_start + 1, nal.payload_end)
        rbsp_index = {offset: i for i, offset in enumerate(byte_map)}
        for span in nal_parse.spans:
            if span.bit_start < 0:
                continue
            # Every byte boundary strictly inside the element's bits must carry it.
            for k in range(span.byte_start + 1, span.byte_end):
                if k not in rbsp_index:
                    continue  # emulation-prevention byte
                bit = 8 * rbsp_index[k]
                if not span.bit_start <= bit < span.bit_end:
                    continue
                assert labels.element[k] == element_class(span), (span.name, k)
                assert labels.bits_into[k] == min(MAX_BITS_INTO, bit - span.bit_start)
                checked += 1
    assert checked > 50


def test_macroblock_positions_are_inside_the_picture():
    stream = _encode()
    labels = parser_state_labels(stream)
    sps = next(iter(HS.parse_stream(stream).sps.values()))
    mb_bounds = [
        (x, y) for x, y in zip(labels.mb_x, labels.mb_y) if x >= 0 or y >= 0
    ]
    assert mb_bounds, "no macroblock-data boundaries labelled"
    assert all(0 <= x < sps.pic_width_in_mbs for x, _ in mb_bounds)
    assert all(0 <= y < sps.pic_height_in_mbs for _, y in mb_bounds)
