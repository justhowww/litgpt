"""FIM fill rules in the H.264 mask: no new picture inside a hole, DPB-bounded refs.

Encodes a tiny one-slice-per-frame CAVLC clip with FFmpeg/libx264 (skipped when
unavailable) and replays it through ``h264_mask`` exactly as the FIM evaluator does.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from litgpt.byte import h264_mask as HM

START_CODE = b"\x00\x00\x01"
MAX_REFS = 2


def _encode(num_frames: int = 6) -> bytes:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=10:duration={num_frames / 10}",
        "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
        "-g", str(num_frames), "-bf", "0", "-refs", str(MAX_REFS),
        "-x264-params", "slices=1:sliced-threads=0",
        "-f", "h264", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        pytest.skip(f"libx264 encode failed: {result.stderr[:200]!r}")
    return result.stdout


def _vcl_starts(stream: bytes) -> list[int]:
    """Byte offsets of the start code of every VCL NAL (one per frame here)."""
    starts, i = [], 0
    while (j := stream.find(START_CODE, i)) >= 0:
        sc = j - 1 if j > 0 and stream[j - 1] == 0 else j
        if stream[j + 3] & 0x1F in (1, 5):
            starts.append(sc)
        i = j + 3
    return starts


def _seed(prefix: bytes) -> HM.MaskState:
    state = HM.MaskState(slice_max_mbs=None, fail_closed=True)
    for byte in prefix:
        HM.advance(state, byte)
    state.generation_started = True
    HM.restrict_fill_to_current_picture(state)
    return state


def _feed(state: HM.MaskState, data: bytes) -> int | None:
    for i, byte in enumerate(data):
        if not HM.get_valid_byte_mask(state)[byte]:
            return i
        HM.advance(state, byte)
    return None


def test_hole_inside_a_frame_accepts_gt_and_forbids_a_new_picture():
    stream = _encode()
    frames = _vcl_starts(stream)
    lo, hi = frames[3], frames[4]
    split = lo + (hi - lo) // 2  # inside frame 3's slice data
    state = _seed(stream[:split])
    assert state.new_picture_budget == 0

    # The ground-truth rest of the frame is accepted and completes the picture.
    assert HM.can_append_bytes(state, stream[split:hi], require_complete=True)

    # After the frame ends, trailing zeros are fine but 00 00 01 is not.
    assert _feed(state, stream[split:hi]) is None
    assert _feed(state, b"\x00\x00") is None
    assert not HM.get_valid_byte_mask(state)[0x01]


def test_hole_at_a_frame_start_allows_exactly_that_frames_start_code():
    stream = _encode()
    frames = _vcl_starts(stream)
    lo, hi, nxt = frames[2], frames[3], frames[4]
    state = _seed(stream[:lo])
    assert state.new_picture_budget == 1

    # The fill may rebuild frame 2 from its start code ...
    assert _feed(state, stream[lo:hi]) is None
    # ... but not continue into frame 3.
    assert _feed(state, stream[hi:nxt]) is not None


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_hole_at_the_idr_start_after_parameter_sets(offset):
    stream = _encode()
    frames = _vcl_starts(stream)
    lo, hi = frames[0], frames[1]
    state = _seed(stream[: lo + offset])
    assert state.new_picture_budget == 1
    assert HM.can_append_bytes(state, stream[lo + offset : hi], require_complete=True)


def test_reference_count_is_bounded_by_max_num_ref_frames():
    stream = _encode()
    frames = _vcl_starts(stream)
    state = _seed(stream[: frames[-1]])
    assert state.picture.reference_pictures == MAX_REFS
