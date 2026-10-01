"""Slice-end regeneration helpers (scripts/byte/eval/helpers/slice_end.py).

Both masks must accept the GT tail of a cut frame and report the slice complete
exactly at the frame end; a random legal completion under the new mask must be
a complete slice that FFmpeg strictly decodes.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if "litgpt.byte.h264_mask" not in sys.modules:
    # Load the legacy mask without litgpt/__init__ (no torch needed).
    from syntax_mask.compare_legacy import _load_legacy

    _load_legacy(REPO)

from scripts.byte.eval.helpers import slice_end as SE  # noqa: E402

FIXTURE = REPO / "tests" / "byte" / "fixtures" / "baseline_qcif.h264"


def _cuts(data: bytes):
    nals = SE.nal_units(data)
    vcl = [(i, s, t) for i, (s, t) in enumerate(nals) if t in SE.VCL_TYPES]
    picks = [v for v in vcl if v[2] == 5][:1] + [v for v in vcl if v[2] == 1][2:4]
    for i, lo, t in picks:
        hi = nals[i + 1][0] if i + 1 < len(nals) else len(data)
        for cut in (0.2, 0.8):
            yield lo, hi, lo + 4 + int((hi - lo - 4) * cut), ("idr" if t == 5 else "p")


@pytest.mark.parametrize("kind", ["new", "old"])
def test_gt_tail_accepted_and_slice_end_detected(kind):
    data = FIXTURE.read_bytes()
    for lo, hi, split, ftype in _cuts(data):
        assert SE.frame_type_at(data, lo) == ftype
        mask = SE.make_mask(kind, data[:split])
        for byte in data[split:hi]:
            assert not mask.slice_complete()
            assert mask.allowed()[byte]
            mask.advance(byte)
        assert mask.slice_complete()


def test_random_legal_completion_is_a_complete_slice():
    data = FIXTURE.read_bytes()
    ffmpeg = shutil.which("ffmpeg")
    for n, (_lo, hi, split, _ftype) in enumerate(_cuts(data)):
        res = SE.random_legal_completion(data[:split], random.Random(n), max_bytes=200_000)
        assert res.stop_reason == "slice_end"
        legality = SE.replay_legality(data[:split], res.data)
        assert legality["all_legal"] and legality["ends_at_slice_end"]
        if ffmpeg:
            stream = data[:split] + res.data + data[hi:]
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-ec", "0",
                 "-err_detect", "explode+bitstream+buffer+compliant",
                 "-f", "h264", "-i", "pipe:0", "-f", "null", "-"],
                input=stream, capture_output=True,
            )
            assert proc.returncode == 0 and not proc.stderr, proc.stderr[:200]


def test_frame_index_counts_vcl_nals():
    data = FIXTURE.read_bytes()
    starts = [s for s, t in SE.nal_units(data) if t in SE.VCL_TYPES]
    assert SE.frame_index_at(data, starts[0]) == 0
    assert SE.frame_index_at(data, starts[3]) == 3
    assert SE.frame_index_at(data, len(data)) == len(starts)
