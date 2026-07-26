from pathlib import Path

from litgpt.byte.data import NALUnit
from scripts.byte.eval import eval_ar_continuation as ar_eval


def test_next_frame_boundary_does_not_consume_window_budget(monkeypatch):
    """A 1+1 window fits even when the excluded third frame is very large."""
    monkeypatch.setattr(ar_eval.HS, "slice_first_mb", lambda _payload: 0)
    nals = [
        NALUnit(0, 100, 3, 7),
        NALUnit(100, 200, 3, 8),
        NALUnit(200, 7_200, 3, 5),
        NALUnit(7_200, 15_200, 3, 1),
        NALUnit(15_200, 35_200, 3, 1),
    ]

    clip = ar_eval._first_qualifying_window(
        bytes(35_200),
        Path("clip.h264"),
        nals,
        needed_frames=2,
        max_bytes=16_384,
        prefix_frames=1,
    )

    assert clip is not None
    assert clip.prefix_end_nal == 3
    assert clip.cont_end_nal == 4
