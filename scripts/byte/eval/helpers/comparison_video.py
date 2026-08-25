"""Shared synchronized video renderer for AR continuation and FIM repair evals."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only affects optional thumbnails.
    Image = None


def _save_png(frame: Tensor, path: Path) -> None:
    if Image is None:
        return
    array = frame.detach().cpu().mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(array, mode="RGB").save(path)


def _border(frame: Tensor, color: tuple[float, float, float], width: int = 4) -> Tensor:
    out = frame.clone()
    w = max(1, min(width, out.shape[0] // 4, out.shape[1] // 4))
    rgb = torch.tensor(color, dtype=out.dtype, device=out.device)
    out[:w, :, :] = rgb
    out[-w:, :, :] = rgb
    out[:, :w, :] = rgb
    out[:, -w:, :] = rgb
    return out


def save_comparison_video(
    *,
    reference_frames: list[Tensor],
    result_frames: list[Tensor],
    middle_frames: list[Tensor] | None = None,
    out_path: Path,
    ffmpeg_binary: str,
    fps: int,
    timeout_sec: int,
    columns: tuple[str, ...],
    left_blank_from: int | None = None,
    middle_blank_from: int | None = None,
    border_changes_at: int | None = None,
    thumbnail_frame: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Write a two- or three-column video aligned to the reference timeline.

    ``left_blank_from`` or ``middle_blank_from`` turns that panel into an AR-input
    view: observed prefix frames remain visible and unknown continuation frames are
    black. ``border_changes_at`` draws green borders before the task boundary and
    red borders from the first target frame onward on every panel. Passing
    ``middle_frames`` produces ``reference | input | result``; omitted preserves a
    two-column ``reference | result`` layout.
    """
    if not reference_frames:
        return False
    expected_columns = 3 if middle_frames is not None else 2
    if len(columns) != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} column labels, received {len(columns)}"
        )
    reference = reference_frames[0]
    blank = torch.zeros_like(reference)
    failed = torch.zeros_like(reference)
    failed[..., 0] = 0.55
    separator = torch.ones(
        (reference.shape[0], 4, reference.shape[2]), dtype=reference.dtype
    )
    panels: list[Tensor] = []
    for frame_index, gt_frame in enumerate(reference_frames):
        left = (
            blank
            if left_blank_from is not None and frame_index >= left_blank_from
            else gt_frame
        )
        right = (
            result_frames[frame_index]
            if frame_index < len(result_frames)
            and result_frames[frame_index].shape == gt_frame.shape
            else failed
        )
        middle = None
        if middle_frames is not None:
            middle = (
                blank
                if middle_blank_from is not None
                and frame_index >= middle_blank_from
                else (
                    middle_frames[frame_index]
                    if frame_index < len(middle_frames)
                    and middle_frames[frame_index].shape == gt_frame.shape
                    else failed
                )
            )
        if border_changes_at is not None:
            color = (
                (0.0, 0.85, 0.0) if frame_index < border_changes_at else (1.0, 0.0, 0.0)
            )
            left = _border(left, color)
            if middle is not None:
                middle = _border(middle, color)
            right = _border(right, color)
        row = (
            (left, separator, middle, separator, right)
            if middle is not None
            else (left, separator, right)
        )
        panels.append(torch.cat(row, dim=1).clamp(0, 1))

    h, w = panels[0].shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        panels = [
            torch.nn.functional.pad(p.permute(2, 0, 1), (0, pad_w, 0, pad_h))
            .permute(1, 2, 0)
            .contiguous()
            for p in panels
        ]
    height, width = panels[0].shape[:2]
    buffer = b"".join(
        panel.mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous()
        .numpy()
        .tobytes()
        for panel in panels
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(
            command, input=buffer, capture_output=True, check=True, timeout=timeout_sec,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return False

    thumb_index = (
        min(max(int(thumbnail_frame), 0), len(panels) - 1)
        if thumbnail_frame is not None
        else 0
    )
    _save_png(panels[thumb_index], out_path.with_name(f"{out_path.stem}_target.png"))
    sidecar = {
        "columns": list(columns),
        "fps": fps,
        "left_blank_from": left_blank_from,
        "middle_blank_from": middle_blank_from,
        "border_changes_at": border_changes_at,
        "green_border": "frame occurs before the task boundary",
        "red_border": "target frame or a later frame",
        "dark_red_tile": "comparison frame failed to decode or is unavailable",
        **(metadata or {}),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    return True
