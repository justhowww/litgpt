"""Shared synchronized video renderer for AR continuation and FIM repair evals."""

from __future__ import annotations

import json
import subprocess
import tempfile
from fractions import Fraction
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


def _probe_fps(path: Path, ffprobe_binary: str, fallback: float) -> float:
    try:
        result = subprocess.run(
            [
                ffprobe_binary,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        stream = json.loads(result.stdout).get("streams", [{}])[0]
        # Raw H.264 has no container duration, and ffprobe commonly reports a
        # synthetic 25 fps ``avg_frame_rate``. ``r_frame_rate`` comes from the
        # bitstream timing and is the meaningful value for these streams.
        for key in ("r_frame_rate", "avg_frame_rate"):
            value = stream.get(key)
            if value and value != "0/0":
                fps = float(Fraction(value))
                if fps > 0:
                    return fps
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        IndexError,
    ):
        pass
    return float(fallback)


def save_stream_comparison_video(
    *,
    clean_stream: bytes,
    corrupted_stream: bytes,
    repaired_stream: bytes,
    strict_corrupted_frame_count: int,
    concealed_corrupted_frame_count: int,
    repaired_frame_count: int,
    reference_frame_count: int,
    frame_height: int,
    frame_width: int,
    target_frame: int,
    out_path: Path,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    fps: int,
    timeout_sec: int,
    columns: tuple[str, str, str, str],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Render four H.264 streams on one FFmpeg presentation timeline.

    Unlike :func:`save_comparison_video`, this does not zip decoded frames by
    Python-list index. FFmpeg decodes each raw H.264 stream in presentation order
    and normalizes it onto one common frame-rate timeline. Missing tails become
    dark red instead of shifting later decoded frames into an earlier slot.
    """
    if not clean_stream or reference_frame_count <= 0:
        return False
    if len(columns) != 4:
        raise ValueError("stream comparison requires exactly four column labels")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{out_path.stem}-", dir=out_path.parent
    ) as temp_name:
        temp_dir = Path(temp_name)
        clean_path = temp_dir / "clean.h264"
        corrupted_path = temp_dir / "corrupted.h264"
        repaired_path = temp_dir / "repaired.h264"
        clean_path.write_bytes(clean_stream)
        corrupted_path.write_bytes(corrupted_stream)
        repaired_path.write_bytes(repaired_stream)

        source_fps = _probe_fps(clean_path, ffprobe_binary, fallback=float(fps))
        duration = reference_frame_count / source_fps
        target_time = min(max(target_frame, 0), reference_frame_count) / source_fps
        strict_flags = [
            "-ec",
            "0",
            "-err_detect",
            "explode+bitstream+buffer+compliant",
        ]
        panel_specs = (
            (clean_path, True, reference_frame_count),
            (corrupted_path, True, strict_corrupted_frame_count),
            (corrupted_path, False, concealed_corrupted_frame_count),
            (repaired_path, True, repaired_frame_count),
        )
        command = [ffmpeg_binary, "-y", "-hide_banner", "-loglevel", "error"]
        for path, strict, frame_count in panel_specs:
            if frame_count > 0:
                # Raw Annex-B carries no container PTS. Assign the bitstream's
                # probed frame rate at demux time so FFmpeg creates a common
                # presentation clock before framesync.
                command.extend(["-fflags", "+genpts", "-r", str(source_fps)])
                if strict:
                    command.extend(strict_flags)
                command.extend(["-f", "h264", "-i", str(path)])
            else:
                command.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        (
                            f"color=c=0x8c0000:s={frame_width}x{frame_height}:"
                            f"r={source_fps}:d={duration}"
                        ),
                    ]
                )

        filters: list[str] = []
        for index, (_, _, frame_count) in enumerate(panel_specs):
            # A zero-frame stream was replaced by a full-duration red lavfi
            # input above. Do not immediately trim that placeholder to zero.
            filter_frame_count = (
                reference_frame_count
                if frame_count <= 0
                else min(frame_count, reference_frame_count)
            )
            chain = [
                f"trim=end_frame={filter_frame_count}",
                "setpts=PTS-STARTPTS",
                f"fps={fps}",
                f"scale={frame_width}:{frame_height}:flags=neighbor",
                f"tpad=stop_mode=add:stop_duration={duration}:color=0x8c0000",
                f"trim=duration={duration}",
                (
                    "drawbox=x=0:y=0:w=iw:h=ih:color=green:t=4:"
                    f"enable='lt(t,{target_time})'"
                ),
                (
                    "drawbox=x=0:y=0:w=iw:h=ih:color=red:t=4:"
                    f"enable='gte(t,{target_time})'"
                ),
            ]
            if index < 3:
                chain.append("pad=iw+4:ih:0:0:white")
            filters.append(f"[{index}:v]{','.join(chain)}[v{index}]")
        filters.append("[v0][v1][v2][v3]hstack=inputs=4[out]")
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-an",
                "-r",
                str(fps),
                "-t",
                str(duration),
                "-c:v",
                "mpeg4",
                "-q:v",
                "3",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
        )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=max(timeout_sec, 30),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        if (
            result.returncode != 0
            or not out_path.is_file()
            or out_path.stat().st_size == 0
        ):
            return False

        thumbnail = out_path.with_name(f"{out_path.stem}_target.png")
        try:
            subprocess.run(
                [
                    ffmpeg_binary,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(target_time),
                    "-i",
                    str(out_path),
                    "-frames:v",
                    "1",
                    str(thumbnail),
                ],
                capture_output=True,
                timeout=timeout_sec,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        sidecar = {
            "columns": list(columns),
            "renderer": "ffmpeg_timestamp_aligned_h264",
            "fps": fps,
            "source_fps": source_fps,
            "duration_seconds": duration,
            "target_frame": target_frame,
            "target_time_seconds": target_time,
            "green_border": "presentation time occurs before the task boundary",
            "red_border": "task-boundary presentation time or later",
            "dark_red_tile": "stream produced no frame or ended before this time",
            "ffmpeg_returncode": result.returncode,
            "ffmpeg_stderr_tail": result.stderr.decode(
                "utf-8", errors="replace"
            )[-4000:],
            **(metadata or {}),
        }
        out_path.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
        )
        return True


def save_comparison_video(
    *,
    reference_frames: list[Tensor],
    result_frames: list[Tensor],
    middle_frames: list[Tensor] | None = None,
    second_middle_frames: list[Tensor] | None = None,
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
    """Write a two-, three-, or four-column aligned comparison video.

    ``left_blank_from`` or ``middle_blank_from`` turns that panel into an AR-input
    view: observed prefix frames remain visible and unknown continuation frames are
    black. ``border_changes_at`` draws green borders before the task boundary and
    red borders from the first target frame onward on every panel. Passing
    ``middle_frames`` produces ``reference | input | result``.
    ``second_middle_frames`` adds a second view of that input before the result,
    as used for strict-versus-concealed FIM corruption decoding.
    """
    if not reference_frames:
        return False
    if second_middle_frames is not None and middle_frames is None:
        raise ValueError("second_middle_frames requires middle_frames")
    expected_columns = 2 + int(middle_frames is not None) + int(
        second_middle_frames is not None
    )
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
        second_middle = None
        if second_middle_frames is not None:
            second_middle = (
                second_middle_frames[frame_index]
                if frame_index < len(second_middle_frames)
                and second_middle_frames[frame_index].shape == gt_frame.shape
                else failed
            )
        if border_changes_at is not None:
            color = (
                (0.0, 0.85, 0.0) if frame_index < border_changes_at else (1.0, 0.0, 0.0)
            )
            left = _border(left, color)
            if middle is not None:
                middle = _border(middle, color)
            if second_middle is not None:
                second_middle = _border(second_middle, color)
            right = _border(right, color)
        row = (
            (
                left,
                separator,
                middle,
                separator,
                second_middle,
                separator,
                right,
            )
            if second_middle is not None
            else (
                (left, separator, middle, separator, right)
                if middle is not None
                else (left, separator, right)
            )
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
