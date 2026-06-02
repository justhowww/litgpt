"""Preprocess source videos into a pinned H.264 byte corpus.

The pipeline is:
1. discover source videos under an input directory,
2. inspect source metadata with ffprobe,
3. re-encode each video into raw Annex-B H.264 with fixed codec settings,
4. write one JSONL manifest row per source video.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG_PATH = Path(__file__).with_name("h264_preprocess_config.json")


# -----------------------------------------------------------------------------
# Config types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FFmpegConfig:
    binary: str
    ffprobe_binary: str
    codec: str
    output_format: str
    profile: str
    disable_audio: bool
    disable_bframes: bool
    refs: int
    scene_cut_threshold: int
    x264_params: dict[str, int | str]


@dataclass(frozen=True)
class PreprocessConfig:
    width: int
    height: int
    resize_mode: str
    fps: int | None
    qp: int
    gop: int
    preset: str
    video_extensions: tuple[str, ...]
    ffmpeg: FFmpegConfig


@dataclass(frozen=True)
class VideoProbe:
    width: int | None
    height: int | None
    fps: float | None
    duration: float | None
    num_frames: int | None
    codec_name: str | None


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def load_config(path: Path) -> PreprocessConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    ffmpeg_raw = raw["ffmpeg"]
    ffmpeg = FFmpegConfig(
        binary=ffmpeg_raw.get("binary", "ffmpeg"),
        ffprobe_binary=ffmpeg_raw.get("ffprobe_binary", "ffprobe"),
        codec=ffmpeg_raw.get("codec", "libx264"),
        output_format=ffmpeg_raw.get("format", "h264"),
        profile=ffmpeg_raw.get("profile", "baseline"),
        disable_audio=ffmpeg_raw.get("disable_audio", True),
        disable_bframes=ffmpeg_raw.get("disable_bframes", True),
        refs=ffmpeg_raw.get("refs", 1),
        scene_cut_threshold=ffmpeg_raw.get("scene_cut_threshold", 0),
        x264_params=ffmpeg_raw.get("x264_params", {}),
    )

    return PreprocessConfig(
        width=raw["width"],
        height=raw["height"],
        resize_mode=raw.get("resize_mode", "crop_resize"),
        fps=raw["fps"],
        qp=raw["qp"],
        gop=raw["gop"],
        preset=raw["preset"],
        video_extensions=tuple(ext.lower() for ext in raw["video_extensions"]),
        ffmpeg=ffmpeg,
    )


def override_config(config: PreprocessConfig, args: argparse.Namespace) -> PreprocessConfig:
    values = {
        "width": args.width if args.width is not None else config.width,
        "height": args.height if args.height is not None else config.height,
        "resize_mode": args.resize_mode if args.resize_mode is not None else config.resize_mode,
        "fps": args.fps if args.fps is not None else config.fps,
        "qp": args.qp if args.qp is not None else config.qp,
        "gop": args.gop if args.gop is not None else config.gop,
        "preset": args.preset if args.preset is not None else config.preset,
    }
    return PreprocessConfig(
        **values,
        video_extensions=config.video_extensions,
        ffmpeg=config.ffmpeg,
    )


# -----------------------------------------------------------------------------
# Source discovery and probing
# -----------------------------------------------------------------------------


def discover_videos(input_dir: Path, extensions: tuple[str, ...]) -> Iterable[Path]:
    extension_set = set(extensions)
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() in extension_set:
                yield path


def probe_video(path: Path, ffprobe_binary: str) -> VideoProbe:
    cmd = [
        ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {path}")

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream found for {path}")

    stream = streams[0]
    return VideoProbe(
        width=stream.get("width"),
        height=stream.get("height"),
        fps=parse_fraction(stream.get("r_frame_rate")),
        duration=parse_optional_float(stream.get("duration")),
        num_frames=parse_optional_int(stream.get("nb_frames")),
        codec_name=stream.get("codec_name"),
    )


# -----------------------------------------------------------------------------
# FFmpeg command construction
# -----------------------------------------------------------------------------


def build_video_filter(config: PreprocessConfig) -> str:
    if config.resize_mode == "none":
        spatial_filter = None
    elif config.resize_mode == "resize":
        spatial_filter = f"scale={config.width}:{config.height}"
    elif config.resize_mode == "crop_resize":
        # Preserve target aspect ratio by center-cropping first, then resizing.
        spatial_filter = (
            f"crop=w='min(iw,ih*{config.width}/{config.height})':"
            f"h='min(ih,iw*{config.height}/{config.width})':"
            "x='(iw-ow)/2':y='(ih-oh)/2',"
            f"scale={config.width}:{config.height}"
        )
    else:
        raise ValueError(f"unsupported resize_mode: {config.resize_mode}")

    filters = []
    if spatial_filter:
        filters.append(spatial_filter)
    if config.fps is not None:
        filters.append(f"fps={config.fps}")
    return ",".join(filters)


def build_encode_command(input_path: Path, output_path: Path, config: PreprocessConfig) -> list[str]:
    cmd = [
        config.ffmpeg.binary,
        "-y",
        "-i",
        str(input_path),
    ]

    video_filter = build_video_filter(config)
    if video_filter:
        cmd.extend(["-vf", video_filter])

    if config.ffmpeg.disable_audio:
        cmd.append("-an")

    cmd.extend(
        [
            "-c:v",
            config.ffmpeg.codec,
            "-profile:v",
            config.ffmpeg.profile,
        ]
    )

    if config.ffmpeg.disable_bframes:
        cmd.extend(["-bf", "0"])

    cmd.extend(
        [
            "-refs",
            str(config.ffmpeg.refs),
            "-g",
            str(config.gop),
            "-keyint_min",
            str(config.gop),
            "-sc_threshold",
            str(config.ffmpeg.scene_cut_threshold),
            "-x264-params",
            build_x264_params(config),
            "-preset",
            config.preset,
            "-f",
            config.ffmpeg.output_format,
            str(output_path),
        ]
    )
    return cmd


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------


def encode_video(input_path: Path, output_path: Path, config: PreprocessConfig) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        build_encode_command(input_path, output_path, config),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg failed for {input_path}")


# -----------------------------------------------------------------------------
# Manifest rows
# -----------------------------------------------------------------------------


def probe_to_manifest(probe: VideoProbe | None) -> dict[str, Any] | None:
    if probe is None:
        return None
    return {
        "width": probe.width,
        "height": probe.height,
        "fps": probe.fps,
        "duration": probe.duration,
        "num_frames": probe.num_frames,
        "codec_name": probe.codec_name,
    }


def output_settings_to_manifest(config: PreprocessConfig) -> dict[str, Any]:
    return {
        "width": config.width,
        "height": config.height,
        "resize_mode": config.resize_mode,
        "fps": config.fps,
        "codec": "h264_baseline_cavlc",
        "qp": config.qp,
        "gop": config.gop,
        "preset": config.preset,
        "ffmpeg_codec": config.ffmpeg.codec,
        "profile": config.ffmpeg.profile,
        "refs": config.ffmpeg.refs,
        "x264_params": {**config.ffmpeg.x264_params, "qp": config.qp},
    }


def build_manifest_row(
    input_path: Path,
    output_path: Path,
    input_dir: Path,
    config: PreprocessConfig,
    status: str,
    source_probe: VideoProbe | None,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": str(input_path.relative_to(input_dir).with_suffix("")),
        "src_path": str(input_path),
        "h264_path": str(output_path),
        "source": probe_to_manifest(source_probe),
        "output": output_settings_to_manifest(config),
        "num_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "status": status,
    }
    if error:
        row["error"] = error
    return row


# -----------------------------------------------------------------------------
# Main preprocessing orchestration
# -----------------------------------------------------------------------------


def preprocess_videos(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    config: PreprocessConfig,
    skip_existing: bool,
    limit: int | None,
) -> None:
    h264_dir = output_dir / "h264"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for input_path in discover_videos(input_dir, config.video_extensions):
            if limit is not None and count >= limit:
                break

            relative = input_path.relative_to(input_dir).with_suffix(".h264")
            output_path = h264_dir / relative
            count += 1
            source_probe: VideoProbe | None = None

            try:
                source_probe = probe_video(input_path, config.ffmpeg.ffprobe_binary)
                if not (skip_existing and output_path.exists()):
                    encode_video(input_path, output_path, config)
                row = build_manifest_row(
                    input_path,
                    output_path,
                    input_dir,
                    config,
                    status="ok",
                    source_probe=source_probe,
                )
            except Exception as exc:
                row = build_manifest_row(
                    input_path,
                    output_path,
                    input_dir,
                    config,
                    status="failed",
                    source_probe=source_probe,
                    error=str(exc),
                )

            manifest.write(json.dumps(row) + "\n")
            manifest.flush()
            print(f"[{row['status']}] {input_path} -> {output_path}")


# -----------------------------------------------------------------------------
# Small parsing / formatting utilities
# -----------------------------------------------------------------------------


def parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    return float(Fraction(value))


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, "N/A"):
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value in (None, "N/A"):
        return None
    return int(value)


def build_x264_params(config: PreprocessConfig) -> str:
    params = dict(config.ffmpeg.x264_params)
    params["qp"] = config.qp
    return ":".join(f"{key}={value}" for key, value in params.items())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--resize-mode", choices=("none", "crop_resize", "resize"), default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--qp", type=int, default=None)
    parser.add_argument("--gop", type=int, default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = override_config(load_config(args.config), args)
    manifest = args.manifest or args.output_dir / "manifest.jsonl"
    preprocess_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=manifest,
        config=config,
        skip_existing=args.skip_existing,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
