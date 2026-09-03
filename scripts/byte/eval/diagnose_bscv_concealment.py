"""Compare FFmpeg concealment across the project's H.264 encodings.

The diagnostic samples source videos, encodes each source with the project,
AVC-LM, and recovered BSCV configurations, applies one deterministic BSCV
byte-excision setting, decodes with FFmpeg's default error concealment, and
writes a 3-row ``clean | corrupted`` comparison video.

Two corruption units are available:

``frame`` (default)
    Uses ffprobe access units, so ``corr_prob`` has the same frames-per-GOP
    meaning for one-slice/frame and multi-slice/frame encodings. This is the
    valid causal comparison across the three encoders.

``original-vcl``
    Reproduces BSCV corrupt_Gen.py's VCL-NAL-as-frame assumption, including
    fixed corr_pos and deletion length measured in hexadecimal characters.
    Use this for one-slice/frame BSCV streams. On AVC-LM, one VCL NAL is one
    macroblock rather than one frame, so this mode is deliberately not a fair
    cross-encoding comparison.
"""

from __future__ import annotations

import argparse
import binascii
import json
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPROCESS_DIR = REPO_ROOT / "preprocessing"
sys.path.insert(0, str(PREPROCESS_DIR))

from parse_mp4_to_h264 import encode_video, load_config  # noqa: E402

CONFIGS = {
    "project": PREPROCESS_DIR / "h264_preprocess_config.json",
    "avclm": PREPROCESS_DIR / "h264_preprocess_config_avclm.json",
    "bscv": PREPROCESS_DIR / "h264_preprocess_config_bscv.json",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


@dataclass(frozen=True)
class Cut:
    gop_index: int
    selected_index: int
    start_hex: int
    end_hex: int
    fallback: bool

    @property
    def deleted_hex_chars(self) -> int:
        return self.end_hex - self.start_hex


def _run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr[-4000:]}")
    return result


def _find_all(data: bytes, pattern: bytes) -> list[int]:
    out: list[int] = []
    start = 0
    while True:
        found = data.find(pattern, start)
        if found < 0:
            return out
        out.append(found)
        start = found + 1


def _splice_hex(data: bytes, cuts: list[Cut]) -> bytes:
    """Apply cuts exactly as hexadecimal-character ranges.

    The original BSCV script can start on an odd hex-character offset. This
    joins a high nibble from the left side with a low nibble from the right;
    preserving that behavior matters for an exact artifact reproduction.
    """
    encoded = binascii.hexlify(data)
    for cut in sorted(cuts, key=lambda item: item.start_hex, reverse=True):
        encoded = encoded[: cut.start_hex] + encoded[cut.end_hex :]
    if len(encoded) % 2:
        raise ValueError("BSCV cuts left an odd number of hexadecimal characters")
    return binascii.unhexlify(encoded)


def _choose_per_gop(items: list[int], count: int, seed: int, gop_index: int) -> list[int]:
    rng = random.Random((seed << 16) ^ gop_index)
    return sorted(rng.sample(items, min(count, len(items))))


def _make_cut(
    *,
    span_start_hex: int,
    span_end_hex: int,
    corr_pos: float,
    corr_len_hex: int,
    gop_index: int,
    selected_index: int,
) -> Cut:
    # corrupt_Gen.py uses x + 7, not x + 8. Keep this nibble-level quirk.
    available = span_end_hex - span_start_hex - 7 - corr_len_hex
    if available > 0:
        start = int(span_start_hex + 7 + available * corr_pos)
        return Cut(gop_index, selected_index, start, start + corr_len_hex, False)

    # Its short-fragment fallback preserves roughly the start code/header and
    # removes the remainder. Keep endpoints equal-parity so unhexlify remains
    # well-defined.
    start = min(span_start_hex + 9, span_end_hex)
    end = span_end_hex
    # corrupt_Gen.py retries at start+1/end+1 when the first splice leaves
    # odd hex length. At EOF end+1 is clipped, so its effective operation is
    # equivalent to advancing the odd start by one nibble.
    if (end - start) % 2:
        start = min(start + 1, end)
    return Cut(gop_index, selected_index, start, end, True)


def corrupt_original_vcl(
    data: bytes, *, corr_prob: int, corr_pos: float, corr_len_hex: int, seed: int
) -> tuple[bytes, list[Cut]]:
    """Reproduce corrupt_Gen.py's fixed-position VCL-NAL operator."""
    encoded = binascii.hexlify(data)
    starts = sorted(_find_all(encoded, b"000001"))
    sps = sorted(_find_all(encoded, b"00000167"))
    idr = sorted(_find_all(encoded, b"00000165"))
    frames = sorted(
        idr
        + _find_all(encoded, b"00000141")
        + _find_all(encoded, b"00000101")
    )
    if not starts or not idr or not frames:
        return data, []

    cuts: list[Cut] = []
    for gop_index, gop_start in enumerate(idr):
        following_sps = [position for position in sps if position > gop_start]
        gop_end = following_sps[0] if following_sps else len(encoded)
        units = [position for position in frames if gop_start <= position < gop_end]
        if not units:
            continue
        selected = _choose_per_gop(units, corr_prob, seed, gop_index)
        for selected_position in selected:
            following = [position for position in starts if position > selected_position]
            unit_end = following[0] if following else len(encoded)
            cuts.append(
                _make_cut(
                    span_start_hex=selected_position,
                    span_end_hex=unit_end,
                    corr_pos=corr_pos,
                    corr_len_hex=corr_len_hex,
                    gop_index=gop_index,
                    selected_index=frames.index(selected_position),
                )
            )
    return _splice_hex(data, cuts), cuts


def _probe_packets(path: Path, ffprobe: str) -> list[dict[str, Any]]:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pos,size,flags",
            "-of",
            "json",
            str(path),
        ]
    )
    packets = json.loads(result.stdout).get("packets", [])
    return [
        {"pos": int(item["pos"]), "size": int(item["size"]), "flags": item.get("flags", "")}
        for item in packets
        if item.get("pos") not in (None, "N/A") and item.get("size") not in (None, "N/A")
    ]


def corrupt_frames(
    data: bytes,
    packets: list[dict[str, Any]],
    *,
    corr_prob: int,
    corr_pos: float,
    corr_len_hex: int,
    seed: int,
) -> tuple[bytes, list[Cut]]:
    """Apply BSCV excision to semantic frame access units."""
    if not packets:
        return data, []
    keyframes = [index for index, packet in enumerate(packets) if "K" in packet["flags"]]
    if not keyframes:
        keyframes = [0]

    cuts: list[Cut] = []
    for gop_index, first in enumerate(keyframes):
        last = keyframes[gop_index + 1] if gop_index + 1 < len(keyframes) else len(packets)
        selected = _choose_per_gop(list(range(first, last)), corr_prob, seed, gop_index)
        for packet_index in selected:
            packet = packets[packet_index]
            start_hex = 2 * packet["pos"]
            end_hex = 2 * (packet["pos"] + packet["size"])
            cuts.append(
                _make_cut(
                    span_start_hex=start_hex,
                    span_end_hex=end_hex,
                    corr_pos=corr_pos,
                    corr_len_hex=corr_len_hex,
                    gop_index=gop_index,
                    selected_index=packet_index,
                )
            )
    return _splice_hex(data, cuts), cuts


def _decode_default(
    source: Path, output: Path, *, ffmpeg: str, fps: int, width: int, height: int, duration: float
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "+genpts",
        "-i",
        str(source),
        "-an",
        "-vf",
        vf,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr_lower = result.stderr.lower()
    warnings = []
    if "too many slices" in stderr_lower:
        warnings.append("decoder_slice_limit")
    if "concealing" in stderr_lower:
        warnings.append("decoder_concealment_used")
    if "invalid data" in stderr_lower or "error while decoding" in stderr_lower:
        warnings.append("decoder_bitstream_error")
    return {
        "ok": result.returncode == 0 and output.exists() and output.stat().st_size > 0,
        "returncode": result.returncode,
        "warnings": warnings,
        "stderr_tail": result.stderr[-4000:],
    }


def _placeholder(output: Path, *, ffmpeg: str, fps: int, width: int, height: int, duration: float) -> None:
    _run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}:d={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def _make_grid(
    inputs: list[tuple[Path, str]],
    output: Path,
    *,
    ffmpeg: str,
    fps: int,
    width: int,
    height: int,
    duration: float,
) -> None:
    filter_list = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True
    )
    has_drawtext = " drawtext " in filter_list.stdout
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "warning"]
    for path, _ in inputs:
        cmd += ["-i", str(path)]

    filters: list[str] = []
    labels: list[str] = []
    for index, (_, label) in enumerate(inputs):
        operations = (
            f"tpad=stop_mode=clone:stop_duration={duration},"
            f"trim=duration={duration},setpts=PTS-STARTPTS"
        )
        if has_drawtext:
            operations += (
                f",drawtext=text='{label}':fontcolor=white:fontsize=24:"
                "box=1:boxcolor=black@0.65:x=12:y=12"
            )
        filters.append(f"[{index}:v]{operations}[v{index}]")
        labels.append(f"[v{index}]")
    layout = f"0_0|{width}_0|0_{height}|{width}_{height}|0_{2 * height}|{width}_{2 * height}"
    filters.append("".join(labels) + f"xstack=inputs=6:layout={layout}:fill=black[out]")
    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-r",
        str(fps),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    _run(cmd)


def _sample_sources(input_dir: Path, count: int, seed: int) -> list[Path]:
    candidates = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(candidates) < count:
        raise ValueError(f"requested {count} videos, but found only {len(candidates)} under {input_dir}")
    return random.Random(seed).sample(candidates, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--num-videos", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--corr-prob", type=int, default=1, help="selected frames or VCL NALs per GOP")
    parser.add_argument("--corr-pos", type=float, default=0.4)
    parser.add_argument(
        "--corr-len-hex",
        type=int,
        default=2048,
        help="original BSCV unit: hexadecimal characters; 2048 means approximately 1024 bytes",
    )
    parser.add_argument("--corruption-unit", choices=("frame", "original-vcl"), default="frame")
    parser.add_argument("--display-fps", type=int, default=6)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.corr_pos <= 1.0:
        raise ValueError("--corr-pos must be in [0, 1]")
    if args.corr_len_hex <= 0 or args.corr_len_hex % 2:
        raise ValueError("--corr-len-hex must be a positive even number")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = _sample_sources(args.input_dir, args.num_videos, args.seed)
    report: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "corruption": {
            "unit": args.corruption_unit,
            "corr_prob": args.corr_prob,
            "corr_pos": args.corr_pos,
            "corr_len_hex": args.corr_len_hex,
            "nominal_deleted_bytes_per_cut": args.corr_len_hex / 2,
            "seed": args.seed,
        },
        "samples": [],
    }

    for sample_index, source in enumerate(sources):
        sample_dir = args.out_dir / f"sample_{sample_index:03d}_{source.stem}"
        sample_report: dict[str, Any] = {"source": str(source), "settings": {}}
        grid_inputs: list[tuple[Path, str]] = []

        for setting, config_path in CONFIGS.items():
            setting_dir = sample_dir / setting
            clean = setting_dir / "clean.h264"
            corrupted = setting_dir / "corrupted.h264"
            setting_dir.mkdir(parents=True, exist_ok=True)

            config = replace(load_config(config_path), clip_duration_sec=args.duration)
            if args.force or not clean.exists():
                encode_video(source, clean, config)
            clean_bytes = clean.read_bytes()
            if args.corruption_unit == "original-vcl":
                corrupted_bytes, cuts = corrupt_original_vcl(
                    clean_bytes,
                    corr_prob=args.corr_prob,
                    corr_pos=args.corr_pos,
                    corr_len_hex=args.corr_len_hex,
                    seed=args.seed + sample_index,
                )
            else:
                packets = _probe_packets(clean, config.ffmpeg.ffprobe_binary)
                corrupted_bytes, cuts = corrupt_frames(
                    clean_bytes,
                    packets,
                    corr_prob=args.corr_prob,
                    corr_pos=args.corr_pos,
                    corr_len_hex=args.corr_len_hex,
                    seed=args.seed + sample_index,
                )
            corrupted.write_bytes(corrupted_bytes)

            clean_mp4 = setting_dir / "clean_default_decode.mp4"
            corrupt_mp4 = setting_dir / "corrupted_default_decode.mp4"
            clean_decode = _decode_default(
                clean,
                clean_mp4,
                ffmpeg=config.ffmpeg.binary,
                fps=args.display_fps,
                width=args.panel_width,
                height=args.panel_height,
                duration=args.duration,
            )
            corrupt_decode = _decode_default(
                corrupted,
                corrupt_mp4,
                ffmpeg=config.ffmpeg.binary,
                fps=args.display_fps,
                width=args.panel_width,
                height=args.panel_height,
                duration=args.duration,
            )
            if not clean_decode["ok"]:
                raise RuntimeError(f"clean decode failed for {clean}: {clean_decode['stderr_tail']}")
            if clean_decode["warnings"]:
                print(
                    f"WARNING: {setting} clean decode reported {clean_decode['warnings']}; "
                    "inspect clean_default_decode.mp4 before interpreting corruption",
                    file=sys.stderr,
                )
            if not corrupt_decode["ok"]:
                _placeholder(
                    corrupt_mp4,
                    ffmpeg=config.ffmpeg.binary,
                    fps=args.display_fps,
                    width=args.panel_width,
                    height=args.panel_height,
                    duration=args.duration,
                )

            actual_deleted_hex = sum(cut.deleted_hex_chars for cut in cuts)
            sample_report["settings"][setting] = {
                "config": str(config_path),
                "clean_bytes": len(clean_bytes),
                "corrupted_bytes": len(corrupted_bytes),
                "actual_deleted_bytes": (len(clean_bytes) - len(corrupted_bytes)),
                "actual_deleted_hex_chars_from_records": actual_deleted_hex,
                "cuts": [asdict(cut) for cut in cuts],
                "clean_decode": clean_decode,
                "corrupted_decode": corrupt_decode,
            }
            (setting_dir / "corruption.json").write_text(
                json.dumps(sample_report["settings"][setting], indent=2) + "\n"
            )
            grid_inputs += [(clean_mp4, f"{setting} clean"), (corrupt_mp4, f"{setting} corrupted default concealment")]

        comparison = sample_dir / "comparison.mp4"
        (sample_dir / "layout.txt").write_text(
            "project clean | project corrupted (default concealment)\n"
            "avclm clean   | avclm corrupted (default concealment)\n"
            "bscv clean    | bscv corrupted (default concealment)\n"
        )
        _make_grid(
            grid_inputs,
            comparison,
            ffmpeg="ffmpeg",
            fps=args.display_fps,
            width=args.panel_width,
            height=args.panel_height,
            duration=args.duration,
        )
        sample_report["comparison"] = str(comparison)
        report["samples"].append(sample_report)
        print(f"wrote {comparison}")

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "samples.txt").write_text("\n".join(map(str, sources)) + "\n")
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
