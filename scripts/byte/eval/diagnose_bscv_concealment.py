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
import re
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


def _probe_fps(path: Path, ffprobe: str) -> float:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = json.loads(result.stdout).get("streams", [{}])[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key, "0/0")
        numerator, denominator = value.split("/", 1)
        if float(denominator) and float(numerator):
            return float(numerator) / float(denominator)
    raise ValueError(f"could not determine frame rate for {path}")


def _probe_display_packet_positions(path: Path, ffprobe: str) -> list[int | None]:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=pkt_pos",
            "-of",
            "json",
            str(path),
        ]
    )
    return [
        int(frame["pkt_pos"]) if frame.get("pkt_pos") not in (None, "N/A") else None
        for frame in json.loads(result.stdout).get("frames", [])
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


def shared_display_frame_schedule(
    num_frames: int, *, gop_size: int, corr_prob: int, seed: int
) -> list[int]:
    """Choose a deterministic schedule independent of encoder GOP decisions."""
    selected: list[int] = []
    for gop_index, first in enumerate(range(0, num_frames, gop_size)):
        last = min(first + gop_size, num_frames)
        selected.extend(
            _choose_per_gop(list(range(first, last)), corr_prob, seed, gop_index)
        )
    return selected


def corrupt_display_frames(
    data: bytes,
    packets: list[dict[str, Any]],
    display_packet_positions: list[int | None],
    selected_display_frames: list[int],
    *,
    corr_pos: float,
    corr_len_hex: int,
    gop_size: int,
) -> tuple[bytes, list[Cut]]:
    """Corrupt exact display-frame indices, including with reordered B-frames."""
    packet_index_by_position = {packet["pos"]: index for index, packet in enumerate(packets)}
    cuts: list[Cut] = []
    for display_index in selected_display_frames:
        if not 0 <= display_index < len(display_packet_positions):
            continue
        packet_position = display_packet_positions[display_index]
        if packet_position is None or packet_position not in packet_index_by_position:
            continue
        packet_index = packet_index_by_position[packet_position]
        packet = packets[packet_index]
        cuts.append(
            _make_cut(
                span_start_hex=2 * packet["pos"],
                span_end_hex=2 * (packet["pos"] + packet["size"]),
                corr_pos=corr_pos,
                corr_len_hex=corr_len_hex,
                gop_index=display_index // gop_size,
                selected_index=packet_index,
            )
        )
    return _splice_hex(data, cuts), cuts


def common_eligible_display_schedule(
    streams: dict[str, tuple[list[dict[str, Any]], list[int | None]]],
    *,
    gop_size: int,
    corr_prob: int,
    corr_len_hex: int,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Choose identical full-length cuts that fit every encoding in a group."""
    if not streams:
        raise ValueError("common-cut schedule requires at least one stream")
    num_frames = min(len(display_positions) for _, display_positions in streams.values())
    packet_maps = {
        name: {packet["pos"]: packet for packet in packets}
        for name, (packets, _) in streams.items()
    }
    selected: list[int] = []
    eligible_by_gop: dict[int, list[int]] = {}
    for gop_index, first in enumerate(range(0, num_frames, gop_size)):
        last = min(first + gop_size, num_frames)
        eligible: list[int] = []
        for display_index in range(first, last):
            fits_all = True
            for name, (_, display_positions) in streams.items():
                position = display_positions[display_index]
                packet = packet_maps[name].get(position) if position is not None else None
                if packet is None or 2 * packet["size"] - 7 - corr_len_hex <= 0:
                    fits_all = False
                    break
            if fits_all:
                eligible.append(display_index)
        eligible_by_gop[gop_index] = eligible
        if len(eligible) < corr_prob:
            names = ", ".join(streams)
            raise ValueError(
                f"strict common corruption cannot select {corr_prob} frame(s) in nominal GOP "
                f"{gop_index}: only {len(eligible)} frame(s) can hold the full "
                f"{corr_len_hex // 2}-byte cut across [{names}]. Lower --corr-len-hex; "
                "the diagnostic will not use unequal short-frame fallbacks."
            )
        selected.extend(_choose_per_gop(eligible, corr_prob, seed, gop_index))
    return selected, {
        "num_common_frames": num_frames,
        "eligible_display_frames_by_nominal_gop": eligible_by_gop,
        "selected_display_frames": selected,
        "actual_bytes_per_cut": corr_len_hex // 2,
    }


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
        "0",
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
    if len(inputs) % 2:
        raise ValueError("comparison grid requires clean/corrupted input pairs")
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
    layout = "|".join(
        f"{column * width}_{row * height}"
        for row in range(len(inputs) // 2)
        for column in range(2)
    )
    filters.append("".join(labels) + f"xstack=inputs={len(inputs)}:layout={layout}:fill=black[out]")
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


def _make_native_triptych(
    clean: Path,
    corrupted: Path,
    output: Path,
    *,
    ffmpeg: str,
    fps: float,
    width: int,
    height: int,
    duration: float,
    affected_ranges: list[tuple[int, int]],
) -> None:
    filter_list = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True
    )
    has_drawtext = " drawtext " in filter_list.stdout
    frame_expression = "+".join(
        f"between(n\\,{first}\\,{last})" for first, last in affected_ranges
    )
    border = (
        f"drawbox=x=0:y=0:w=iw:h=ih:color=red:t=8:enable='{frame_expression}'"
        if frame_expression
        else "null"
    )
    clean_label = (
        ",drawtext=text='clean':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.65:x=12:y=12"
        if has_drawtext
        else ""
    )
    corrupt_label = (
        ",drawtext=text='corrupted default concealment':fontcolor=white:fontsize=24:"
        "box=1:boxcolor=black@0.65:x=12:y=12"
        if has_drawtext
        else ""
    )
    diff_label = (
        ",drawtext=text='grayscale absolute difference x4':fontcolor=white:fontsize=24:"
        "box=1:boxcolor=black@0.65:x=12:y=12"
        if has_drawtext
        else ""
    )
    filters = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},"
        f"setpts=PTS-STARTPTS,split=2[c][cd];"
        f"[1:v]tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration},"
        f"setpts=PTS-STARTPTS,split=2[x][xd];"
        f"[c]{border}{clean_label}[co];"
        f"[x]{border}{corrupt_label}[xo];"
        f"[cd][xd]blend=all_mode=difference,format=gray,lutyuv=y=val*4,format=yuv420p{diff_label}[d];"
        "[co][xo][d]hstack=inputs=3[out]"
    )
    _run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(clean),
            "-i",
            str(corrupted),
            "-filter_complex",
            filters,
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
    )


_PSNR_PATTERN = re.compile(r"n:(?P<n>\d+).*?psnr_avg:(?P<psnr>[-+a-zA-Z0-9.]+)")


def _measure_psnr(
    clean: Path,
    corrupted: Path,
    stats_path: Path,
    *,
    ffmpeg: str,
    cut_frames: set[int],
    follow_frames: int,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "+genpts",
            "-i",
            str(clean),
            "-fflags",
            "+genpts",
            "-i",
            str(corrupted),
            "-lavfi",
            (
                "[0:v]settb=AVTB,setpts=PTS-STARTPTS[a];"
                "[1:v]settb=AVTB,setpts=PTS-STARTPTS[b];"
                f"[a][b]psnr=stats_file={stats_path}"
            ),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    if stats_path.exists():
        for line in stats_path.read_text().splitlines():
            match = _PSNR_PATTERN.search(line)
            if not match:
                continue
            frame_index = int(match.group("n")) - 1
            value = match.group("psnr")
            infinite = value.lower() == "inf"
            psnr = None if infinite else float(value)
            rows.append(
                {
                    "frame_index": frame_index,
                    "psnr_db": psnr,
                    "psnr_is_infinite": infinite,
                    "is_corrupted_frame": frame_index in cut_frames,
                    "within_following_window": any(
                        cut <= frame_index <= cut + follow_frames for cut in cut_frames
                    ),
                }
            )
    finite = [row["psnr_db"] for row in rows if row["psnr_db"] is not None]
    payload = {
        "ok": result.returncode == 0 and bool(rows),
        "paired_frames": len(rows),
        "finite_psnr_frames": len(finite),
        "mean_finite_psnr_db": sum(finite) / len(finite) if finite else None,
        "minimum_psnr_db": min(finite) if finite else None,
        "frames": rows,
        "stderr_tail": result.stderr[-4000:],
    }
    stats_path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _factorial_configs(duration: float, slices: int) -> dict[str, tuple[Any, str]]:
    base = replace(load_config(CONFIGS["project"]), clip_duration_sec=duration)
    out: dict[str, tuple[Any, str]] = {}
    for cabac in (False, True):
        for slice_count in (1, slices):
            params = dict(base.ffmpeg.x264_params)
            params.pop("slice-max-mbs", None)
            params["cabac"] = int(cabac)
            params["slices"] = slice_count
            ffmpeg_config = replace(
                base.ffmpeg,
                profile="high" if cabac else "baseline",
                x264_params=params,
            )
            name = f"{'cabac' if cabac else 'cavlc'}_{slice_count:02d}slice"
            out[name] = (
                replace(base, ffmpeg=ffmpeg_config),
                f"controlled factorial: cabac={int(cabac)}, slices/frame={slice_count}",
            )
    return out


def _bscv_ablation_configs(duration: float) -> dict[str, tuple[Any, str]]:
    base = replace(load_config(CONFIGS["bscv"]), clip_duration_sec=duration)

    no_b_params = dict(base.ffmpeg.x264_params)
    no_b_params["bframes"] = 0
    no_b_params["b-pyramid"] = 0
    no_b = replace(
        base,
        ffmpeg=replace(base.ffmpeg, disable_bframes=True, x264_params=no_b_params),
    )
    ref1 = replace(base, ffmpeg=replace(base.ffmpeg, refs=1))
    qp28 = replace(base, qp=28)
    no_scenecut = replace(
        base,
        ffmpeg=replace(base.ffmpeg, scene_cut_threshold=0),
    )
    return {
        "bscv_no_bframes": (no_b, "BSCV exact except B-frames disabled"),
        "bscv_ref1": (ref1, "BSCV exact except references reduced from 3 to 1"),
        "bscv_qp28": (qp28, "BSCV exact except QP changed from 1 to 28"),
        "bscv_no_scenecut": (no_scenecut, "BSCV exact except scene-cut keyframes disabled"),
    }


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
    parser.add_argument(
        "--following-frames",
        type=int,
        default=16,
        help="mark and score this many frames after every corrupted frame",
    )
    parser.add_argument(
        "--factorial-slices",
        type=int,
        default=16,
        help="frequent-slice level in the controlled CABAC/CAVLC x slice-count ablation",
    )
    parser.add_argument("--no-factorial", action="store_true")
    parser.add_argument(
        "--no-bscv-ablation",
        action="store_true",
        help="skip the BSCV-origin B-frame/reference/QP/scene-cut ablation",
    )
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
        "comparison_scopes": {
            "comparison.mp4": "observational only; real presets intentionally differ in resolution/fps/codec",
            "factorial_comparison.mp4": "controlled: common source geometry/fps/GOP and identical full-length cuts",
            "bscv_ablation_comparison.mp4": "controlled: BSCV base geometry/fps/GOP and identical full-length cuts",
        },
        "samples": [],
    }

    for sample_index, source in enumerate(sources):
        sample_dir = args.out_dir / f"sample_{sample_index:03d}_{source.stem}"
        sample_report: dict[str, Any] = {"source": str(source), "settings": {}}
        real_settings = {
            name: (replace(load_config(path), clip_duration_sec=args.duration), str(path))
            for name, path in CONFIGS.items()
        }
        factorial_settings = (
            {} if args.no_factorial else _factorial_configs(args.duration, args.factorial_slices)
        )
        bscv_ablation_settings = (
            {} if args.no_bscv_ablation else _bscv_ablation_configs(args.duration)
        )
        settings = {**real_settings, **factorial_settings, **bscv_ablation_settings}
        real_grid_inputs: list[tuple[Path, str]] = []
        factorial_grid_inputs: list[tuple[Path, str]] = []
        bscv_ablation_grid_inputs: list[tuple[Path, str]] = []

        # Encode and probe every stream before selecting controlled cuts. This
        # is necessary because eligibility is an intersection across variants.
        prepared: dict[str, dict[str, Any]] = {}
        for setting, (config, config_description) in settings.items():
            setting_dir = sample_dir / setting
            clean = setting_dir / "clean.h264"
            setting_dir.mkdir(parents=True, exist_ok=True)
            if args.force or not clean.exists():
                encode_video(source, clean, config)
            prepared[setting] = {
                "config": config,
                "config_description": config_description,
                "setting_dir": setting_dir,
                "clean": clean,
                "packets": _probe_packets(clean, config.ffmpeg.ffprobe_binary),
                "display_positions": _probe_display_packet_positions(
                    clean, config.ffmpeg.ffprobe_binary
                ),
            }

        controlled_schedule_by_setting: dict[str, list[int]] = {}
        controlled_group_by_setting: dict[str, str] = {}
        controlled_schedule_reports: dict[str, Any] = {}
        if args.corruption_unit == "frame":
            controlled_groups = {}
            if factorial_settings:
                controlled_groups["cabac_by_slice_count"] = list(factorial_settings)
            if bscv_ablation_settings:
                controlled_groups["bscv_feature_ablation"] = ["bscv", *bscv_ablation_settings]
            for group_name, group_settings in controlled_groups.items():
                geometry = {
                    (
                        prepared[name]["config"].width,
                        prepared[name]["config"].height,
                        prepared[name]["config"].resize_mode,
                        prepared[name]["config"].fps,
                        prepared[name]["config"].gop,
                        prepared[name]["config"].clip_duration_sec,
                    )
                    for name in group_settings
                }
                if len(geometry) != 1:
                    raise ValueError(
                        f"controlled group {group_name} differs in geometry, fps, GOP, or duration"
                    )
                group_streams = {
                    name: (prepared[name]["packets"], prepared[name]["display_positions"])
                    for name in group_settings
                }
                gop_sizes = {prepared[name]["config"].gop for name in group_settings}
                if len(gop_sizes) != 1:
                    raise ValueError(f"controlled group {group_name} does not share one GOP size")
                schedule, schedule_report = common_eligible_display_schedule(
                    group_streams,
                    gop_size=gop_sizes.pop(),
                    corr_prob=args.corr_prob,
                    corr_len_hex=args.corr_len_hex,
                    seed=args.seed + sample_index,
                )
                controlled_schedule_reports[group_name] = {
                    "settings": group_settings,
                    **schedule_report,
                }
                for name in group_settings:
                    controlled_schedule_by_setting[name] = schedule
                    controlled_group_by_setting[name] = group_name
        sample_report["controlled_schedules"] = controlled_schedule_reports

        for setting, (config, config_description) in settings.items():
            item = prepared[setting]
            setting_dir = item["setting_dir"]
            clean = item["clean"]
            corrupted = setting_dir / "corrupted.h264"
            clean_bytes = clean.read_bytes()
            packets: list[dict[str, Any]] = item["packets"]
            selected_display_frames: list[int] = []
            display_positions: list[int | None] = item["display_positions"]
            if args.corruption_unit == "original-vcl":
                corrupted_bytes, cuts = corrupt_original_vcl(
                    clean_bytes,
                    corr_prob=args.corr_prob,
                    corr_pos=args.corr_pos,
                    corr_len_hex=args.corr_len_hex,
                    seed=args.seed + sample_index,
                )
            else:
                if setting in controlled_schedule_by_setting:
                    selected_display_frames = controlled_schedule_by_setting[setting]
                    corrupted_bytes, cuts = corrupt_display_frames(
                        clean_bytes,
                        packets,
                        display_positions,
                        selected_display_frames,
                        corr_pos=args.corr_pos,
                        corr_len_hex=args.corr_len_hex,
                        gop_size=config.gop,
                    )
                    if any(cut.fallback for cut in cuts):
                        raise AssertionError(f"controlled setting {setting} unexpectedly used a short fallback")
                    expected_deleted = len(cuts) * (args.corr_len_hex // 2)
                    actual_deleted = len(clean_bytes) - len(corrupted_bytes)
                    if actual_deleted != expected_deleted:
                        raise AssertionError(
                            f"controlled setting {setting} deleted {actual_deleted} bytes; "
                            f"expected exactly {expected_deleted}"
                        )
                else:
                    corrupted_bytes, cuts = corrupt_frames(
                        clean_bytes,
                        packets,
                        corr_prob=args.corr_prob,
                        corr_pos=args.corr_pos,
                        corr_len_hex=args.corr_len_hex,
                        seed=args.seed + sample_index,
                    )
            corrupted.write_bytes(corrupted_bytes)

            position_to_display = {
                position: index for index, position in enumerate(display_positions) if position is not None
            }
            if packets:
                cut_packet_positions = [
                    packets[cut.selected_index]["pos"]
                    for cut in cuts
                    if 0 <= cut.selected_index < len(packets)
                ]
                cut_frames = {
                    position_to_display[position]
                    for position in cut_packet_positions
                    if position in position_to_display
                }
            else:
                cut_packet_positions = []
                cut_frames = set()
            affected_ranges = sorted(
                (frame, frame + args.following_frames) for frame in cut_frames
            )

            native_fps = float(config.fps) if config.fps is not None else _probe_fps(
                clean, config.ffmpeg.ffprobe_binary
            )
            clean_native_mp4 = setting_dir / "clean_native_default_decode.mp4"
            corrupt_native_mp4 = setting_dir / "corrupted_native_default_decode.mp4"
            clean_native_decode = _decode_default(
                clean,
                clean_native_mp4,
                ffmpeg=config.ffmpeg.binary,
                fps=max(1, round(native_fps)),
                width=args.panel_width,
                height=args.panel_height,
                duration=args.duration,
            )
            corrupt_native_decode = _decode_default(
                corrupted,
                corrupt_native_mp4,
                ffmpeg=config.ffmpeg.binary,
                fps=max(1, round(native_fps)),
                width=args.panel_width,
                height=args.panel_height,
                duration=args.duration,
            )
            if not corrupt_native_decode["ok"]:
                _placeholder(
                    corrupt_native_mp4,
                    ffmpeg=config.ffmpeg.binary,
                    fps=max(1, round(native_fps)),
                    width=args.panel_width,
                    height=args.panel_height,
                    duration=args.duration,
                )
            native_triptych = setting_dir / "native_clean_corrupted_difference.mp4"
            _make_native_triptych(
                clean_native_mp4,
                corrupt_native_mp4,
                native_triptych,
                ffmpeg=config.ffmpeg.binary,
                fps=native_fps,
                width=args.panel_width,
                height=args.panel_height,
                duration=args.duration,
                affected_ranges=affected_ranges,
            )
            psnr = _measure_psnr(
                clean,
                corrupted,
                setting_dir / "per_frame_psnr.log",
                ffmpeg=config.ffmpeg.binary,
                cut_frames=cut_frames,
                follow_frames=args.following_frames,
            )

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
            clean_warnings = sorted(set(clean_decode["warnings"] + clean_native_decode["warnings"]))
            if clean_warnings:
                print(
                    f"WARNING: {setting} clean decode reported {clean_warnings}; "
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
                "config": config_description,
                "resolved_config": asdict(config),
                "clean_bytes": len(clean_bytes),
                "corrupted_bytes": len(corrupted_bytes),
                "actual_deleted_bytes": (len(clean_bytes) - len(corrupted_bytes)),
                "actual_deleted_hex_chars_from_records": actual_deleted_hex,
                "cuts": [asdict(cut) for cut in cuts],
                "cut_packet_positions": cut_packet_positions,
                "cut_display_frame_indices": sorted(cut_frames),
                "requested_shared_display_frame_indices": selected_display_frames,
                "controlled_comparison_group": controlled_group_by_setting.get(setting),
                "affected_display_frame_ranges": affected_ranges,
                "native_fps": native_fps,
                "clean_decode": clean_decode,
                "corrupted_decode": corrupt_decode,
                "clean_native_decode": clean_native_decode,
                "corrupted_native_decode": corrupt_native_decode,
                "native_triptych": str(native_triptych),
                "psnr": psnr,
            }
            (setting_dir / "corruption.json").write_text(
                json.dumps(sample_report["settings"][setting], indent=2) + "\n"
            )
            pair = [(clean_mp4, f"{setting} clean"), (corrupt_mp4, f"{setting} corrupted default concealment")]
            if setting in real_settings:
                real_grid_inputs += pair
                if setting == "bscv" and bscv_ablation_settings:
                    bscv_ablation_grid_inputs += pair
            elif setting in factorial_settings:
                factorial_grid_inputs += pair
            else:
                bscv_ablation_grid_inputs += pair

        comparison = sample_dir / "comparison.mp4"
        (sample_dir / "layout.txt").write_text(
            "project clean | project corrupted (default concealment)\n"
            "avclm clean   | avclm corrupted (default concealment)\n"
            "bscv clean    | bscv corrupted (default concealment)\n"
        )
        _make_grid(
            real_grid_inputs,
            comparison,
            ffmpeg="ffmpeg",
            fps=args.display_fps,
            width=args.panel_width,
            height=args.panel_height,
            duration=args.duration,
        )
        sample_report["comparison"] = str(comparison)
        if factorial_grid_inputs:
            factorial_comparison = sample_dir / "factorial_comparison.mp4"
            (sample_dir / "factorial_layout.txt").write_text(
                "CAVLC, 1 slice/frame: clean | corrupted\n"
                f"CAVLC, {args.factorial_slices} slices/frame: clean | corrupted\n"
                "CABAC, 1 slice/frame: clean | corrupted\n"
                f"CABAC, {args.factorial_slices} slices/frame: clean | corrupted\n"
            )
            _make_grid(
                factorial_grid_inputs,
                factorial_comparison,
                ffmpeg="ffmpeg",
                fps=args.display_fps,
                width=args.panel_width,
                height=args.panel_height,
                duration=args.duration,
            )
            sample_report["factorial_comparison"] = str(factorial_comparison)
        if bscv_ablation_grid_inputs:
            bscv_ablation_comparison = sample_dir / "bscv_ablation_comparison.mp4"
            (sample_dir / "bscv_ablation_layout.txt").write_text(
                "BSCV exact: clean | corrupted\n"
                "BSCV without B-frames: clean | corrupted\n"
                "BSCV with one reference: clean | corrupted\n"
                "BSCV at QP 28: clean | corrupted\n"
                "BSCV without scene cuts: clean | corrupted\n"
            )
            _make_grid(
                bscv_ablation_grid_inputs,
                bscv_ablation_comparison,
                ffmpeg="ffmpeg",
                fps=args.display_fps,
                width=args.panel_width,
                height=args.panel_height,
                duration=args.duration,
            )
            sample_report["bscv_ablation_comparison"] = str(bscv_ablation_comparison)
        report["samples"].append(sample_report)
        print(f"wrote {comparison}")

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out_dir / "samples.txt").write_text("\n".join(map(str, sources)) + "\n")
    print(f"wrote {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
