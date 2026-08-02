"""Measure storage savings from short, self-contained BSCV H.264 segments.

For one deterministic reservoir sample of source videos, encode four paired
variants with the verified BSCV x264 configuration:

* full source duration (control),
* 1 GOP = 16 output frames,
* 2 GOPs = 32 output frames,
* 4 GOPs = 64 output frames.

All short variants use the same deterministic temporal start for a source. Each
new raw Annex-B stream starts with fresh SPS/PPS and an IDR because it is a new
x264 encode. The report verifies that property, strict FFmpeg decoding, exact
frame count, byte size, paired savings, and projected size for one million
videos.

The output directory is resumable: ``samples.json`` pins the selected videos,
non-empty streams are reused, and the latest row for each sample/variant wins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.parse_mp4_to_h264 import (  # noqa: E402
    PreprocessConfig,
    build_encode_command,
    discover_videos,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gop-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--projection-videos", type=int, default=1_000_000)
    parser.add_argument("--encode-timeout", type=int, default=1800)
    parser.add_argument("--skip-full", action="store_true")
    return parser.parse_args()


def reservoir_sample(paths: Iterable[Path], size: int, seed: int) -> list[Path]:
    if size < 1:
        raise ValueError("sample size must be positive")
    rng = random.Random(seed)
    sample: list[Path] = []
    seen = 0
    for seen, path in enumerate(paths, start=1):
        if len(sample) < size:
            sample.append(path)
            continue
        replacement = rng.randrange(seen)
        if replacement < size:
            sample[replacement] = path
    if seen < size:
        raise RuntimeError(f"requested {size} samples but found only {seen} videos")
    return sorted(sample)


def probe_duration(path: Path, ffprobe: str, timeout: int) -> float | None:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    values = [
        stream.get("duration") for stream in payload.get("streams", [])
    ] + [payload.get("format", {}).get("duration")]
    durations: list[float] = []
    for value in values:
        if value not in (None, "N/A"):
            try:
                durations.append(float(value))
            except (TypeError, ValueError):
                pass
    return max(durations) if durations else None


def deterministic_start(
    relative_path: str,
    seed: int,
    duration: float | None,
    longest_clip_seconds: float,
) -> float:
    if duration is None:
        return 0.0
    maximum = max(0.0, duration - longest_clip_seconds - 0.25)
    if maximum == 0.0:
        return 0.0
    return random.Random(f"{seed}:{relative_path}").uniform(0.0, maximum)


def load_or_select_samples(
    source_root: Path,
    output_dir: Path,
    config: PreprocessConfig,
    sample_size: int,
    seed: int,
    max_frames: int,
    timeout: int,
) -> list[dict[str, Any]]:
    sample_path = output_dir / "samples.json"
    if sample_path.exists():
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        if len(samples) != sample_size:
            raise RuntimeError(
                f"existing {sample_path} has {len(samples)} samples, requested {sample_size}"
            )
        return samples

    selected = reservoir_sample(
        discover_videos(source_root, config.video_extensions), sample_size, seed
    )
    fps = config.fps
    if fps is None or fps <= 0:
        raise ValueError("the pilot requires a positive fixed output fps")
    longest_seconds = max_frames / fps
    samples = []
    for index, source in enumerate(selected):
        relative = str(source.relative_to(source_root))
        duration = probe_duration(source, config.ffmpeg.ffprobe_binary, timeout)
        samples.append(
            {
                "sample_id": index,
                "source_path": str(source),
                "relative_path": relative,
                "source_bytes": source.stat().st_size,
                "source_duration_sec": duration,
                "clip_start_sec": deterministic_start(
                    relative, seed, duration, longest_seconds
                ),
            }
        )
    sample_path.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    return samples


def variant_command(
    source: Path,
    output: Path,
    config: PreprocessConfig,
    start_sec: float | None,
    frames: int | None,
) -> list[str]:
    cmd = build_encode_command(source, output, config)
    if start_sec is not None and start_sec > 0:
        input_index = cmd.index("-i")
        # Input-side accurate seek avoids decoding the entire source before a
        # late random crop. Re-encoding still starts the new stream at an IDR.
        cmd[input_index:input_index] = ["-ss", f"{start_sec:.6f}"]
    if frames is not None:
        format_index = cmd.index("-f")
        cmd[format_index:format_index] = ["-frames:v", str(frames)]
    return cmd


def encode_atomic(cmd: list[str], output: Path, timeout: int) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".tmp.h264", dir=output.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    cmd = [str(tmp) if arg == str(output) else arg for arg in cmd]
    started = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-4000:] or "FFmpeg encode failed")
        if tmp.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty stream")
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return time.monotonic() - started


def annexb_nal_types(data: bytes) -> list[int]:
    types: list[int] = []
    index = 0
    while index + 3 < len(data):
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            header = index + 4
            index = header
        elif data[index : index + 3] == b"\x00\x00\x01":
            header = index + 3
            index = header
        else:
            index += 1
            continue
        if header < len(data):
            types.append(data[header] & 0x1F)
    return types


def header_is_self_contained(nal_types: list[int]) -> bool:
    first_vcl_index = next(
        (index for index, nal_type in enumerate(nal_types) if 1 <= nal_type <= 5),
        None,
    )
    if first_vcl_index is None:
        return False
    before_vcl = nal_types[:first_vcl_index]
    return 7 in before_vcl and 8 in before_vcl and nal_types[first_vcl_index] == 5


def probe_output(path: Path, config: PreprocessConfig, timeout: int) -> dict[str, Any]:
    ffprobe_cmd = [
        config.ffmpeg.ffprobe_binary,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=nb_read_frames,profile,level,pix_fmt,color_range,"
            "has_b_frames,r_frame_rate,avg_frame_rate"
        ),
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(
        ffprobe_cmd, capture_output=True, text=True, timeout=timeout
    )
    stream: dict[str, Any] = {}
    if probe.returncode == 0:
        streams = json.loads(probe.stdout).get("streams", [])
        if streams:
            stream = streams[0]

    decode_cmd = [
        config.ffmpeg.binary,
        "-v",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        str(path),
        "-f",
        "null",
        "-",
    ]
    decode = subprocess.run(
        decode_cmd, capture_output=True, text=True, timeout=timeout
    )
    data = path.read_bytes()
    nal_types = annexb_nal_types(data)
    required_options = (
        b"cabac=1",
        b"ref=3",
        b"bframes=3",
        b"keyint=16",
        b"keyint_min=1",
        b"scenecut=40",
        b"rc=cqp",
        b"qp=1",
    )
    return {
        "ffprobe_ok": probe.returncode == 0,
        "decode_ok": decode.returncode == 0,
        "decode_error": decode.stderr[-2000:] if decode.returncode else None,
        "frame_count": int(stream["nb_read_frames"])
        if stream.get("nb_read_frames") not in (None, "N/A")
        else None,
        "profile": stream.get("profile"),
        "level": stream.get("level"),
        "pix_fmt": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "has_b_frames": stream.get("has_b_frames"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nal_types_head": nal_types[:16],
        "self_contained_header": header_is_self_contained(nal_types),
        "x264_options_match": all(option in data for option in required_options),
    }


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def latest_rows(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["sample_id"], row["variant"])] = row
    return rows


def valid_row(row: dict[str, Any]) -> bool:
    target = row.get("target_frames")
    return bool(
        row.get("error") is None
        and row.get("ffprobe_ok")
        and row.get("decode_ok")
        and row.get("self_contained_header")
        and row.get("x264_options_match")
        and (target is None or row.get("frame_count") == target)
    )


def summarize(
    rows: dict[tuple[int, str], dict[str, Any]],
    variants: list[tuple[str, int | None]],
    projection_videos: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"projection_videos": projection_videos, "variants": {}}
    full_by_sample = {
        sample_id: row
        for (sample_id, variant), row in rows.items()
        if variant == "full" and valid_row(row)
    }
    for variant, target_frames in variants:
        selected = [
            row
            for (_sample_id, name), row in rows.items()
            if name == variant and valid_row(row)
        ]
        sizes = [int(row["output_bytes"]) for row in selected]
        paired_savings = []
        if variant != "full":
            for row in selected:
                full = full_by_sample.get(row["sample_id"])
                if full:
                    paired_savings.append(1.0 - row["output_bytes"] / full["output_bytes"])
        entry: dict[str, Any] = {
            "target_frames": target_frames,
            "valid": len(selected),
            "attempted": sum(1 for (_sid, name) in rows if name == variant),
        }
        if sizes:
            avg = statistics.mean(sizes)
            entry.update(
                {
                    "mean_bytes": avg,
                    "median_bytes": statistics.median(sizes),
                    "p95_bytes": percentile(sizes, 0.95),
                    "projected_decimal_tb": avg * projection_videos / 1e12,
                    "projected_tib": avg * projection_videos / (2**40),
                    "paired_savings_mean": statistics.mean(paired_savings)
                    if paired_savings
                    else None,
                }
            )
        summary["variants"][variant] = entry
    return summary


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# BSCV short-segment storage pilot",
        "",
        f"Projection population: {summary['projection_videos']:,} videos",
        "",
        "| Variant | Target frames | Valid | Mean/video | P95/video | Projected | Savings vs full |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in summary["variants"].items():
        if "mean_bytes" not in entry:
            lines.append(
                f"| {name} | {entry['target_frames'] or 'full'} | "
                f"{entry['valid']}/{entry['attempted']} | — | — | — | — |"
            )
            continue
        savings = entry.get("paired_savings_mean")
        lines.append(
            f"| {name} | {entry['target_frames'] or 'full'} | "
            f"{entry['valid']}/{entry['attempted']} | "
            f"{entry['mean_bytes']/1e6:.2f} MB | "
            f"{entry['p95_bytes']/1e6:.2f} MB | "
            f"{entry['projected_decimal_tb']:.2f} TB | "
            f"{savings*100:.1f}% |" if savings is not None else
            f"| {name} | {entry['target_frames'] or 'full'} | "
            f"{entry['valid']}/{entry['attempted']} | "
            f"{entry['mean_bytes']/1e6:.2f} MB | "
            f"{entry['p95_bytes']/1e6:.2f} MB | "
            f"{entry['projected_decimal_tb']:.2f} TB | control |"
        )
    lines.extend(
        [
            "",
            "A row is valid only when FFprobe succeeds, strict FFmpeg decoding succeeds,",
            "the stream begins with SPS/PPS followed by an IDR, the recovered x264 options",
            "match the BSCV configuration, and short variants contain exactly the requested",
            "number of decoded frames.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if any(count < 1 for count in args.gop_counts):
        raise ValueError("all GOP counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    max_frames = max(args.gop_counts) * config.gop
    samples = load_or_select_samples(
        args.source_root,
        args.output_dir,
        config,
        args.sample_size,
        args.seed,
        max_frames,
        args.encode_timeout,
    )
    variants: list[tuple[str, int | None]] = []
    if not args.skip_full:
        variants.append(("full", None))
    variants.extend((f"gop{count}", count * config.gop) for count in args.gop_counts)

    metrics_path = args.output_dir / "metrics.jsonl"
    rows = latest_rows(metrics_path)
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for sample in samples:
            source = Path(sample["source_path"])
            relative = Path(sample["relative_path"])
            for variant, target_frames in variants:
                key = (sample["sample_id"], variant)
                if key in rows and valid_row(rows[key]):
                    print(f"[reuse] sample={sample['sample_id']} variant={variant}")
                    continue
                output = args.output_dir / "streams" / variant / relative.with_suffix(".h264")
                row: dict[str, Any] = {
                    **sample,
                    "variant": variant,
                    "target_frames": target_frames,
                    "output_path": str(output),
                    "error": None,
                }
                try:
                    elapsed = 0.0
                    if not output.exists() or output.stat().st_size == 0:
                        start = None if variant == "full" else sample["clip_start_sec"]
                        cmd = variant_command(source, output, config, start, target_frames)
                        elapsed = encode_atomic(cmd, output, args.encode_timeout)
                    row.update(probe_output(output, config, args.encode_timeout))
                    row["output_bytes"] = output.stat().st_size
                    row["encode_seconds"] = elapsed
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                os.fsync(metrics.fileno())
                rows[key] = row
                status = "ok" if valid_row(row) else "FAILED"
                size_mb = row.get("output_bytes", 0) / 1e6
                print(
                    f"[{status}] sample={sample['sample_id']} variant={variant} "
                    f"frames={row.get('frame_count')} size={size_mb:.2f}MB"
                )
                summary = summarize(rows, variants, args.projection_videos)
                write_report(args.output_dir, summary)

    summary = summarize(rows, variants, args.projection_videos)
    write_report(args.output_dir, summary)
    print((args.output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
