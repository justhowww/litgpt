import json
import math

import torch

from litgpt.data.byte_data import ByteSliceDataset
from litgpt.eval.byte_reconstruction import (
    ReconstructionSample,
    image_psnr,
    image_ssim,
    parse_ppm,
    replace_target_nal,
    select_reconstruction_samples,
)


def nal(header: int, payload: bytes) -> bytes:
    return b"\x00\x00\x00\x01" + bytes([header]) + payload


def make_dataset(tmp_path) -> ByteSliceDataset:
    stream = b"".join(
        [
            nal(0x67, b"sps"),
            nal(0x68, b"pps"),
            nal(0x65, b"I" * 12),
            nal(0x41, bytes(range(32, 72))),
            nal(0x41, bytes(range(72, 112))),
        ]
    )
    h264_path = tmp_path / "h264" / "clip.h264"
    h264_path.parent.mkdir()
    h264_path.write_bytes(stream)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"status": "ok", "h264_path": str(h264_path)}) + "\n"
    )
    return ByteSliceDataset(
        [{"status": "ok", "h264_path": str(h264_path)}],
        max_seq_length=256,
        p_fim=0.0,
        num_ref_slices=1,
    )


def test_select_reconstruction_samples_uses_ar_prompt_and_vcl_frame_index(tmp_path):
    dataset = make_dataset(tmp_path)

    samples = select_reconstruction_samples(dataset, num_samples=1, max_target_bytes=256)

    assert len(samples) == 1  # The deterministic probe selects the requested count.
    assert samples[0].frame_index == 1  # The first eligible P slice is decoded as frame 1.
    assert samples[0].prompt_ids[-1].item() == 257  # The AR prompt ends with SLICE_BOS.
    assert samples[0].target_length > 0  # Ground-truth length defines the controlled generation budget.


def test_replace_target_nal_preserves_surrounding_stream_bytes(tmp_path):
    path = tmp_path / "stream.h264"
    path.write_bytes(b"prefix-original-suffix")
    sample = ReconstructionSample(
        h264_path=path,
        target_start=7,
        target_end=15,
        target_nal_index=3,
        frame_index=0,
        prompt_ids=torch.tensor([1]),
        prompt_region_ids=torch.tensor([1]),
        prompt_offset_ids=torch.tensor([0]),
        target_length=3,
    )

    rebuilt = replace_target_nal(path.read_bytes(), sample, b"new")

    assert rebuilt == b"prefix-new-suffix"  # Only the selected target NAL range is replaced.


def test_parse_ppm_and_identical_image_metrics():
    ppm = b"P6\n2 1\n255\n" + bytes([0, 10, 20, 30, 40, 50])

    image = parse_ppm(ppm)

    assert image.shape == (1, 2, 3)  # PPM dimensions become HWC RGB.
    assert math.isinf(image_psnr(image, image))  # Identical frames have infinite PSNR.
    assert abs(image_ssim(image, image) - 1.0) < 1e-6  # Identical frames have perfect SSIM.
