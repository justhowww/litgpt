# Whole-Clip BSCV-Style Evaluation

This launcher runs `scripts/byte/eval/eval_bscv_clips.py`, which corrupts
multiple VCL NALs per GOP, decodes the whole H.264 stream, and compares:

- deleted-gap with strict FFmpeg decoding
- deleted-gap with FFmpeg default concealment
- model-filled spans with strict FFmpeg decoding
- model-filled spans with FFmpeg default concealment

Default launch:

```bash
STAGED_CORPUS="$HOME/scratch.metzler-prj/OpenVid-1M_Data/data" \
RUN_NAME=byte-stage2-fim-psm-ce-only-within-video-1k-8k \
CHECKPOINT_STEPS="5000 5500 6000 6500" \
NUM_CLIPS=20 \
bash scripts/hpc/zaratan/eval_bscv_clips/submit_h100.sh
```

Useful knobs:

- `CORR_PROB`: number of VCL frames corrupted per GOP, BSCV-style.
- `CORR_LEN_BYTES`: deleted bytes per selected frame. BSCV reports hex-char
  lengths, so BSCV `4096` corresponds to `2048` bytes here.
- `CORR_POS`: relative deletion position inside the selected VCL payload.
- `MAX_FRAMES`: maximum decoded frames per clip for metrics.
- `MAX_SPANS_PER_CLIP`: optional cap on generated repairs per clip. `0` keeps
  every BSCV-selected span.
- `TARGET_NAL_TYPES`: eligible VCL NAL types. Default is `1` to match current
  P-slice-only FIM training. Use `"1 5"` only after IDR/I-slice FIM is trained.

Current scope: the corruption/decode is whole-clip, but each model repair still
uses the current FIM prompt construction with local prefix/suffix and reference
conditioning. This is the right next diagnostic for temporal propagation, but it
is not yet a full deployed streaming repair loop.

Read `*_default_*` as the primary whole-clip metric. Strict decoding is retained
as a syntax diagnostic, but whole clips with multiple corruptions often fail
strictly after the first decoder error. Also check `*_frame_count_match_rate`
and `*_frame_coverage_mean`; PSNR/SSIM are only trustworthy when frame alignment
is effectively intact.
