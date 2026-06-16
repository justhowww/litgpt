# BSCV corruption vs. our `eval_bscv_clips.py`

`bscv_corrupt_gen.py` is the **original BSCV** `corrupt_Gen.py` (LIU TIANYI, 2023-02-07),
vendored verbatim as a reference fixture. It is not executed — it walks a
`./{train,test}/GT_h264/` tree we don't have. It exists so our corruption
(`scripts/byte/eval/eval_bscv_clips.py`) can be compared to it line-by-line.

Our equivalents:
- selection  → `select_bscv_spans` + `eligible_payload_space`
- excision   → `delete_spans` (the `deleted_*` arms)
- decode     → `decode_stream_frames`

## TL;DR

**The decode path is identical** — our `deleted_default` arm and BSCV both decode the
corrupted `.h264` with stock ffmpeg defaults: error-concealment ON, no `err_detect`,
never fails on a bad byte. So the claim "our `deleted_default` == theirs" holds **for
the decoder**. The differences are entirely in *which bytes get removed*, not in how
the result is decoded. Three of those differences matter; one is the dominant one.

## Decode path — SAME ✓

| | BSCV | ours (`deleted_default`) |
|---|---|---|
| command | `ffmpeg -i BSC.h264 ... %05d.jpg` | `ffmpeg -f h264 -i pipe:0 ... ppm` |
| error concealment | default = **ON** (`guess_mvs+deblock`) | no `-ec` flag → default = **ON** |
| `err_detect` | default = **0** (never explode) | not set → default = **0** |
| fail on bad byte | no — conceals & continues | no — conceals & continues |
| output codec | JPEG `-qscale:v 2` (near-lossless) | PPM (lossless) |

Only cosmetic gaps: we emit PPM (lossless) instead of qscale-2 JPEG, add `-vsync 0` and
`-frames:v`. None change concealment behavior; PPM is actually the cleaner metric source.
Our **`strict`** arm (`-ec 0 -err_detect explode+…`) has **no BSCV analogue** — BSCV
never runs a fail-on-error decode, so don't compare the `strict` columns to BSCV.

## Corruption selection — three divergences

### 1. IDR eligibility — THE dominant difference
- **BSCV**: `frameIndexes = I_index(0x65) + P_index(0x41) + B_index(0x01)`, and
  `random.sample` draws from the whole GOP **including the IDR**. So the IDR keyframe
  (nal type 5) is a victim with probability ≈ `corr_prob / |GOP|` (~6% at prob=1,
  GOP≈16; ~12% at prob=2). Note `0x41` and `0x01` are *both* nal_unit_type 1.
- **ours**: `--target-nal-types 1` → only non-IDR slices (covers both `0x41` and `0x01`),
  **type-5 IDR is never selected.**
- **Effect**: BSCV's degraded set has a ~6–12% **catastrophic tail** (corrupt keyframe →
  no clean reference for the GOP → whole-GOP smear that propagates to the next IDR). We
  truncate that tail entirely, which is why our floor is both higher *and* tighter
  (idr 1.0 / near .993 / mid .966 / far .940, low variance). **To match BSCV, add 5:
  `--target-nal-types 1 5`** and report tail / worst-decile, not just the mean.

### 2. `corr_len` units — bytes vs hex chars
- **BSCV** works in `binascii.hexlify` space: the removed window is `corr_len` **hex
  chars = `corr_len / 2` bytes**. So their `_142048` removes **1024 bytes**, `_144096`
  removes **2048 bytes**.
- **ours**: `--corr-len-bytes` is bytes (arg help already notes "divide their value by
  two"). Our runs used `2048 bytes` = **4096 hex** = BSCV's **`_144096`** length —
  *not* `_142048`. (Earlier I mislabeled our leg as `_142048`; corrected: with
  pos 0.4 + 2048 B it is BSCV's `_144096`.) To reproduce `_142048` exactly, set
  `--corr-len-bytes 1024`.

### 3. Small-frame fallback
- **BSCV**: if every frame in the GOP is shorter than `corr_len`, it deletes
  `x+9 … y+1` — **through the next start code**, destroying NAL framing for following
  frames (extra destructiveness on tiny GOPs).
- **ours**: NALs with `eligible_payload_space < corr_len_bytes` are **skipped**
  (no corruption). Divergence only on undersized frames; we never break framing.

## Things that already match
- **Per-GOP count**: both `random.sample(GOP_frames, corr_prob)`; `corr_prob` is a
  per-GOP *count*, not a probability. Our `random` mode == theirs (our `early`/`late`
  modes are additions).
- **Excision, not replacement**: both *delete* bytes and shorten the stream
  (`remove_ranges` ≈ `delete_spans`). BSCV's odd-length parity guard keeps hex
  byte-aligned; we operate in bytes so always aligned — same outcome.
- **Placement**: BSCV `start = x + 7 + es·corr_pos`, `es = (y−x) − 7 − corr_len`;
  ours `start = payload_start + (nal.end − corr_len − payload_start)·corr_pos`. Same
  "fractional position into the payload" arithmetic, modulo: BSCV's `x+7` (3.5 bytes)
  can clip the 2nd nibble of the nal-header byte, while our `payload_start` sits one
  byte later. Both preserve the start code.

## To make `deleted_default` reproduce a BSCV config exactly
1. `--target-nal-types 1 5`  (let the IDR be hit at the BSCV rate)  ← biggest lever
2. `--corr-len-bytes 2048` → `_144096`, or `1024` → `_142048`
3. (optional) replicate the small-frame fallback instead of skipping
4. keep `deleted_default` (already ec-on / no-explode)
5. report the **distribution** (median + worst-decile), since the BSCV/ours gap lives in
   the keyframe tail, not the mean.
