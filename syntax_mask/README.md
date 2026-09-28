# syntax_mask — H.264 next-byte syntax mask from precompiled field transitions

Stand-alone rebuild (stdlib only, no torch/litgpt imports) of the legal-next-byte
mask. It follows *efficient syntax mask generation plan.md* and
*h264 syntax field.md*. The old mask tests every bit on a cloned
automaton. This one walks **syntax fields** and looks up **precompiled codeword
transition tables**.

```
bytes ──► Step 3 stream.py   Annex-B start codes + RBSP→EBSP filter (O(1) bit ops)
             │ RBSP byte mask
             ▼
          Step 2 compile.py  align fields to 8 bits; OR partial chunks, recurse on completions
             │ (code, code_state, bits_left) lookups
             ▼
          Step 1b codes.py   precompiled per-code step tables  (partial / completions)
          Step 1  grammar.py next-field oracle: _expect(field, code+legal set, handler)
```

* **Step 1: oracle** (`grammar.py`). The parser state is always positioned at
  (or inside) one physical field. `_expect(name, code, handler)` returns the
  field, its code with the legal-value set folded in, and the handler. The
  handler is the value-dependent transition: an O(1) state update plus the choice
  of the next field, with absent fields collapsed into it. Fields whose future
  does not depend on the value are marked `vindep` (mvd, signs, PCM/SEI bytes).
  Fields where many values share a future carry a `vclass` (Intra16x16 mb_types
  with equal CBP, level codes with equal suffixLength). Covers the NAL header,
  SPS+VUI+HRD, PPS, SEI framing, AUD, filler, end-of-seq/stream, the full slice
  header (POC, list modification, MMCO) and CAVLC slice data. Slice data tracks
  per-slice nC, Intra4x4/16x16/chroma prediction-mode availability,
  constrained_intra_pred, and DPB-bounded `ref_idx`.
* **Step 1b: tables** (`codes.py`). Each code (`u(n)`, `ue/se/me/te`, CAVLC VLC,
  unary `level_prefix`) is a small DFA. `Code.step(cs, r)` gives, for the `r`
  bits left in the byte:
  * `partial`: the chunks that leave the field open;
  * `comps`: the `(value, length, chunk)` codewords that finish inside the byte.
  
  Codes are interned by (kind, width, legal set) and their step tables are
  memoized, so after warm-up they act as precompiled tables.
* **Step 2: alignment** (`compile.py::solve`). This is a recursion over field
  completions within the byte. Completions that share a future (`vindep` or
  `vclass`) are solved once and spread over all their chunks.
  `reference_mask` is the naive bit-by-bit definition, used for exactness tests.
* **Step 3: byte stream** (`stream.py::MaskStream`).
  * Start code: `00` until two zeros, then `00|01`.
  * Payload, when `zero_run >= 2`: `00..02` are removed and `03` (an EPB) is
    allowed iff an RBSP byte `00..03` is legal.
  * After an EPB: only `00..03`.

## Contract

* **`false` = illegal.** A masked-out byte cannot continue a conforming
  Baseline stream under the project profile below.
* **`true` = no objection.** Where a rule needs information the oracle does not
  model, the field is left permissive:
  * MMCO/long-term argument validity;
  * POC arithmetic;
  * motion vectors inside the picture/level MV range (mvd only bounded to ±8192);
  * HRD/bitrate/timing level limits;
  * reserved VUI values;
  * the SEI payload contents.

Project-profile restrictions (stricter than H.264 Baseline, match our x264 corpora):

| Rule | Why |
|---|---|
| `profile_idc = 66`, CAVLC, `frame_mbs_only = 1`, `num_slice_groups_minus1 = 0`, weighted pred off | Baseline scope, no FMO |
| slice layout `frame` (one slice per picture, `first_mb = 0`) or `mb` (one MB per slice, in order) | `--layout`; `more_rbsp_data` is otherwise ambiguous at MB ends |
| no SPS/PPS/SEI/AUD inside an open picture (mb layout) | x264 only emits parameter sets between pictures |
| SEI `payloadType` 128 cannot follow the first SEI message | `0x80` there is read as rbsp_trailing_bits (no look-ahead) |
| After an SPS changes its core content, the next picture must be IDR; non-IDR slices keep the active SPS | 7.4.1.2.1 |

Spec rules this enforces that the legacy mask did not include:

* all non-VCL syntax;
* level limits on picture size/DPB;
* `level_prefix ≤ 15`;
* intra prediction modes only where the needed neighbours are available (this
  matters a lot in 1-MB slices, where only DC is legal);
* frame_num / idr_pic_id / cross-slice picture consistency;
* `ref_idx_l0 <` available references.

## Usage

Run from the repo root (`litgpt/`); only Python ≥ 3.10 is needed.

```bash
# correctness + speed on GT files (dirs are searched for *.h264)
python -m syntax_mask.scan ../data/part1 --workers 32 --layout frame \
    --reference-rate 0.01 --probe-rate 0.005 --json scan.json
# AVC-LM style corpora (one macroblock per slice)
python -m syntax_mask.scan /path/to/avclm --layout mb --workers 32
# write the training table: <name>.masks, 32 bytes per input byte
python -m syntax_mask.scan ../data/part1 --workers 32 --dump-dir masks/
# differential comparison with litgpt/byte/h264_mask (slow: legacy speed)
python -m syntax_mask.compare_legacy FILE.h264 --max-bytes 6000
# tests (pytest also works)
python -m syntax_mask.tests
```

`scan` exits non-zero on any GT rejection, reference mismatch, or probe dead end.
Checks per file:

* every GT byte is legal;
* sampled positions match the bit-by-bit reference exactly;
* sampled random legal walks of `--probe-len` bytes (off the GT path) never
  reach an empty mask, i.e. no dead ends.

`.masks` format: position `i` holds `mask.to_bytes(32, "little")`, so byte value
`b` is legal iff `(buf[32*i + b//8] >> (b%8)) & 1`.

API:

```python
from syntax_mask.stream import MaskStream, iter_masks
from syntax_mask.grammar import Profile
ms = MaskStream(Profile(slice_layout="frame"))
m = ms.mask()          # int, bit b set iff byte b is legal next
ms.advance(byte)       # raises grammar.Illegal on a violation
```

## Validation so far (local, CPython 3.11, Apple M-series)

Final code (after all fixes):

| Set | Files / bytes | GT accepted | reference parity checks / mismatches | probes / dead ends |
|---|---|---|---|---|
| `data/part1` random 150 (JPEG-LM style, 256×144, ref 3, QP 37) | 150 / 2.88 MB | 150/150 | 28,715 / 0 | 14,269 / 0 |
| synthetic x264, frame layout: default QP 28, QP 4, constrained-intra, mandelbrot 320×240 with AUD, 2 repo fixtures | 7 / 3.90 MB | 7/7 | 39,040 / 0 | 19,325 / 0 |
| synthetic AVC-LM 1 MB/slice (± constrained-intra) | 2 / 195 KB | 2/2 | 3,905 / 0 | 3,836 / 0 |

An earlier 200-file part1 run, before the DPB/SPS fixes, also accepted 200/200
(3.81 MB). Earlier probe runs found three dead ends, all now fixed:

* an SPS resent mid-picture;
* a P slice with no PPS on the active SPS;
* SPS dimensions incompatible with `max_num_ref_frames`.

Speed: about 170–270 µs/byte in a single process, against about 1,750–1,900
µs/byte for the legacy compiler on the same data, i.e. about 7× faster with
many more constraints. The masks allow 225 legal bytes on average in part1
(`mean_legal_bytes`); per-field numbers are printed by `scan`.

Where the new mask is looser than legacy, it is by design. Legacy forces
`adaptive_ref_pic_marking_mode_flag = 0` and rejects `num_ref_idx_active` above
the decoded reference count; both are legal H.264. The real constraint, a
`ref_idx` bounded by the available references, is enforced here.

## Performance notes for the big run

* Pure Python with no dependencies: **run it under PyPy** for a large constant-factor
  gain. The scan is embarrassingly parallel per file (`--workers`).
* Most of the remaining cost is about 100 clone+handler calls per byte in
  slice data. The next step would be memoizing `solve` on a compact
  per-field state key, or a C/numba port of `grammar.py`'s slice-data handlers.

## Whole corpus on Zaratan

`scripts/hpc/zaratan/syntax_mask/submit_scan.sh` submits a Slurm array (one
shard per task, `--resume` so reruns continue) plus a merge job:

```bash
PILOT=1 bash scripts/hpc/zaratan/syntax_mask/submit_scan.sh            # 2000 files, measure speed
NUM_SHARDS=32 bash scripts/hpc/zaratan/syntax_mask/submit_scan.sh      # full manifest
```

Per-file results go to `RUN_DIR/shard_*.jsonl`; `python -m syntax_mask.merge
RUN_DIR/shard_*.jsonl` (run automatically) writes `summary.json` / `summary.txt`.
