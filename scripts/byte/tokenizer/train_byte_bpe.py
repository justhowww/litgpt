#!/usr/bin/env python
"""Derive a small byte-level BPE tokenizer for H.264 Annex-B streams (AVC-LM recipe).

Vocab = 256 raw bytes + reserved special tokens + BPE merges, totalling
``--vocab-size`` (default 1024, matching AVC-LM's "10K videos -> 1024 BPE tokens").
The tokenizer is lossless over arbitrary bytes (all 256 bytes are base tokens).

Outputs to ``--out-dir``:
  * tokenizer.json         -- loadable via tokenizers.Tokenizer.from_file
  * tokenizer_report.json  -- vocab breakdown, compression ratio, tokens/window,
                              start-code handling, roundtrip-losslessness, top merges

By default the corpus is split on the NAL start code (00 00 01) before training so
no merge spans a NAL boundary; the start code is carried as the <nal_start> special
at encode time. Pass --no-reserve-start-code to let BPE learn it instead.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

try:
    from tokenizers import Tokenizer, decoders, models, trainers
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script needs the `tokenizers` package (pip install tokenizers)."
    ) from exc

START_CODE = b"\x00\x00\x01"
# Reserved specials for AR + PSM-FIM (see the design discussion). BOS is the
# stream-start anchor; the three FIM_* are the PSM sentinels; <nal_start> is the
# optional NAL-boundary token (kept unless --no-reserve-start-code).
BASE_SPECIALS = ["<pad>", "<bos>", "<eos>", "<fim_begin>", "<fim_hole>", "<fim_end>"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos-dir", type=Path, help="Directory searched recursively for *.h264 / *.264.")
    src.add_argument("--files", type=Path, nargs="+", help="Explicit list of .h264 files.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-videos", type=int, default=10000, help="Sample this many streams for training.")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-segment-bytes", type=int, default=1024,
                   help="Truncate each NAL segment to this many bytes for BPE TRAINING (0 = no cap). "
                        "Bounds peak memory (HF holds every word's symbols); merges come from frequent "
                        "local patterns, so a prefix suffices. Does not affect the report/encoding.")
    p.add_argument("--reserve-start-code", action="store_true", default=True,
                   help="Split corpus on 00 00 01 and reserve <nal_start> (default).")
    p.add_argument("--no-reserve-start-code", dest="reserve_start_code", action="store_false")
    p.add_argument("--report-sample", type=int, default=500, help="Files used for the compression/roundtrip report.")
    p.add_argument("--window-bytes", type=int, default=16384, help="Byte window used to derive tokens-per-window.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = [Path(f) for ext in ("*.h264", "*.264")
                 for f in glob.glob(str(args.videos_dir / "**" / ext), recursive=True)]
    files = [f for f in files if f.is_file()]
    if not files:
        raise SystemExit("No .h264 / .264 files found.")
    random.Random(args.seed).shuffle(files)
    return files[: args.num_videos]


def byte_segments(data: bytes, reserve_start_code: bool) -> list[bytes]:
    """Split on the start code when reserving <nal_start>, else the whole stream."""
    if not reserve_start_code:
        return [data] if data else []
    return [seg for seg in data.split(START_CODE) if seg]


def corpus_iter(files: list[Path], reserve_start_code: bool, max_segment_bytes: int):
    """Yield latin-1 strings (each char == one raw byte -> lossless base alphabet).

    Segments are truncated to ``max_segment_bytes`` to bound trainer memory; this
    only affects which patterns BPE *sees*, not encoding/decoding losslessness.
    """
    for path in files:
        for seg in byte_segments(path.read_bytes(), reserve_start_code):
            if max_segment_bytes:
                seg = seg[:max_segment_bytes]
            if seg:
                yield seg.decode("latin-1")


def encode_ids(tok: Tokenizer, data: bytes, reserve_start_code: bool, nal_start_id: int | None) -> list[int]:
    """Encode a raw byte stream the way the dataset will: BPE the between-start-code
    segments, interleaving the <nal_start> token at each boundary when reserved."""
    if not reserve_start_code:
        return tok.encode(data.decode("latin-1"), add_special_tokens=False).ids
    ids: list[int] = []
    # data.split keeps the structure: [seg0, seg1, ...] were separated by start codes.
    parts = data.split(START_CODE)
    for i, seg in enumerate(parts):
        if i > 0 and nal_start_id is not None:
            ids.append(nal_start_id)
        if seg:
            ids.extend(tok.encode(seg.decode("latin-1"), add_special_tokens=False).ids)
    return ids


def detokenize(tok: Tokenizer, ids: list[int], reserve_start_code: bool, nal_start_id: int | None) -> bytes:
    id_to_tok = {v: k for k, v in tok.get_vocab().items()}
    out = bytearray()
    for i in ids:
        if reserve_start_code and i == nal_start_id:
            out += START_CODE
            continue
        out += id_to_tok[i].encode("latin-1")
    return bytes(out)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_files(args)
    print(f"Training on {len(files)} streams (reserve_start_code={args.reserve_start_code})", flush=True)

    specials = list(BASE_SPECIALS)
    if args.reserve_start_code:
        specials.append("<nal_start>")

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=specials,
        initial_alphabet=[chr(i) for i in range(256)],  # every byte is a base token
        min_frequency=args.min_frequency,
        show_progress=True,
    )
    tok.train_from_iterator(
        corpus_iter(files, args.reserve_start_code, args.max_segment_bytes),
        trainer=trainer,
    )

    tokenizer_path = args.out_dir / "tokenizer.json"
    tok.save(str(tokenizer_path))
    vocab = tok.get_vocab()
    total = len(vocab)
    nal_start_id = vocab.get("<nal_start>")
    n_merges = total - 256 - len(specials)

    # ---- report over a held-out sample ----
    rng = random.Random(args.seed + 1)
    sample = files if len(files) <= args.report_sample else rng.sample(files, args.report_sample)
    total_bytes = total_tokens = 0
    tokens_per_window: list[int] = []
    roundtrip_ok = True
    for path in sample:
        data = path.read_bytes()
        ids = encode_ids(tok, data, args.reserve_start_code, nal_start_id)
        total_bytes += len(data)
        total_tokens += len(ids)
        if detokenize(tok, ids, args.reserve_start_code, nal_start_id) != data:
            roundtrip_ok = False
        # tokens for the first window_bytes of the stream (proxy for block_size sizing)
        win_ids = encode_ids(tok, data[: args.window_bytes], args.reserve_start_code, nal_start_id)
        tokens_per_window.append(len(win_ids))

    bytes_per_token = total_bytes / max(1, total_tokens)
    # byte-length of every non-special token (base bytes have length 1)
    tok_byte_len = Counter(len(t.encode("latin-1")) for t in vocab if t not in specials)
    top_merges = sorted(
        ((t.encode("latin-1").hex(), len(t.encode("latin-1"))) for t in vocab
         if t not in specials and len(t.encode("latin-1")) > 1),
        key=lambda x: -x[1],
    )[:20]

    def pct(v, q):
        s = sorted(v)
        return s[min(len(s) - 1, int(q * len(s)))] if s else None

    report = {
        "tokenizer_path": str(tokenizer_path),
        "vocab": {"total": total, "bytes": 256, "specials": specials, "n_specials": len(specials), "merges": n_merges},
        "compression": {
            "bytes_per_token_mean": round(bytes_per_token, 3),
            "report_files": len(sample),
            "total_bytes": total_bytes,
            "total_tokens": total_tokens,
        },
        "tokens_per_window": {
            "window_bytes": args.window_bytes,
            "mean": round(sum(tokens_per_window) / max(1, len(tokens_per_window)), 1),
            "p50": pct(tokens_per_window, 0.50),
            "p90": pct(tokens_per_window, 0.90),
            "max": max(tokens_per_window) if tokens_per_window else None,
        },
        "start_code": (
            {"reserved": True, "token": "<nal_start>", "id": nal_start_id}
            if args.reserve_start_code else
            {"reserved": False, "encoded_ids": tok.encode(START_CODE.decode("latin-1"), add_special_tokens=False).ids}
        ),
        "roundtrip_lossless": roundtrip_ok,
        "token_byte_length_hist": dict(sorted(tok_byte_len.items())),
        "top_merges_hex": top_merges,
    }
    (args.out_dir / "tokenizer_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not roundtrip_ok:
        print("WARNING: roundtrip NOT lossless -- inspect latin-1/decoder handling before using.", flush=True)


if __name__ == "__main__":
    main()
