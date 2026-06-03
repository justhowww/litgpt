"""Smoke-check the byte-domain H.264 dataset on a real processed corpus.

This is intentionally not a pytest unit test because it depends on a local
preprocessed corpus manifest.

Example:
    python scripts/debug_byte_data.py /path/to/corpus/manifest.jsonl --max-seq-length 32768
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from litgpt.data.byte_data import (
    IGNORE_INDEX,
    REGION_BRIDGE,
    REGION_META,
    REGION_ORPHAN,
    REGION_PREFIX,
    REGION_REF,
    REGION_TARGET,
    VOCAB_SIZE,
    ByteDataConfig,
    ByteDataModule,
)


REGION_NAMES = {
    REGION_REF: "ref",
    REGION_TARGET: "target",
    REGION_PREFIX: "prefix",
    REGION_ORPHAN: "orphan",
    REGION_BRIDGE: "bridge",
    REGION_META: "meta",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--p-fim", type=float, default=0.0)
    parser.add_argument("--num-ref-slices", type=int, default=1)
    parser.add_argument("--target-nal-types", type=int, nargs="+", default=[1])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fim-min-gap", type=int, default=64)
    parser.add_argument("--fim-max-gap", type=int, default=1400)
    parser.add_argument("--no-parameter-sets", action="store_true")
    return parser.parse_args()


def summarize_sample(sample: dict) -> None:
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    region_ids = sample["region_ids"]
    supervised = labels != IGNORE_INDEX
    region_counts = Counter(region_ids.tolist())
    named_region_counts = {REGION_NAMES.get(k, str(k)): v for k, v in sorted(region_counts.items())}

    print("sample_meta:", sample["sample_meta"])
    print("sample input shape:", tuple(input_ids.shape))
    print("sample labels shape:", tuple(labels.shape))
    print("sample supervised labels:", int(supervised.sum()))
    print("sample max token id:", int(input_ids.max()))
    print("sample region counts:", named_region_counts)

    assert input_ids.shape == labels.shape
    assert input_ids.shape == region_ids.shape
    assert int(input_ids.max()) < VOCAB_SIZE
    assert int(input_ids.min()) >= 0
    assert int(supervised.sum()) > 0


def summarize_batch(batch: dict) -> None:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    region_ids = batch["region_ids"]
    supervised = labels != IGNORE_INDEX

    print("batch input shape:", tuple(input_ids.shape))
    print("batch labels shape:", tuple(labels.shape))
    print("batch region shape:", tuple(region_ids.shape))
    print("batch supervised labels:", int(supervised.sum()))
    print("batch max token id:", int(input_ids.max()))

    assert input_ids.shape == labels.shape
    assert input_ids.shape == region_ids.shape
    assert int(input_ids.max()) < VOCAB_SIZE
    assert int(input_ids.min()) >= 0
    assert int(supervised.sum()) > 0


def main() -> None:
    args = parse_args()
    config = ByteDataConfig(
        p_fim=args.p_fim,
        num_ref_slices=args.num_ref_slices,
        target_nal_types=tuple(args.target_nal_types),
        num_workers=args.num_workers,
        fim_min_gap=args.fim_min_gap,
        fim_max_gap=args.fim_max_gap,
        include_parameter_sets=not args.no_parameter_sets,
        default_max_seq_length=args.max_seq_length,
    )
    dm = ByteDataModule(manifest_path=args.manifest, config=config)
    dm.connect(batch_size=args.batch_size, max_seq_length=args.max_seq_length)
    dm.setup()

    train_len = len(dm.train_dataset) if dm.train_dataset is not None else 0
    val_len = len(dm.val_dataset) if dm.val_dataset is not None else 0
    print("train samples:", train_len)
    print("val samples:", val_len)

    assert train_len > 0
    assert val_len > 0

    sample = dm.train_dataset[0]
    summarize_sample(sample)

    batch = next(iter(dm.train_dataloader()))
    summarize_batch(batch)

    print("byte dataset smoke check passed")


if __name__ == "__main__":
    main()
