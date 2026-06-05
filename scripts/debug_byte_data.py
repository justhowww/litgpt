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
from torch.utils.data import Subset

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
    ByteSliceDataset,
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
    parser.add_argument("--no-sps-pps-metadata", action="store_true")
    parser.add_argument("--block-size-candidates", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--block-size-samples", type=int, default=20000)
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


def summarize_block_size_candidates(dataset: ByteSliceDataset, indices: list[int], candidates: list[int]) -> None:
    if not indices:
        return

    lengths: list[int] = []
    targets: list[int] = []
    refs: list[int] = []
    metas: list[int] = []
    candidate_stats = {
        block_size: {"target_too_large": 0, "full_context": 0, "dropped_ref": 0}
        for block_size in candidates
    }

    for idx in indices:
        sample = dataset.samples[idx]
        nals = dataset.nal_index[str(sample.h264_path)]
        target_len = nals[sample.target_index].end - nals[sample.target_index].start
        meta_len = sum(nals[i].end - nals[i].start for i in sample.meta_indices)
        ref_lens = [nals[i].end - nals[i].start for i in sample.ref_indices]
        ref_len = sum(ref_lens)
        ideal_len = meta_len + ref_len + target_len

        lengths.append(ideal_len)
        targets.append(target_len)
        refs.append(ref_len)
        metas.append(meta_len)

        for block_size in candidates:
            stats = candidate_stats[block_size]
            if target_len > block_size:
                stats["target_too_large"] += 1
                continue
            if ideal_len <= block_size:
                stats["full_context"] += 1
                continue

            budget = block_size - target_len
            if meta_len >= budget:
                stats["dropped_ref"] += int(bool(ref_lens))
                continue

            remaining = budget - meta_len
            used = 0
            kept_refs = 0
            for ref_len_i in reversed(ref_lens):
                if used + ref_len_i <= remaining:
                    used += ref_len_i
                    kept_refs += 1
            stats["dropped_ref"] += len(ref_lens) - kept_refs

    print("block size analysis samples:", len(indices))
    print("ideal input length quantiles:", summarize_quantiles(lengths))
    print("target length quantiles:", summarize_quantiles(targets))
    print("ref length quantiles:", summarize_quantiles(refs))
    print("metadata length quantiles:", summarize_quantiles(metas))
    for block_size in candidates:
        stats = candidate_stats[block_size]
        n = len(indices)
        print(
            f"block_size={block_size}: "
            f"full_context={stats['full_context'] / n:.2%}, "
            f"target_too_large={stats['target_too_large'] / n:.2%}, "
            f"avg_dropped_ref_slices={stats['dropped_ref'] / n:.3f}"
        )


def summarize_quantiles(values: list[int]) -> dict[str, int]:
    values = sorted(values)
    if not values:
        return {}
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
    }


def percentile(values: list[int], q: float) -> int:
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[idx]


def get_indexed_dataset_and_indices(dataset) -> tuple[ByteSliceDataset, list[int]]:
    if isinstance(dataset, Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))


def main() -> None:
    args = parse_args()
    config = ByteDataConfig(
        p_fim=args.p_fim,
        num_ref_slices=args.num_ref_slices,
        target_nal_types=tuple(args.target_nal_types),
        num_workers=args.num_workers,
        fim_min_gap=args.fim_min_gap,
        fim_max_gap=args.fim_max_gap,
        condition_on_sps_pps=not args.no_sps_pps_metadata,
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

    indexed_dataset, train_indices = get_indexed_dataset_and_indices(dm.train_dataset)
    summarize_block_size_candidates(
        indexed_dataset,
        train_indices[: args.block_size_samples],
        sorted(set(args.block_size_candidates)),
    )

    batch = next(iter(dm.train_dataloader()))
    summarize_batch(batch)

    print("byte dataset smoke check passed")


if __name__ == "__main__":
    main()
