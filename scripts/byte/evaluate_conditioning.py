"""Evaluate byte-model reference conditioning and a byte-unigram baseline."""

from __future__ import annotations

import argparse
import copy
import json
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from litgpt.config import Config
from litgpt.byte.data import (
    BYTE_VOCAB_SIZE,
    IGNORE_INDEX,
    REFERENCE_MODES,
    ByteSliceDataset,
    collate_byte_samples,
    load_manifest_rows,
)
from litgpt.model import GPT
from litgpt.byte.training import get_model_inputs_and_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-manifest-rows", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--unigram-train-samples", type=int, default=100000)
    parser.add_argument("--num-ref-slices", type=int, default=1)
    parser.add_argument("--target-nal-types", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(checkpoint_dir: Path, device: torch.device) -> GPT:
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    model = GPT(config)
    checkpoint = torch.load(
        checkpoint_dir / "lit_model.pth",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {_strip_compile_prefix(key): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _strip_compile_prefix(key: str) -> str:
    for prefix in ("_forward_module.", "_orig_mod.", "module."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def make_mode_dataset(dataset: ByteSliceDataset, mode: str) -> ByteSliceDataset:
    mode_dataset = copy.copy(dataset)
    mode_dataset.reference_mode = mode
    mode_dataset.shuffled_ref_indices = (
        mode_dataset._build_shuffled_ref_indices() if mode == "shuffled_ref" else {}
    )
    return mode_dataset


def make_loader(
    dataset: ByteSliceDataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    max_seq_length: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(collate_byte_samples, max_seq_length=max_seq_length),
    )


@torch.no_grad()
def evaluate_model(
    model: GPT, loader: DataLoader, device: torch.device, max_seq_length: int
) -> dict[str, float | int]:
    loss_sum = 0.0
    correct = 0
    token_count = 0
    use_amp = device.type == "cuda"

    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        model_inputs, targets = get_model_inputs_and_targets(batch, max_seq_length)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(**model_inputs)

        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = targets.reshape(-1)
        supervised = flat_targets != IGNORE_INDEX
        loss_sum += F.cross_entropy(
            flat_logits, flat_targets, ignore_index=IGNORE_INDEX, reduction="sum"
        ).item()
        correct += (flat_logits[supervised].argmax(dim=-1) == flat_targets[supervised]).sum().item()
        token_count += supervised.sum().item()

    loss = loss_sum / token_count
    return {
        "loss_nats_per_byte": loss,
        "bits_per_byte": loss / math.log(2),
        "top1_byte_accuracy": correct / token_count,
        "supervised_bytes": token_count,
    }


def fit_unigram(
    dataset: ByteSliceDataset, train_indices: list[int], max_samples: int
) -> torch.Tensor:
    selected = train_indices[:max_samples] if max_samples > 0 else train_indices
    counts = torch.ones(BYTE_VOCAB_SIZE, dtype=torch.float64)
    current_path = None
    current_data = b""
    for idx in sorted(selected, key=lambda i: (str(dataset.samples[i].h264_path), dataset.samples[i].target_index)):
        sample = dataset.samples[idx]
        if sample.h264_path != current_path:
            current_path = sample.h264_path
            current_data = sample.h264_path.read_bytes()
        nal = dataset.nal_index[str(sample.h264_path)][sample.target_index]
        target = torch.tensor(list(current_data[nal.start : nal.end]), dtype=torch.long)
        counts += torch.bincount(target, minlength=BYTE_VOCAB_SIZE)
    return counts / counts.sum()


def evaluate_unigram(
    probabilities: torch.Tensor, dataset: ByteSliceDataset, val_indices: list[int]
) -> dict[str, float | int]:
    loss_sum = 0.0
    correct = 0
    token_count = 0
    prediction = int(probabilities.argmax())
    log_probabilities = probabilities.log()

    for idx in val_indices:
        sample = dataset.samples[idx]
        data = sample.h264_path.read_bytes()
        nal = dataset.nal_index[str(sample.h264_path)][sample.target_index]
        target = torch.tensor(list(data[nal.start : nal.end]), dtype=torch.long)
        loss_sum -= log_probabilities[target].sum().item()
        correct += (target == prediction).sum().item()
        token_count += target.numel()

    loss = loss_sum / token_count
    return {
        "loss_nats_per_byte": loss,
        "bits_per_byte": loss / math.log(2),
        "top1_byte_accuracy": correct / token_count,
        "supervised_bytes": token_count,
    }


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = load_manifest_rows(
        args.manifest,
        max_rows=None if args.max_manifest_rows == 0 else args.max_manifest_rows,
    )
    base_dataset = ByteSliceDataset(
        rows,
        max_seq_length=args.max_seq_length,
        p_fim=0.0,
        num_ref_slices=args.num_ref_slices,
        target_nal_types=tuple(args.target_nal_types),
        reference_mode="normal",
        seed=args.seed,
    )
    val_size = max(1, int(len(base_dataset) * args.val_fraction))
    train_size = len(base_dataset) - val_size
    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_indices = list(train_subset.indices)
    val_indices = list(val_subset.indices)
    if args.max_val_samples > 0:
        val_indices = val_indices[: args.max_val_samples]

    model = load_model(args.checkpoint_dir, device)
    conditions = {}
    for mode in REFERENCE_MODES:
        print(f"Evaluating reference_mode={mode} on {len(val_indices)} samples")
        mode_dataset = make_mode_dataset(base_dataset, mode)
        loader = make_loader(
            mode_dataset,
            val_indices,
            args.batch_size,
            args.num_workers,
            args.max_seq_length,
        )
        conditions[mode] = evaluate_model(model, loader, device, args.max_seq_length)

    print("Fitting byte-unigram baseline")
    unigram = fit_unigram(base_dataset, train_indices, args.unigram_train_samples)
    unigram_metrics = evaluate_unigram(unigram, base_dataset, val_indices)
    normal_loss = conditions["normal"]["loss_nats_per_byte"]
    report = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "manifest": str(args.manifest),
        "num_validation_samples": len(val_indices),
        "validation_samples": [
            {
                "h264_path": str(base_dataset.samples[idx].h264_path),
                "target_index": base_dataset.samples[idx].target_index,
            }
            for idx in val_indices
        ],
        "num_unigram_train_samples": min(
            len(train_indices),
            args.unigram_train_samples if args.unigram_train_samples > 0 else len(train_indices),
        ),
        "conditions": conditions,
        "unigram": unigram_metrics,
        "loss_delta_vs_normal": {
            mode: metrics["loss_nats_per_byte"] - normal_loss
            for mode, metrics in conditions.items()
            if mode != "normal"
        },
    }
    output = args.output or args.checkpoint_dir / "conditioning_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
