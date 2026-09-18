from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from litgpt.byte.data import (
    FIM_BEGIN_ID,
    FIM_END_ID,
    FIM_HOLE_ID,
    IGNORE_INDEX,
    REGION_BRIDGE,
    SEQ_EOS_ID,
    patch_byte_sample,
)
from scripts.byte.eval import eval_fim_avclm as FIM


def test_gt_automaton_preflight_runs_for_masked_learned_eos(monkeypatch) -> None:
    samples = [object(), object()]
    monkeypatch.setattr(FIM, "gt_parser_reconnects", lambda *args, **kwargs: True)

    result = FIM.validate_parser_reconnection(
        SimpleNamespace(mask_illegal_bytes=True, stop_modes=["learned_eos"]),
        samples,
        slice_max_mbs=None,
    )

    assert result == [True, True]


def test_gt_automaton_preflight_fails_closed_for_masked_eval(monkeypatch) -> None:
    samples = [object(), object()]
    results = iter((True, False))
    monkeypatch.setattr(
        FIM, "gt_parser_reconnects", lambda *args, **kwargs: next(results)
    )

    with pytest.raises(RuntimeError, match="only 1/2 samples passed"):
        FIM.validate_parser_reconnection(
            SimpleNamespace(mask_illegal_bytes=True, stop_modes=["learned_eos"]),
            samples,
            slice_max_mbs=None,
        )


def test_gt_automaton_preflight_skips_unmasked_learned_eos(monkeypatch) -> None:
    monkeypatch.setattr(
        FIM,
        "gt_parser_reconnects",
        lambda *args, **kwargs: pytest.fail("unexpected automaton preflight"),
    )

    result = FIM.validate_parser_reconnection(
        SimpleNamespace(mask_illegal_bytes=False, stop_modes=["learned_eos"]),
        [object()],
        slice_max_mbs=None,
    )

    assert result == [None]


def test_recorded_video_split_selects_disjoint_train_and_val(tmp_path: Path) -> None:
    train_path = str(tmp_path / "train.h264")
    val_path = str(tmp_path / "val.h264")
    dataset = SimpleNamespace(
        samples=[
            SimpleNamespace(h264_path=Path(train_path), start_nal=0, end_nal=10),
            SimpleNamespace(h264_path=Path(val_path), start_nal=0, end_nal=10),
        ]
    )
    split_file = tmp_path / "train_split.json"
    split_file.write_text(
        json.dumps(
            {
                "split_by_video": True,
                "videos": [train_path],
                "windows": [
                    {"h264_path": train_path, "start_nal": 0, "end_nal": 10}
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = SimpleNamespace(hole_placement="training_random")

    train_indices, _ = FIM._select_eval_windows(
        SimpleNamespace(train_split_file=split_file, eval_split="train"),
        dataset,
        policy,
    )
    val_indices, _ = FIM._select_eval_windows(
        SimpleNamespace(train_split_file=split_file, eval_split="val"),
        dataset,
        policy,
    )

    assert train_indices == [0]
    assert val_indices == [1]


def test_recorded_window_split_selects_exact_complement(tmp_path: Path) -> None:
    path = str(tmp_path / "shared.h264")
    dataset = SimpleNamespace(
        samples=[
            SimpleNamespace(h264_path=Path(path), start_nal=0, end_nal=10),
            SimpleNamespace(h264_path=Path(path), start_nal=10, end_nal=20),
        ]
    )
    split_file = tmp_path / "train_split.json"
    split_file.write_text(
        json.dumps(
            {
                "split_by_video": False,
                "videos": [path],
                "windows": [
                    {"h264_path": path, "start_nal": 0, "end_nal": 10}
                ],
            }
        ),
        encoding="utf-8",
    )

    val_indices, _ = FIM._select_eval_windows(
        SimpleNamespace(train_split_file=split_file, eval_split="val"),
        dataset,
        SimpleNamespace(hole_placement="training_random"),
    )

    assert val_indices == [1]


def _full_fim_item() -> tuple[dict, bytes, int, int]:
    window = bytes(range(20))
    split = 5
    gap = 3
    missing = list(window[split : split + gap])
    input_ids = torch.tensor(
        [
            FIM_BEGIN_ID,
            *window[:split],
            FIM_HOLE_ID,
            *window[split + gap :],
            FIM_END_ID,
            *missing,
        ],
        dtype=torch.long,
    )
    labels = torch.cat((input_ids[1:], torch.tensor([SEQ_EOS_ID])))
    regions = torch.full_like(input_ids, REGION_BRIDGE)
    offsets = torch.arange(input_ids.numel())
    return (
        {
            "input_ids": input_ids,
            "labels": labels,
            "region_ids": regions,
            "offset_ids": offsets,
            "token_counts": {
                "raw": input_ids.numel(),
                "raw_plus_prompt_template": input_ids.numel(),
            },
            "sample_meta": {
                "task": "fim",
                "h264_path": "synthetic.h264",
                "start_nal": 0,
                "end_nal": 1,
                "frame_lo": 0,
                "frame_hi": len(window),
                "fim_split": split,
                "fim_gap": gap,
            },
        },
        window,
        split,
        gap,
    )


def test_full_sequence_metric_keeps_native_patch_phase() -> None:
    item, window, split, gap = _full_fim_item()
    full = patch_byte_sample(item, patch_size=8)

    metric_mask = FIM._teacher_forced_span_mask(full["labels"], gap)
    assert metric_mask is not None
    assert full["labels"][metric_mask].tolist() == [
        *window[split : split + gap],
        SEQ_EOS_ID,
    ]

    span_item = {**item, "labels": torch.full_like(item["labels"], IGNORE_INDEX)}
    span_item["labels"][-(gap + 1) :] = item["labels"][-(gap + 1) :]
    repatched_span = patch_byte_sample(span_item, patch_size=8)

    # This is the original bug: both samples contain the same target tokens, but
    # starting supervision at the hole changes their global eight-byte grouping.
    assert not torch.equal(full["input_ids"], repatched_span["input_ids"])


def test_build_eval_sample_selection_uses_recorded_full_sequence_layout(
    monkeypatch, tmp_path: Path,
) -> None:
    item, window, split, gap = _full_fim_item()

    class FakeDataset:
        use_eos = True
        samples = [
            SimpleNamespace(
                h264_path=Path("synthetic.h264"), start_nal=0, end_nal=1
            )
        ]

        def __init__(self, *args, **kwargs) -> None:
            assert kwargs["fim_loss_scope"] == "full"
            assert kwargs["window_unit"] == "gop"

        def __getitem__(self, idx: int) -> dict:
            assert idx == 0
            return item

        def ar_item(self, idx: int) -> dict:
            assert idx == 0
            return {
                "labels": torch.tensor([*window, SEQ_EOS_ID], dtype=torch.long)
            }

    split_file = tmp_path / "train_split.json"
    split_file.write_text(
        '{"fim_loss_scope": "full", "window_unit": "gop", "windows": '
        '[{"h264_path": "synthetic.h264", "start_nal": 0, "end_nal": 1}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(FIM, "load_manifest_rows", lambda *args, **kwargs: [{}])
    monkeypatch.setattr(FIM, "ByteStreamWindowDataset", FakeDataset)
    monkeypatch.setattr(
        FIM,
        "default_nal_index_path",
        lambda manifest: tmp_path / "missing.sqlite",
    )

    args = SimpleNamespace(
        fim_loss_scope="auto",
        window_unit="auto",
        manifest=tmp_path / "manifest.jsonl",
        max_manifest_rows=0,
        nal_index_path=None,
        max_window_bytes=1024,
        window_min_frames=2,
        fim_format="psm",
        use_eos=True,
        fim_min_gap=1,
        fim_max_gap=16,
        slice_header_guard_bytes=0,
        seed=42,
        train_split_file=split_file,
        eval_split="all",
        split_by_video=False,
        val_fraction=0.05,
        num_clips=1,
    )

    selection = FIM.build_eval_sample_selection(args)
    assert args.fim_loss_scope == "auto"
    assert args.window_unit == "auto"
    assert selection.policy.fim_loss_scope == "full"
    assert selection.policy.window_unit == "gop"
    samples = selection.samples
    assert len(samples) == 1
    sample = samples[0]
    assert sample.target_bytes == window[split : split + gap]
    assert int(sample.prompt_ids[-1]) == FIM_END_ID
    assert torch.equal(sample.teacher_labels, item["labels"])
