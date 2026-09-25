from types import SimpleNamespace

from litgpt.byte import h264_syntax as HS
from scripts.hpc.zaratan.jpeglm_small_sample import eval_tf


def test_syntax_annotations_keep_straddling_bytes_separate(monkeypatch):
    spans = [
        HS.SyntaxSpan("slice_header", HS.Category.SLICE_HEADER, 0, 8, 10, 12),
        HS.SyntaxSpan("luma", HS.Category.RESIDUAL_LUMA, 8, 24, 11, 13),
    ]
    monkeypatch.setattr(
        eval_tf.HS,
        "parse_stream",
        lambda *_args, **_kwargs: SimpleNamespace(all_spans=lambda: spans),
    )
    sample = SimpleNamespace(gt_truncated_stream=b"", split=10, target_length=4)

    annotations = eval_tf.syntax_annotations(sample)

    assert [item["bucket"] for item in annotations] == [
        "structural_syntax",
        "mixed",
        "content_dependent",
        "unclassified",
    ]
    assert annotations[1]["owners"] == "slice_header, luma"


def test_syntax_bucket_ce_is_byte_weighted():
    rows = [
        {
            "target_bytes": 2,
            "byte_loss_bits_sum": 6.0,
            "byte_correct": 0,
            "eos_probability": 0.5,
            "eos_rank": 1,
            "syntax_buckets": {
                "structural_syntax": {"bytes": 1, "loss_bits_sum": 1.0},
                "content_dependent": {"bytes": 1, "loss_bits_sum": 5.0},
            },
        },
        {
            "target_bytes": 1,
            "byte_loss_bits_sum": 3.0,
            "byte_correct": 1,
            "eos_probability": 0.7,
            "eos_rank": 2,
            "syntax_buckets": {
                "structural_syntax": {"bytes": 1, "loss_bits_sum": 3.0},
            },
        },
    ]

    summary = eval_tf.aggregate(rows)

    assert summary["span_bits_per_byte"] == 3.0
    assert summary["by_syntax_bucket"]["structural_syntax"] == {
        "bytes": 2,
        "byte_fraction": 2 / 3,
        "ce_bits_per_byte": 2.0,
        "contribution_bits_per_target_byte": 4 / 3,
    }
    assert summary["by_syntax_bucket"]["content_dependent"]["ce_bits_per_byte"] == 5.0


def test_small_eval_selects_each_severity_with_its_own_size_limit(monkeypatch):
    seen = []

    def select(args):
        seen.append((args.corr_frame_type, args.corr_len_bytes_list, args.corr_eligibility_bytes))
        return SimpleNamespace(
            samples=[
                SimpleNamespace(
                    corruption_frame_type=args.corr_frame_type,
                    gap=args.corr_len_bytes_list[0],
                )
                for _ in range(args.corr_samples_per_length)
            ]
        )

    monkeypatch.setattr(eval_tf.FIM, "build_eval_sample_selection", select)
    values = {
        "MANIFEST": "/data/manifest.jsonl",
        "NAL_INDEX": "/data/nal_index.sqlite",
        "OUT_DIR": "/data/run",
        "MAX_ROWS": "256",
        "RAW_CONTEXT_BYTES": "131072",
        "WINDOW_MIN_FRAMES": "2",
        "WINDOW_UNIT": "gop",
        "VAL_FRACTION": "0.05",
        "FIM_FORMAT": "psm",
        "FIM_LOSS_SCOPE": "full",
        "FIM_MIN_GAP": "64",
        "FIM_MAX_GAP": "1400",
        "SLICE_HEADER_GUARD_BYTES": "0",
    }
    evaluation = {
        "corruption_lengths": [64, 128, 256, 400, 600],
        "samples_per_length": 5,
        "frame_types": ["idr", "p"],
        "seed": 42,
        "corruption_position": 0.4,
        "corruption_header_guard_bytes": 0,
    }

    samples = eval_tf.build_samples(values, evaluation, "val")

    assert len(samples) == 35
    assert seen == [
        ("idr", [64], 64),
        ("idr", [128], 128),
        ("idr", [256], 256),
        ("idr", [400], 400),
        ("idr", [600], 600),
        ("p", [64], 64),
        ("p", [128], 128),
    ]
    _, omitted = eval_tf.evaluation_strata(evaluation)
    assert [(item["frame_type"], item["corruption_length_bytes"]) for item in omitted] == [
        ("p", 256), ("p", 400), ("p", 600),
    ]
