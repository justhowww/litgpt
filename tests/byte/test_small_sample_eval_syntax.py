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
