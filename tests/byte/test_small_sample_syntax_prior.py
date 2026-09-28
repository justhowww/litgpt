import math

import torch

from litgpt.byte.data import EOS_VOCAB_SIZE, SEQ_EOS_ID
from scripts.hpc.zaratan.jpeglm_small_sample.eval_syntax_prior import (
    aggregate,
    score_byte_legality,
)


def test_illegal_mass_uses_bytes_plus_eos_and_reports_eos_separately():
    logits = torch.full((2, EOS_VOCAB_SIZE), -100.0)
    # First position: byte 0 is legal; byte 1 is illegal and wins among bytes.
    logits[0, 0] = math.log(2)
    logits[0, 1] = math.log(3)
    logits[0, SEQ_EOS_ID] = math.log(5)
    logits[0, 256] = 100.0  # FIM control IDs cannot be sampled at inference.
    # Second position: byte 1 is legal and wins, with no illegal-byte mass.
    logits[1, 1] = 0.0
    masks = torch.zeros((2, 256), dtype=torch.bool)
    masks[0, 0] = True
    masks[1, 1] = True
    row = score_byte_legality(logits, masks, torch.tensor([True, False]))

    assert row["positions"] == 2
    assert row["strict_positions"] == 1
    assert math.isclose(row["illegal_mass_sum"], 0.3, rel_tol=1e-6)
    assert row["byte_top_illegal_count"] == 1
    assert row["full_top_illegal_byte_count"] == 0
    assert row["full_top_eos_count"] == 1
    summary = aggregate([row])
    assert math.isclose(summary["mean_illegal_byte_probability"], 0.15, rel_tol=1e-6)
    assert math.isclose(summary["mean_illegal_byte_probability_strict"], 0.3, rel_tol=1e-6)
    assert summary["byte_top_illegal_rate_strict"] == 1.0


def test_illegal_top_byte_is_counted_when_it_wins_over_bytes_and_eos():
    logits = torch.full((1, EOS_VOCAB_SIZE), -100.0)
    logits[0, 0] = 0.0
    logits[0, 1] = 1.0
    masks = torch.zeros((1, 256), dtype=torch.bool)
    masks[0, 0] = True
    row = score_byte_legality(logits, masks, torch.tensor([True]))

    assert row["byte_top_illegal_count"] == 1
    assert row["full_top_illegal_byte_count"] == 1
    assert row["full_top_eos_count"] == 0
