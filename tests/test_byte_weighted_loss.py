import torch
import torch.nn.functional as F

from litgpt.data.byte_data import IGNORE_INDEX, SEQ_EOS_ID
from litgpt.pretrain import byte_weighted_cross_entropy


def test_byte_weighted_cross_entropy_weights_only_positive_eos_targets():
    logits = torch.zeros(1, 3, SEQ_EOS_ID + 1)
    targets = torch.tensor([[7, SEQ_EOS_ID, IGNORE_INDEX]])

    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    expected = (losses[0] + 10 * losses[1]) / 2

    actual = byte_weighted_cross_entropy(logits, targets, eos_loss_weight=10)

    assert torch.allclose(
        actual, expected
    )  # EOS gets 10x weight while the ignored position stays excluded.


def test_byte_weighted_cross_entropy_matches_standard_ce_at_unit_weight():
    logits = torch.randn(2, 4, SEQ_EOS_ID + 1)
    targets = torch.tensor(
        [[1, 2, SEQ_EOS_ID, IGNORE_INDEX], [3, 4, 5, SEQ_EOS_ID]]
    )

    weighted = byte_weighted_cross_entropy(logits, targets, eos_loss_weight=1)
    standard = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )

    assert torch.allclose(
        weighted, standard
    )  # The control run remains numerically identical at weight 1.
