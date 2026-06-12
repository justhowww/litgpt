import torch
import torch.nn.functional as F

from litgpt.data.byte_data import IGNORE_INDEX, SEQ_EOS_ID
from litgpt.pretrain import (
    balanced_eos_auxiliary_loss,
    byte_training_loss,
    byte_weighted_cross_entropy,
)


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


def test_balanced_eos_auxiliary_loss_normalizes_classes_separately():
    logits = torch.zeros(1, 4, SEQ_EOS_ID + 1)
    targets = torch.tensor([[1, 2, 3, SEQ_EOS_ID]])
    eos_probability = 1 / (SEQ_EOS_ID + 1)
    expected_positive = -torch.log(torch.tensor(eos_probability))
    expected_negative = -torch.log(torch.tensor(1 - eos_probability))

    actual = balanced_eos_auxiliary_loss(logits, targets)

    assert torch.allclose(
        actual, 0.5 * (expected_positive + expected_negative)
    )  # One EOS and three non-EOS positions receive equal aggregate class weight.


def test_balanced_eos_auxiliary_loss_penalizes_premature_eos():
    targets = torch.tensor([[7, SEQ_EOS_ID]])
    calibrated = torch.zeros(1, 2, SEQ_EOS_ID + 1)
    premature = calibrated.clone()
    premature[0, 0, SEQ_EOS_ID] = 10

    calibrated_loss = balanced_eos_auxiliary_loss(calibrated, targets)
    premature_loss = balanced_eos_auxiliary_loss(premature, targets)

    assert premature_loss > calibrated_loss  # High EOS probability before the endpoint is penalized.


def test_byte_training_loss_adds_balanced_eos_objective():
    logits = torch.randn(2, 4, SEQ_EOS_ID + 1)
    targets = torch.tensor(
        [[1, 2, 3, SEQ_EOS_ID], [4, 5, 6, SEQ_EOS_ID]]
    )
    byte_ce = byte_weighted_cross_entropy(logits, targets)
    eos_aux = balanced_eos_auxiliary_loss(logits, targets)

    actual = byte_training_loss(
        logits,
        targets,
        eos_aux_loss_weight=0.5,
    )

    assert torch.allclose(actual, byte_ce + 0.5 * eos_aux)  # The coefficient controls only EOS calibration.
