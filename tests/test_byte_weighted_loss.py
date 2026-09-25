import pytest
import torch
import torch.nn.functional as F

from litgpt.byte.data import REGION_BRIDGE, REGION_PREFIX
from litgpt.byte.training import byte_training_loss_terms
from litgpt.data.byte_data import IGNORE_INDEX, SEQ_EOS_ID
from litgpt.pretrain import (
    balanced_eos_auxiliary_loss,
    byte_training_loss,
    byte_weighted_cross_entropy,
    fim_span_byte_cross_entropy,
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


def test_fim_span_loss_selects_bridge_bytes_and_excludes_eos():
    logits = torch.randn(1, 4, SEQ_EOS_ID + 1)
    targets = torch.tensor([[11, 12, SEQ_EOS_ID, 13]])
    regions = torch.tensor(
        [[REGION_PREFIX, REGION_BRIDGE, REGION_BRIDGE, REGION_BRIDGE]]
    )

    expected = F.cross_entropy(logits[0, [1, 3]], targets[0, [1, 3]])
    actual = fim_span_byte_cross_entropy(logits, targets, regions)

    assert torch.allclose(actual, expected)


def test_byte_training_loss_adds_normalized_fim_span_objective():
    logits = torch.randn(1, 4, SEQ_EOS_ID + 1)
    targets = torch.tensor([[11, 12, SEQ_EOS_ID, 13]])
    regions = torch.tensor(
        [[REGION_PREFIX, REGION_BRIDGE, REGION_BRIDGE, REGION_BRIDGE]]
    )
    full_ce = byte_weighted_cross_entropy(logits, targets)
    span_ce = fim_span_byte_cross_entropy(logits, targets, regions)

    actual = byte_training_loss(
        logits,
        targets,
        target_region_ids=regions,
        fim_span_loss_weight=0.5,
    )

    assert torch.allclose(actual, full_ce + 0.5 * span_ce)


def test_shared_loss_computation_matches_independent_terms_and_gradients():
    torch.manual_seed(123)
    shape = (2, 3, 4, SEQ_EOS_ID + 1)
    targets = torch.tensor(
        [
            [[1, 2, 3, 4], [5, SEQ_EOS_ID, IGNORE_INDEX, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, SEQ_EOS_ID, 19], [20, IGNORE_INDEX, 22, 23]],
        ]
    )
    regions = torch.tensor(
        [
            [[REGION_PREFIX] * 4, [REGION_BRIDGE] * 4, [REGION_PREFIX] * 4],
            [[REGION_BRIDGE] * 4, [REGION_PREFIX] * 4, [REGION_BRIDGE] * 4],
        ]
    )
    optimized_logits = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    reference_logits = optimized_logits.detach().clone().requires_grad_(True)

    optimized = byte_training_loss_terms(
        optimized_logits,
        targets,
        eos_loss_weight=1.7,
        eos_aux_loss_weight=0.4,
        target_region_ids=regions,
        fim_span_loss_weight=0.6,
    )
    reference_full = byte_weighted_cross_entropy(
        reference_logits, targets, eos_loss_weight=1.7
    )
    reference_span = fim_span_byte_cross_entropy(reference_logits, targets, regions)
    reference_eos = balanced_eos_auxiliary_loss(reference_logits, targets)
    reference_objective = reference_full + 0.6 * reference_span + 0.4 * reference_eos

    assert torch.allclose(optimized["full_ce"], reference_full, atol=1e-6, rtol=1e-6)
    assert torch.allclose(optimized["fim_span_ce"], reference_span, atol=1e-6, rtol=1e-6)
    assert torch.allclose(optimized["eos_aux"], reference_eos, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        optimized["objective"], reference_objective, atol=1e-6, rtol=1e-6
    )

    optimized["objective"].backward()
    reference_objective.backward()
    assert torch.allclose(
        optimized_logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-6
    )


def test_shared_loss_computation_keeps_confident_wrong_eos_finite():
    logits = torch.full((1, 2, SEQ_EOS_ID + 1), -100.0)
    logits[0, 0, SEQ_EOS_ID] = 100.0
    logits.requires_grad_(True)
    targets = torch.tensor([[7, SEQ_EOS_ID]])

    terms = byte_training_loss_terms(
        logits,
        targets,
        eos_aux_loss_weight=1.0,
    )

    assert torch.isfinite(terms["eos_aux"])
    terms["objective"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_frame_type_span_diagnostics_reuse_training_ce_without_changing_objective():
    torch.manual_seed(5)
    logits = torch.randn(2, 4, SEQ_EOS_ID + 1)
    targets = torch.tensor([[1, 2, SEQ_EOS_ID, IGNORE_INDEX], [3, 4, 5, SEQ_EOS_ID]])
    regions = torch.tensor(
        [[REGION_BRIDGE] * 4, [REGION_BRIDGE] * 4]
    )
    baseline = byte_training_loss_terms(
        logits, targets, target_region_ids=regions, fim_span_loss_weight=1.0
    )
    typed = byte_training_loss_terms(
        logits,
        targets,
        target_region_ids=regions,
        fim_span_loss_weight=1.0,
        fim_frame_nal_type=torch.tensor([5, 1]),
    )

    assert torch.equal(typed["objective"], baseline["objective"])
    assert typed["fim_sample_count_idr"] == typed["fim_sample_count_p"] == 1
    assert typed["fim_span_target_count_idr"] == 2
    assert typed["fim_span_target_count_p"] == 3
    per_byte = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=IGNORE_INDEX, reduction="none").reshape(2, 4)
    assert torch.allclose(typed["fim_span_nll_sum_idr"], per_byte[0, :2].sum())
    assert torch.allclose(typed["fim_span_nll_sum_p"], per_byte[1, :3].sum())


def test_byte_only_ce_excludes_eos_and_control_logits():
    logits = torch.zeros(1, 1, SEQ_EOS_ID + 1, requires_grad=True)
    targets = torch.tensor([[7]])

    loss = byte_training_loss(logits, targets, ce_byte_only=True)
    loss.backward()

    assert torch.isclose(loss, torch.log(torch.tensor(256.0)))
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[..., 256:]) == 0


def test_byte_only_ce_rejects_control_targets():
    logits = torch.zeros(1, 1, SEQ_EOS_ID + 1)
    targets = torch.tensor([[SEQ_EOS_ID]])

    with pytest.raises(ValueError, match="raw byte"):
        byte_training_loss(logits, targets, ce_byte_only=True)
