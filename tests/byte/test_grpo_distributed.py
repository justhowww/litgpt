from __future__ import annotations

import pytest
import torch
from torch import nn

from litgpt.byte.data import SEQ_EOS_ID
from litgpt.byte.grpo import (
    GRPOConfig,
    GRPOPreparationResult,
    _scored_preparation,
    grpo_update_direction_metrics,
    group_token_log_probabilities,
)
from litgpt.byte.grpo_context import OnlineGRPOContextSampler
from litgpt.byte.megabyte_inference import GeneratedCandidate
from litgpt.byte.training import ByteTrainingRuntime


class _LogitModel(nn.Module):
    def forward(self, **_kwargs):
        return torch.zeros((2, 3, 8, SEQ_EOS_ID + 1))


class _EOSBiasedLogitModel(nn.Module):
    def __init__(self, eos_logit: float):
        super().__init__()
        self.eos_logit = eos_logit

    def forward(self, **_kwargs):
        logits = torch.zeros((1, 1, 1, SEQ_EOS_ID + 1))
        logits[..., SEQ_EOS_ID] = self.eos_logit
        return logits


class _ForwardTrackingWrapper(nn.Module):
    """Looks unwrap-able like DDP while recording wrapper forward usage."""

    def __init__(self):
        super().__init__()
        self.module = _LogitModel()
        self.called = False

    def forward(self, **kwargs):
        self.called = True
        return self.module(**kwargs)


def test_grpo_candidate_scoring_does_not_bypass_distributed_wrapper():
    model = _ForwardTrackingWrapper()
    labels = torch.zeros((2, 3, 8), dtype=torch.long)

    gathered = group_token_log_probabilities(model, {}, labels)

    assert model.called
    assert gathered.shape == labels.shape


def test_ar_grpo_scoring_excludes_eos_like_fixed_frame_rollout():
    labels = torch.zeros((1, 1, 1), dtype=torch.long)
    low_eos = group_token_log_probabilities(
        _EOSBiasedLogitModel(-20), {}, labels, include_eos=False
    )
    high_eos = group_token_log_probabilities(
        _EOSBiasedLogitModel(20), {}, labels, include_eos=False
    )

    assert torch.allclose(low_eos, high_eos)
    expected = torch.full_like(low_eos, -torch.log(torch.tensor(256.0)))
    assert torch.allclose(low_eos, expected)


def test_learned_eos_grpo_scoring_retains_eos_action():
    labels = torch.zeros((1, 1, 1), dtype=torch.long)
    low_eos = group_token_log_probabilities(
        _EOSBiasedLogitModel(-20), {}, labels, include_eos=True
    )
    high_eos = group_token_log_probabilities(
        _EOSBiasedLogitModel(20), {}, labels, include_eos=True
    )

    assert high_eos < low_eos


def test_distributed_context_positions_are_disjoint_within_an_update():
    config = GRPOConfig(
        interval=1,
        group_size=2,
        context_sampling="online",
        context_seed=123,
    )
    sampler = OnlineGRPOContextSampler(
        dataset=object(),  # dataset_index() only needs the configured index set
        indices=tuple(range(32)),
        config=config,
    )

    first_positions = {
        sampler.dataset_index("fim", rank, stride=4) for rank in range(4)
    }
    retry_positions = {
        sampler.dataset_index("fim", rank, attempt=1, stride=4) for rank in range(4)
    }

    assert len(first_positions) == 4
    assert len(retry_positions) == 4
    assert first_positions.isdisjoint(retry_positions)


class _ReadinessFabric:
    world_size = 4
    device = torch.device("cpu")

    def __init__(self, reduced_value: float):
        self.reduced_value = reduced_value

    def all_reduce(self, tensor, reduce_op):
        assert reduce_op == "min"
        return tensor.new_tensor(self.reduced_value)


def test_grpo_skips_collectively_when_any_rank_is_not_ready():
    all_ready = _ReadinessFabric(1.0)
    peer_failed = _ReadinessFabric(0.0)

    assert ByteTrainingRuntime._all_ranks_ready(all_ready, True)
    assert not ByteTrainingRuntime._all_ranks_ready(peer_failed, True)


def test_gradient_l2_norm_detects_nonzero_backward_signal():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()

    assert ByteTrainingRuntime.gradient_l2_norm(model) > 0


def test_grpo_direction_metrics_accept_reward_aligned_update():
    before = torch.tensor([-2.0, -2.0, -2.0])
    after = torch.tensor([-2.2, -2.0, -1.8])
    rewards = torch.tensor([-1.0, 0.0, 1.0])
    advantages = torch.tensor([-1.0, 0.0, 1.0])

    metrics = grpo_update_direction_metrics(before, after, rewards, advantages)

    assert metrics["policy_score_delta"] > 0
    assert metrics["advantage_delta_correlation"] > 0
    assert metrics["pairwise_improved_fraction"] == 1.0


def test_grpo_direction_metrics_reject_reward_reversed_update():
    before = torch.tensor([-2.0, -2.0, -2.0])
    after = torch.tensor([-1.8, -2.0, -2.2])
    rewards = torch.tensor([-1.0, 0.0, 1.0])
    advantages = torch.tensor([-1.0, 0.0, 1.0])

    metrics = grpo_update_direction_metrics(before, after, rewards, advantages)

    assert metrics["policy_score_delta"] < 0
    assert metrics["advantage_delta_correlation"] < 0
    assert metrics["pairwise_improved_fraction"] == 0.0


def test_grpo_direction_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="equal lengths"):
        grpo_update_direction_metrics(
            torch.zeros(2), torch.zeros(3), torch.zeros(2), torch.zeros(2)
        )


def test_zero_variance_group_keeps_candidates_with_zero_advantages():
    candidates = [
        GeneratedCandidate(data=bytes([i]), stopped=False) for i in range(4)
    ]

    result = _scored_preparation(
        sample=object(),
        task="fim",
        candidates=candidates,
        results=[(-1.0, False, None)] * 4,
        device=torch.device("cpu"),
    )

    assert result.status == "zero_reward_variance"
    assert not result.has_policy_signal
    assert result.prepared is not None
    assert torch.equal(
        result.prepared.advantages, torch.zeros_like(result.prepared.advantages)
    )
    assert result.candidate_count == 4
    assert result.stopped_count == 0
    assert result.decoded_count == 0


class _PreparationSumFabric:
    world_size = 4
    device = torch.device("cpu")

    def all_reduce(self, tensor, reduce_op):
        assert reduce_op == "sum"
        # Three peer ranks are ready and have policy signal.  Each contributes
        # four stopped candidates, three of which decode.
        peer_totals = tensor.new_tensor(
            [3, 3, 0, 0, 0, 12, 12, 0, 9, 3]
        )
        return tensor + peer_totals


def test_distributed_preparation_metrics_preserve_zero_variance_rank():
    local = GRPOPreparationResult(
        status="zero_reward_variance",
        prepared=object(),
        candidate_count=4,
        stopped_count=0,
        decoded_count=0,
        reward_std=0.0,
        has_policy_signal=False,
    )

    metrics = ByteTrainingRuntime._distributed_grpo_preparation_metrics(
        _PreparationSumFabric(), local
    )

    assert metrics["grpo/prepared_contexts"] == 4
    assert metrics["grpo/contexts_with_policy_signal"] == 3
    assert metrics["grpo/contexts_zero_reward_variance"] == 1
    assert metrics["grpo/candidates_total"] == 16
    assert metrics["grpo/candidates_stopped"] == 12
    assert metrics["grpo/candidates_not_stopped"] == 4
    assert metrics["grpo/candidates_decoded"] == 9
    assert metrics["grpo/candidates_decode_failed_after_stop"] == 3
