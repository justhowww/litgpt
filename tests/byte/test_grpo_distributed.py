from __future__ import annotations

import torch
from torch import nn

from litgpt.byte.data import SEQ_EOS_ID
from litgpt.byte.grpo import GRPOConfig, group_token_log_probabilities
from litgpt.byte.grpo_context import OnlineGRPOContextSampler
from litgpt.byte.training import ByteTrainingRuntime


class _LogitModel(nn.Module):
    def forward(self, **_kwargs):
        return torch.zeros((2, 3, 8, SEQ_EOS_ID + 1))


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
