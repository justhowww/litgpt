import torch
from torch.utils.data import Dataset, Subset

from litgpt.byte.sampling import DistributedLengthBucketBatchSampler


class _LengthDataset(Dataset):
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> int:
        return self.lengths[index]

    def sample_length(self, index: int) -> int:
        return self.lengths[index]


def _rank_batches(dataset, *, epoch: int = 0):
    samplers = [
        DistributedLengthBucketBatchSampler(
            dataset,
            local_batch_size=2,
            num_replicas=2,
            rank=rank,
            pool_size=16,
            seed=7,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(epoch)
    return [list(sampler) for sampler in samplers]


def test_length_bucket_sampler_partitions_global_batches_across_ranks():
    dataset = _LengthDataset(list(range(1, 23)))
    rank_batches = _rank_batches(dataset)

    assert len(rank_batches[0]) == len(dataset) // 4
    assert len(rank_batches[0]) == len(rank_batches[1])
    assert all(len(batch) == 2 for batches in rank_batches for batch in batches)

    selected = [
        index
        for step in range(len(rank_batches[0]))
        for rank in range(2)
        for index in rank_batches[rank][step]
    ]
    expected = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7))[
        : len(dataset) // 4 * 4
    ].tolist()
    assert sorted(selected) == sorted(expected)
    assert len(selected) == len(set(selected))


def test_length_bucket_sampler_reduces_padding_and_balances_ranks():
    lengths = [1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400, 1000, 2000, 3000, 4000]
    dataset = _LengthDataset(lengths)
    rank_batches = _rank_batches(dataset)

    bucketed_padding = 0
    for step in range(len(rank_batches[0])):
        rank_maxima = []
        for rank in range(2):
            batch_lengths = [lengths[index] for index in rank_batches[rank][step]]
            rank_maxima.append(max(batch_lengths))
            bucketed_padding += len(batch_lengths) * max(batch_lengths) - sum(batch_lengths)
        # Strided rank assignment should keep both ranks in the same length band.
        assert max(rank_maxima) / min(rank_maxima) <= 2

    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(7)).tolist()
    random_padding = 0
    for lo in range(0, len(permutation), 4):
        global_batch = permutation[lo : lo + 4]
        for rank in range(2):
            batch_lengths = [lengths[index] for index in global_batch[rank::2]]
            random_padding += len(batch_lengths) * max(batch_lengths) - sum(batch_lengths)

    assert bucketed_padding < random_padding


def test_length_bucket_sampler_is_deterministic_and_changes_by_epoch():
    dataset = _LengthDataset(list(range(1, 65)))
    assert _rank_batches(dataset, epoch=3) == _rank_batches(dataset, epoch=3)
    assert _rank_batches(dataset, epoch=3) != _rank_batches(dataset, epoch=4)


def test_length_bucket_sampler_resolves_subset_indices():
    base = _LengthDataset([100, 1, 80, 2, 60, 3, 40, 4])
    subset = Subset(base, [1, 3, 5, 7])
    sampler = DistributedLengthBucketBatchSampler(
        subset, local_batch_size=2, pool_size=4, seed=1
    )

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
