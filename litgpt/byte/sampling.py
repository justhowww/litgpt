"""Training samplers for variable-length byte-stream windows."""

from __future__ import annotations

from torch import Generator, randperm
from torch.utils.data import Dataset, Sampler, Subset


def _dataset_sample_length(dataset: Dataset, index: int) -> int:
    """Resolve a cheap sample-length key through optional torch Subsets."""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    sample_length = getattr(dataset, "sample_length", None)
    if sample_length is None:
        raise TypeError(
            "Length bucketing requires a dataset with sample_length(index); "
            f"got {type(dataset).__name__}"
        )
    length = int(sample_length(index))
    if length < 1:
        raise ValueError(f"sample_length({index}) must be positive, got {length}")
    return length


class DistributedLengthBucketBatchSampler(Sampler[list[int]]):
    """Build rank-aligned, similarly sized local batches from shuffled pools.

    Every rank constructs the same shuffled global sample order. Within each
    finite pool, samples are sorted by byte length and divided into global
    microbatches. Global microbatch order is shuffled again, then each global
    batch is distributed strided across ranks. The strided split keeps the
    length distribution balanced between ranks while ensuring that every
    sample appears on exactly one rank.

    This is deliberately a *sortish* sampler rather than a global length sort:
    it reduces padding while retaining stochastic mixing throughout an epoch.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        local_batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        pool_size: int = 8192,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if local_batch_size < 1:
            raise ValueError("local_batch_size must be positive")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if not drop_last:
            raise ValueError("Distributed length bucketing currently requires drop_last=True")
        self.dataset = dataset
        self.local_batch_size = local_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.global_batch_size = local_batch_size * num_replicas
        if pool_size < self.global_batch_size:
            raise ValueError(
                "length bucket pool must contain at least one global microbatch: "
                f"pool_size={pool_size}, required>={self.global_batch_size}"
            )
        # Align pools so only the final incomplete global microbatch is dropped.
        self.pool_size = (pool_size // self.global_batch_size) * self.global_batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset) // self.global_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.epoch = int(state_dict.get("epoch", 0))

    def __iter__(self):
        generator = Generator().manual_seed(self.seed + self.epoch)
        usable = len(self) * self.global_batch_size
        permutation = randperm(len(self.dataset), generator=generator)

        for pool_start in range(0, usable, self.pool_size):
            pool_end = min(pool_start + self.pool_size, usable)
            pool = permutation[pool_start:pool_end].tolist()
            pool.sort(key=lambda index: _dataset_sample_length(self.dataset, index))
            num_global_batches = len(pool) // self.global_batch_size
            batch_order = randperm(num_global_batches, generator=generator).tolist()
            for batch_index in batch_order:
                lo = batch_index * self.global_batch_size
                global_batch = pool[lo : lo + self.global_batch_size]
                local_batch = global_batch[self.rank :: self.num_replicas]
                if len(local_batch) != self.local_batch_size:
                    raise RuntimeError("length-bucket rank split produced an incomplete batch")
                yield local_batch

        # CycleIterator creates a new iterator after exhaustion. Advance the
        # seed here so subsequent epochs reshuffle without external callbacks.
        self.epoch += 1
