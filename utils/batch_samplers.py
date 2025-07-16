import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Sampler


class ApproxStratifiedSampler(Sampler):
    """
    Sampler that draws samples with class-balanced probabilities.
    Does not guarantee perfect stratification but approximates it.

    Args:
        labels (Sequence[int]): Dataset labels.
        batch_size (int): Number of samples per batch.
        num_batches (int, optional): Number of batches to sample. Defaults to covering entire dataset.
    """
    def __init__(self, labels, batch_size, num_batches=None):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.indices = np.arange(len(labels))
        self.num_classes = len(np.unique(self.labels))

        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[self.labels]
        self.probabilities = sample_weights / sample_weights.sum()

        total_samples = num_batches * batch_size if num_batches else len(labels)
        self.num_batches = total_samples // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            yield np.random.choice(
                self.indices,
                size=self.batch_size,
                p=self.probabilities,
                replace=False
            ).tolist()

    def __len__(self):
        return self.num_batches


class DistributedStratifiedBatchSampler(Sampler):
    """
    Stratified batch sampler for DistributedDataParallel training.

    Ensures roughly balanced class distribution within each batch,
    and evenly splits batches across replicas.

    Args:
        labels (Sequence[int])
        batch_size (int)
        num_replicas (int, optional)
        rank (int, optional)
        drop_last (bool): Not used, included for compatibility
    """
    def __init__(self, labels, batch_size, num_replicas=None, rank=None, drop_last=False):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())

        self.num_samples = len(self.labels)
        total_batches = self.num_samples // self.batch_size
        self.num_batches_per_replica = total_batches // self.num_replicas
        self.total_batches = self.num_batches_per_replica * self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.epoch)
        shuffled_class_indices = {
            cls: rng.permutation(indices).tolist()
            for cls, indices in self.class_to_indices.items()
        }

        class_cursors = {cls: 0 for cls in self.classes}
        pooled_indices = []

        while len(pooled_indices) < self.total_batches * self.batch_size:
            for cls in self.classes:
                if class_cursors[cls] < len(shuffled_class_indices[cls]):
                    pooled_indices.append(shuffled_class_indices[cls][class_cursors[cls]])
                    class_cursors[cls] += 1
                    if len(pooled_indices) >= self.total_batches * self.batch_size:
                        break

        batches = [
            pooled_indices[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(self.total_batches)
        ]
        replica_batches = batches[self.rank::self.num_replicas]

        for batch in replica_batches:
            yield batch

    def __len__(self):
        return self.num_batches_per_replica
    

class DistributedStratifiedBatchSamplerSoftBalance(Sampler):
    """
    DDP-compatible stratified sampler that samples a fixed number of classes per batch,
    distributing samples uniformly across selected classes.

    Args:
        labels (Sequence[int])
        batch_size (int)
        num_classes_per_batch (int)
        num_replicas (int, optional)
        rank (int, optional)
        drop_last (bool): Not used
    """
    def __init__(self, labels, batch_size, num_classes_per_batch=5, num_replicas=None, rank=None, drop_last=False):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.drop_last = drop_last

        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.epoch = 0

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())

        total_batches = len(self.labels) // batch_size
        self.num_batches_per_replica = total_batches // self.num_replicas
        self.total_batches = self.num_batches_per_replica * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.epoch)

        class_indices = {
            cls: rng.permutation(idxs).tolist()
            for cls, idxs in self.class_to_indices.items()
        }
        class_cursors = {cls: 0 for cls in self.classes}
        pooled_batches = []

        for _ in range(self.total_batches):
            selected_classes = rng.choice(self.classes, size=self.num_classes_per_batch, replace=False)
            samples_per_class = self.batch_size // self.num_classes_per_batch
            batch = []

            for cls in selected_classes:
                idxs = class_indices[cls]
                cur = class_cursors[cls]

                if cur + samples_per_class > len(idxs):
                    idxs = rng.permutation(self.class_to_indices[cls]).tolist()
                    class_indices[cls] = idxs
                    cur = 0

                batch.extend(idxs[cur:cur + samples_per_class])
                class_cursors[cls] = cur + samples_per_class

            pooled_batches.append(batch)

        replica_batches = pooled_batches[self.rank::self.num_replicas]

        for batch in replica_batches:
            yield batch

    def __len__(self):
        return self.num_batches_per_replica