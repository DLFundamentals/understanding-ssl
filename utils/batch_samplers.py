import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Sampler

class ApproxStratifiedSampler(Sampler):
    def __init__(self, labels, batch_size, num_batches=None):
        """
        labels: List or tensor of dataset labels
        batch_size: Number of samples per batch
        num_batches: Total batches (default: use full dataset)
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = len(np.unique(labels))
        self.indices = np.arange(len(labels))

        # Compute class weights (inverse of class frequency)
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[self.labels]

        # Compute number of batches
        total_samples = num_batches * batch_size if num_batches else len(labels)
        self.num_batches = total_samples // batch_size

        # Weighted random sampling for rough balance
        self.probabilities = sample_weights / sample_weights.sum()

    def __iter__(self):
        """Yield batches with approximately balanced class distribution."""
        for _ in range(self.num_batches):
            batch_indices = np.random.choice(self.indices, size=self.batch_size, p=self.probabilities, replace=False)
            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches

class DistributedStratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_replicas=None, rank=None, drop_last=False):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(self.labels)

        # Group samples by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

        # Compute total number of batches per replica
        total_batches = (self.num_samples // self.batch_size)
        self.num_batches_per_replica = total_batches // self.num_replicas
        self.total_batches = self.num_batches_per_replica * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.epoch)

        # Shuffle indices within each class
        shuffled_class_indices = {
            cls: rng.permutation(indices).tolist()
            for cls, indices in self.class_to_indices.items()
        }

        # Interleave across classes
        pooled_indices = []
        class_cursors = {cls: 0 for cls in self.classes}

        while len(pooled_indices) < self.total_batches * self.batch_size:
            for cls in self.classes:
                if class_cursors[cls] < len(shuffled_class_indices[cls]):
                    pooled_indices.append(shuffled_class_indices[cls][class_cursors[cls]])
                    class_cursors[cls] += 1
                    if len(pooled_indices) >= self.total_batches * self.batch_size:
                        break

        # Partition across replicas
        batches = [
            pooled_indices[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(self.total_batches)
        ]
        # Select only the portion for this rank
        replica_batches = batches[self.rank::self.num_replicas]

        for batch in replica_batches:
            yield batch

    def __len__(self):
        return self.num_batches_per_replica
    
class DistributedStratifiedOversamplingBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_replicas=None, rank=None, drop_last=False):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # Group samples by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.max_class_size = max(len(idxs) for idxs in self.class_to_indices.values())
        self.num_samples = self.max_class_size * len(self.classes)

        # Total batches & per-replica batches
        total_batches = self.num_samples // self.batch_size
        self.num_batches_per_replica = total_batches // self.num_replicas
        self.total_batches = self.num_batches_per_replica * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.epoch)

        # Oversample to balance classes
        oversampled_class_indices = {}
        for cls, indices in self.class_to_indices.items():
            if len(indices) < self.max_class_size:
                sampled = rng.choice(indices, self.max_class_size, replace=True).tolist()
            else:
                sampled = rng.permutation(indices).tolist()
            oversampled_class_indices[cls] = sampled

        # Flatten: interleave class samples to maintain balance
        interleaved_indices = []
        for i in range(self.max_class_size):
            for cls in self.classes:
                interleaved_indices.append(oversampled_class_indices[cls][i])

        # Slice into batches
        batches = [
            interleaved_indices[i * self.batch_size: (i + 1) * self.batch_size]
            for i in range(self.total_batches)
        ]

        # Assign batches to replicas
        replica_batches = batches[self.rank::self.num_replicas]

        for batch in replica_batches:
            yield batch

    def __len__(self):
        return self.num_batches_per_replica
    
class DistributedStratifiedBatchSamplerSoftBalance(Sampler):
    def __init__(self, labels, batch_size, num_classes_per_batch=5, num_replicas=None, rank=None, drop_last=False):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.drop_last = drop_last

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # Group samples by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

        # Estimate how many batches we can get
        est_total_batches = len(self.labels) // batch_size
        self.num_batches_per_replica = est_total_batches // self.num_replicas
        self.total_batches = self.num_batches_per_replica * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(seed=self.epoch)

        # Shuffle indices within each class
        class_indices = {
            cls: rng.permutation(idxs).tolist()
            for cls, idxs in self.class_to_indices.items()
        }

        # Track cursor per class
        class_cursors = {cls: 0 for cls in self.classes}

        pooled_batches = []

        for _ in range(self.total_batches):
            # Sample subset of classes for this batch
            selected_classes = rng.choice(self.classes, size=self.num_classes_per_batch, replace=False)
            samples_per_class = self.batch_size // self.num_classes_per_batch
            batch = []

            for cls in selected_classes:
                idxs = class_indices[cls]
                cur = class_cursors[cls]

                # Replenish if exhausted
                if cur + samples_per_class > len(idxs):
                    idxs = rng.permutation(self.class_to_indices[cls]).tolist()
                    class_indices[cls] = idxs
                    cur = 0

                batch.extend(idxs[cur:cur + samples_per_class])
                class_cursors[cls] = cur + samples_per_class

            pooled_batches.append(batch)

        # Shard across DDP replicas
        replica_batches = pooled_batches[self.rank::self.num_replicas]

        for batch in replica_batches:
            yield batch

    def __len__(self):
        return self.num_batches_per_replica