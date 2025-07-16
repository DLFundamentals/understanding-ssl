import numpy as np
import torch
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader, Sampler, DistributedSampler, Subset, ConcatDataset
from torchvision import datasets, transforms

from utils.augmentations import get_transforms
from utils.batch_samplers import (
    ApproxStratifiedSampler,
    DistributedStratifiedBatchSampler,
    DistributedStratifiedBatchSamplerSoftBalance,
)
from utils.dataset import SimCLRDataset


def get_dataset(
    dataset_name,
    dataset_path,
    augment_both_views=True,
    batch_size=64,
    num_workers=8,
    shuffle=True,
    **kwargs
):
    """
    Returns the dataset and dataloader(s) for training (and optionally testing),
    optionally filtered to specific classes and configured for multi-GPU DDP.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'cifar10', 'svhn').
        dataset_path (str): Path to store/download datasets.
        augment_both_views (bool): Whether to apply augmentations to both SimCLR views.
        batch_size (int): Batch size per process.
        num_workers (int): Dataloader worker processes.
        shuffle (bool): If True, enable shuffling (ignored for DDP).
        **kwargs:
            multi_gpu (bool)
            world_size (int)
            supervision (str): 'SSL' | 'SCL'
            test (bool): Whether to return test dataloader.
            classes (list): Optional list of class labels to include.

    Returns:
        train_dataset, train_loader, [test_dataset, test_loader, labels_train, labels_test]
    """
    multi_gpu = kwargs.get("multi_gpu", False)
    world_size = kwargs.get("world_size", 1)
    supervision = kwargs.get("supervision", "SSL")
    return_test = kwargs.get("test", False)
    selected_classes = kwargs.get("classes", None)

    # Load datasets and labels
    raw_train, raw_test, labels_train, labels_test = _load_raw_datasets(dataset_name, dataset_path)
    train_tfms, basic_tfms = _get_transforms(dataset_name)

    # Filter for selected classes if specified
    if selected_classes is not None:
        raw_train, labels_train = filter_class_indices(raw_train, selected_classes, labels_train)
        raw_test, labels_test = filter_class_indices(raw_test, selected_classes, labels_test)

    # Wrap with SimCLR-style augmentation dataset
    train_dataset = SimCLRDataset(
        raw_train, train_tfms, basic_tfms,
        augment_both_views=augment_both_views,
        dataset_name=dataset_name,
    )

    # Adjust for DDP
    effective_bs = batch_size // world_size if multi_gpu else batch_size
    drop_last = multi_gpu  # avoid uneven batches in DDP
    shuffle = not multi_gpu  # handled via sampler in DDP

    train_loader = _build_dataloader(
        train_dataset, supervision, dataset_name, labels_train,
        batch_size=effective_bs, num_workers=num_workers,
        multi_gpu=multi_gpu, world_size=world_size, drop_last=drop_last
    )

    if return_test:
        test_dataset = SimCLRDataset(
            raw_test, train_tfms, basic_tfms,
            augment_both_views=False, dataset_name=dataset_name
        )
        test_loader = DataLoader(
            test_dataset, batch_size=effective_bs, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        return train_dataset, train_loader, test_dataset, test_loader, labels_train, labels_test

    return train_dataset, train_loader


def filter_class_indices(dataset, classes, labels):
    """
    Filters dataset to retain only the specified classes.

    Args:
        dataset (Dataset or ConcatDataset)
        classes (list[int])
        labels (np.ndarray)

    Returns:
        Subset: Filtered dataset
        np.ndarray: Filtered labels
    """
    if labels is None:
        labels = np.array(dataset.targets)
    class_indices = np.where(np.isin(labels, classes))[0]
    return Subset(dataset, class_indices.tolist()), labels[class_indices]


def _load_raw_datasets(dataset_name, dataset_path):
    """
    Loads datasets and extracts their labels.

    Returns:
        train_dataset, test_dataset, labels_train, labels_test
    """
    if dataset_name == "imagenet":
        ds = load_dataset("timm/mini-imagenet")
        return ds["train"], ds["test"], np.array(ds["train"]["label"]), np.array(ds["test"]["label"])

    elif dataset_name == "cifar10":
        train = datasets.CIFAR10(root=dataset_path, train=True, download=True)
        test = datasets.CIFAR10(root=dataset_path, train=False, download=True)
        return train, test, np.array(train.targets), np.array(test.targets)

    elif dataset_name == "cifar100":
        train = datasets.CIFAR100(root=dataset_path, train=True, download=True)
        test = datasets.CIFAR100(root=dataset_path, train=False, download=True)
        return train, test, np.array(train.targets), np.array(test.targets)

    elif dataset_name == "svhn":
        train = datasets.SVHN(root=dataset_path, split="train", download=True)
        extra = datasets.SVHN(root=dataset_path, split="extra", download=True)
        test = datasets.SVHN(root=dataset_path, split="test", download=True)

        labels_train = np.concatenate([train.labels, extra.labels])
        combined_train = ConcatDataset([train, extra])
        return combined_train, test, labels_train, np.array(test.labels)

    raise NotImplementedError(f"Unsupported dataset: {dataset_name}")


def _get_transforms(dataset_name):
    """
    Returns the data augmentation and evaluation transforms for the dataset.
    """
    return get_transforms("cifar" if "cifar" in dataset_name else dataset_name)


def _build_sampler(supervision, dataset_name, labels, batch_size, multi_gpu, world_size):
    """
    Chooses a suitable batch sampler based on supervision mode and dataset.

    Returns:
        Sampler or None
    """
    if supervision != "SCL":
        return None

    if multi_gpu:
        rank = torch.distributed.get_rank()
        if dataset_name == "svhn":
            return DistributedStratifiedBatchSamplerSoftBalance(
                labels, batch_size, num_replicas=world_size, rank=rank
            )
        return DistributedStratifiedBatchSampler(
            labels, batch_size, num_replicas=world_size, rank=rank
        )

    return ApproxStratifiedSampler(labels, batch_size)


def _build_dataloader(
    dataset, supervision, dataset_name, labels,
    batch_size, num_workers, multi_gpu, world_size, drop_last
):
    """
    Builds a DataLoader with optional custom or distributed batch samplers.
    """
    if supervision == "SSL":
        sampler = DistributedSampler(dataset, shuffle=True) if multi_gpu else None
    else:
        sampler = _build_sampler(supervision, dataset_name, labels, batch_size, multi_gpu, world_size)

    if isinstance(sampler, Sampler) and not isinstance(sampler, DistributedSampler):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True
    )
