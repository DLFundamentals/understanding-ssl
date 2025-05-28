import numpy as np
from collections import defaultdict
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Sampler, DistributedSampler, \
                            Subset, ConcatDataset
from torchvision import datasets, transforms

from utils.augmentations import get_transforms
from utils.batch_samplers import (ApproxStratifiedSampler,
                                  DistributedStratifiedBatchSampler,
                                  DistributedStratifiedBatchSamplerSoftBalance)
from utils.dataset import SimCLRDataset


def get_dataset(dataset_name, dataset_path,
                augment_both_views=True,
                batch_size=64, num_workers=8, 
                shuffle=True, **kwargs):
    
    multi_gpu = kwargs.get('multi_gpu', False)
    world_size = kwargs.get('world_size', 1)
    supervision = kwargs.get('supervision', 'SSL')
    test = kwargs.get('test', None)
    classes = kwargs.get('classes', None)
    
    # Load and transform raw datasets
    raw_train, raw_test, labels_train, labels_test = _load_raw_datasets(dataset_name, dataset_path)
    train_transforms, basic_transforms = _get_transforms(dataset_name)
    
    # Filter specific classes (optional)
    if classes is not None:
        raw_train, labels = filter_class_indices(raw_train, classes, labels)
        raw_test, labels_test = filter_class_indices(raw_test, classes, labels_test)
        
    train_dataset = SimCLRDataset(raw_train, 
                                train_transforms, basic_transforms,
                                augment_both_views=augment_both_views,
                                dataset_name=dataset_name)
    
    # Adjust for multi-GPU
    shuffle = not multi_gpu  # DistributedSampler handles shuffling
    effective_batch_size = batch_size // world_size if multi_gpu else batch_size
    drop_last = multi_gpu  # Avoids uneven batches in DDP

    # Build train dataloader
    train_dataloader = _build_dataloader(train_dataset, supervision, dataset_name, labels_train,
                                     batch_size=effective_batch_size, num_workers=num_workers,
                                     multi_gpu=multi_gpu, world_size=world_size, drop_last=drop_last)

    # Build test loader if needed
        
    if test is not None:
        test_dataset = SimCLRDataset(raw_test,
                                     train_transforms, basic_transforms,
                                     augment_both_views=False,
                                     dataset_name=dataset_name)
        test_dataloader = DataLoader(test_dataset, batch_size=effective_batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     pin_memory=True)
        return train_dataset, train_dataloader, test_dataset, test_dataloader, labels, labels_test
    
    return train_dataset, train_dataloader

def filter_class_indices(dataset, classes, labels):
    """
    Filter indices of a dataset for a subset of classes.
    """
    if labels is None:
        labels = np.array(dataset.targets)
    class_indices = np.where(np.isin(labels, classes))[0]
    labels = labels[class_indices]
    class_indices = list(map(int, class_indices))
    return Subset(dataset, class_indices), labels

def _load_raw_datasets(dataset_name, dataset_path):
    if dataset_name == 'imagenet':
        ds = load_dataset('timm/mini-imagenet')
        return ds['train'], ds['test'], np.array(ds['train']['label']), np.array(ds['test']['label'])

    elif dataset_name == 'cifar10':
        train = datasets.CIFAR10(root=dataset_path, train=True, download=True)
        test = datasets.CIFAR10(root=dataset_path, train=False, download=True)
        return train, test, np.array(train.targets), np.array(test.targets)

    elif dataset_name == 'cifar100':
        train = datasets.CIFAR100(root=dataset_path, train=True, download=True)
        test = datasets.CIFAR100(root=dataset_path, train=False, download=True)
        return train, test, np.array(train.targets), np.array(test.targets)

    elif dataset_name == 'svhn':
        train = datasets.SVHN(root=dataset_path, split='train', download=True)
        extra = datasets.SVHN(root=dataset_path, split='extra', download=True)
        test = datasets.SVHN(root=dataset_path, split='test', download=True)
        labels_train = np.concatenate([train.labels, extra.labels])
        train = ConcatDataset([train, extra])
        return train, test, labels_train, np.array(test.labels)

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

def _get_transforms(dataset_name):
    return get_transforms('cifar' if 'cifar' in dataset_name else dataset_name)

def _build_sampler(supervision, dataset_name, labels, batch_size, multi_gpu, world_size):
    if supervision != 'SCL':
        return None

    if multi_gpu:
        rank = torch.distributed.get_rank()
        if dataset_name == 'svhn':
            return DistributedStratifiedBatchSamplerSoftBalance(labels, batch_size, num_replicas=world_size, rank=rank)
        return DistributedStratifiedBatchSampler(labels, batch_size, num_replicas=world_size, rank=rank)
    
    return ApproxStratifiedSampler(labels, batch_size)

def _build_dataloader(dataset, supervision, dataset_name, labels,
                      batch_size, num_workers, multi_gpu, world_size, drop_last):

    sampler = _build_sampler(supervision, dataset_name, labels, batch_size, multi_gpu, world_size)

    if isinstance(sampler, Sampler):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=not multi_gpu, sampler=None, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=True)