import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler, \
                            WeightedRandomSampler, Sampler, Subset
from torchvision import datasets, transforms

from utils.augmentations import get_transforms
from utils.dataset import SimCLRDataset


def get_dataset(dataset_name, dataset_path,
                augment_both_views=True,
                batch_size=64, num_workers=8, 
                shuffle=True, **kwargs):
    
    multi_gpu = kwargs.pop('multi_gpu', False)
    world_size = kwargs.pop('world_size', 1)
    supervision = kwargs.pop('supervision', 'SSL')
    test = kwargs.pop('test', None)
    classes = kwargs.pop('classes', None)

    if dataset_name is None:
        # default to cifar10
        dataset_name = 'cifar10'

    if dataset_name == 'imagenet':
        dataset_download = load_dataset('timm/mini-imagenet')
        train_dataset_download = dataset_download['train']
        test_dataset_download = dataset_download['test']
        train_transforms, basic_transforms = get_transforms('imagenet')
        num_workers = 16
        
    elif dataset_name == 'cifar10':
        train_dataset_download = datasets.CIFAR10(root=dataset_path, train=True, 
                                          download=True, transform=None)
        test_dataset_download = datasets.CIFAR10(root=dataset_path, train=False,
                                       download=True, transform=None)      
        train_transforms, basic_transforms = get_transforms('cifar')

    elif dataset_name == 'cifar100':
        train_dataset_download = datasets.CIFAR100(root=dataset_path, train=True, 
                                         download=True, transform=None)
        test_dataset_download = datasets.CIFAR100(root=dataset_path, train=False,
                                                  download=True, transform=None) 
        train_transforms, basic_transforms = get_transforms('cifar')

    else:
        raise NotImplementedError(f'no known dataset named {dataset_name}')
    

    if classes is not None:
        train_dataset_download = filter_class_indices(train_dataset_download, classes)
        test_dataset_download = filter_class_indices(test_dataset_download, classes)
        
    train_dataset = SimCLRDataset(train_dataset_download, 
                                train_transforms, basic_transforms,
                                augment_both_views=augment_both_views,
                                dataset_name=dataset_name)
    
    # Adjust for multi-GPU
    shuffle = not multi_gpu  # Ensures DistributedSampler handles shuffling
    effective_batch_size = batch_size // world_size if multi_gpu else batch_size
    drop_last = multi_gpu  # Avoids uneven batches in DDP

    sampler = DistributedSampler(train_dataset, num_replicas=world_size) if multi_gpu else None
    
    if supervision == 'SSL' or supervision == 'CL':
        train_dataloader = DataLoader(train_dataset, batch_size=effective_batch_size,
                                    shuffle=shuffle, num_workers=num_workers,
                                    pin_memory=True, drop_last=drop_last, 
                                    sampler=sampler)
    elif supervision == 'SCL':
        print("Using stratified sampling")
        # Approximate stratified sampling
        labels = np.array(train_dataset_download.targets)
        sampler = ApproxStratifiedSampler(labels, batch_size)
        train_dataloader = DataLoader(train_dataset, batch_sampler=sampler,
                                      num_workers=num_workers, pin_memory=True,
                                      shuffle=False)
        
        # train_dataloader = DataLoader(train_dataset, batch_size=effective_batch_size,
        #                             shuffle=shuffle, num_workers=num_workers,
        #                             pin_memory=True, drop_last=drop_last, 
        #                             sampler=sampler)
    if test is not None:
        test_dataset = SimCLRDataset(test_dataset_download,
                                     train_transforms, basic_transforms,
                                     augment_both_views=False,
                                     dataset_name=dataset_name)
        test_dataloader = DataLoader(test_dataset, batch_size=effective_batch_size,
                                     shuffle=False, num_workers=num_workers,
                                     pin_memory=True)
        return train_dataset, train_dataloader, test_dataset, test_dataloader
    
    return train_dataset, train_dataloader

def filter_class_indices(dataset, classes):
    """
    Filter indices of a dataset for a subset of classes.
    """
    targets = np.array(dataset.targets)
    class_indices = np.where(np.isin(targets, classes))[0]
    return Subset(dataset, class_indices)

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

